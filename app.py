import os
import uuid
import threading
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from fairsquare.ingestion import GameIngestion
from fairsquare.analysis import StockfishAnalyzer
from phase2_features import build_move_features, win_prob
from phase4_game_model import build_game_level_features

app = Flask(__name__, template_folder="templates", static_folder="static")

TASKS = {}

DEFAULT_STOCKFISH_PATH = os.path.join("bin", "stockfish-engine")
if not os.path.exists(DEFAULT_STOCKFISH_PATH):
    DEFAULT_STOCKFISH_PATH = "stockfish"

def get_models():
    if not os.path.exists("models/move_model.joblib") or not os.path.exists("models/game_model.joblib"):
        from train_and_save_models import train_and_save_pipeline_models
        train_and_save_pipeline_models()
    move_clf = joblib.load("models/move_model.joblib")
    game_clf = joblib.load("models/game_model.joblib")
    return move_clf, game_clf

def run_chesscom_analysis_task(task_id: str, username: str, max_games: int, depth: int):
    try:
        TASKS[task_id] = {
            "status": "running",
            "progress_pct": 10,
            "progress_message": f"Fetching games for '{username}' from Chess.com API..."
        }

        games = GameIngestion.fetch_chesscom_user_games(username, max_games=max_games)
        if not games:
            TASKS[task_id] = {
                "status": "failed",
                "error": f"No recent blitz or rapid games found for Chess.com user '{username}'."
            }
            return

        move_clf, game_clf = get_models()
        analyzer = StockfishAnalyzer(engine_path=DEFAULT_STOCKFISH_PATH, depth=depth)

        game_breakdowns = []
        all_game_scores = []
        total_games = len(games)

        for idx, g in enumerate(games):
            pct = 15 + int(((idx + 1) / total_games) * 75)
            TASKS[task_id] = {
                "status": "running",
                "progress_pct": pct,
                "progress_message": f"Analyzing game {idx+1} of {total_games} with Stockfish (Depth {depth})..."
            }

            records = analyzer.analyze_game(g["game_obj"], target_color=g["target_color"])
            if not records:
                continue

            formatted_records = []
            for r in records:
                formatted_records.append({
                    "game_id": g["id"],
                    "player_id": 1,
                    "player_elo": g["target_elo"],
                    "opponent_elo": 1500,
                    "half_move": r["ply"],
                    "move_player": r["played_uci"],
                    "move_stockfish_1": r["best_uci"],
                    "move_stockfish_9": r["best_uci"],
                    "move_stockfish_15": r["best_uci"],
                    "eval_before": r["best_cp"],
                    "eval_after": r["played_cp"],
                    "centipawn_loss": max(0.0, float(r["best_cp"] - r["played_cp"])),
                    "normalized_centipawn_loss": max(0.0, float(win_prob(r["best_cp"]) - win_prob(r["played_cp"]))),
                    "is_used": True,
                    "is_cheating_player_game": 0
                })

            if not formatted_records:
                continue

            df_moves = pd.DataFrame(formatted_records)
            X_move = build_move_features(df_moves)

            move_probs = move_clf.predict_proba(X_move)[:, 1]
            df_game = build_game_level_features(df_moves, move_probs)

            game_feature_cols = [
                "mean_move_score", "max_move_score", "top10_mean_score",
                "max_sf_streak", "streaks_len3_plus_count", "critical_match_rate",
                "game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"
            ]

            X_game = df_game[game_feature_cols]
            game_prob = float(game_clf.predict_proba(X_game)[0, 1])
            all_game_scores.append(game_prob)

            row_game = df_game.iloc[0]
            game_breakdowns.append({
                "game_id": g["id"],
                "event": g["event"],
                "white": g["white"],
                "black": g["black"],
                "moves_analyzed": len(records),
                "avg_cpl": round(float(row_game["game_avg_cpl"]), 1),
                "sf_match_pct": round(float(row_game["game_sf15_match_rate"] * 100.0), 1),
                "max_streak": int(row_game["max_sf_streak"]),
                "streaks_count": int(row_game["streaks_len3_plus_count"]),
                "critical_match_pct": round(float(row_game["critical_match_rate"] * 100.0), 1),
                "suspicion_prob": round(float(game_prob * 100.0), 1)
            })

        if not all_game_scores:
            TASKS[task_id] = {
                "status": "failed",
                "error": "No valid move sequences from move 21 onward were found in the fetched games."
            }
            return

        avg_suspicion_prob = float(np.mean(all_game_scores))
        suspicion_percentage = round(avg_suspicion_prob * 100.0, 1)

        if suspicion_percentage >= 75.0:
            risk_level = "HIGH SUSPICION"
        elif suspicion_percentage >= 45.0:
            risk_level = "MODERATE SUSPICION"
        else:
            risk_level = "LOW / NORMAL HUMAN PLAY"

        confidence_pct = min(100.0, max(25.0, round((len(game_breakdowns) / 20.0) * 100.0, 1)))

        factors = []
        avg_streak = float(np.mean([g["max_streak"] for g in game_breakdowns]))
        avg_match = float(np.mean([g["sf_match_pct"] for g in game_breakdowns]))
        avg_crit = float(np.mean([g["critical_match_pct"] for g in game_breakdowns]))

        if avg_streak >= 5.0:
            factors.append({
                "feature_label": "Consecutive Engine Streaks",
                "description": f"Player exhibits an average peak engine move streak of {avg_streak:.1f} consecutive moves per game."
            })
        if avg_match >= 55.0:
            factors.append({
                "feature_label": "Engine Choice Match Rate",
                "description": f"Top Stockfish choice match rate is {avg_match:.1f}%, significantly above normal human variance."
            })
        if avg_crit >= 60.0:
            factors.append({
                "feature_label": "Critical Position Accuracy",
                "description": f"Player matches top engine choices in {avg_crit:.1f}% of critical tactical swing positions."
            })

        if not factors:
            factors.append({
                "feature_label": "Human Play Profile",
                "description": "Engine match rates and move streaks fall within expected human rating variance."
            })

        report = {
            "player_name": username,
            "total_games_analyzed": len(game_breakdowns),
            "player_elo": int(games[0]["target_elo"]) if games else 1500,
            "suspicion_score_pct": suspicion_percentage,
            "risk_level": risk_level,
            "confidence_pct": confidence_pct,
            "top_contributing_factors": factors,
            "game_breakdown": game_breakdowns,
            "legal_notice": "NOTICE: This output represents statistical likelihood based on position criticality, Stockfish streaks, and move accuracy models. It is intended for research and educational purposes and does not constitute a legal or definitive verdict."
        }

        TASKS[task_id] = {
            "status": "completed",
            "progress_pct": 100,
            "result": report
        }
    except Exception as e:
        TASKS[task_id] = {
            "status": "failed",
            "error": str(e)
        }

def run_pgn_analysis_task(task_id: str, pgn_text: str, target_color: str, player_name: str, depth: int):
    try:
        TASKS[task_id] = {
            "status": "running",
            "progress_pct": 10,
            "progress_message": "Parsing PGN game text..."
        }

        games = GameIngestion.parse_pgn_file(pgn_text)
        if not games:
            TASKS[task_id] = {
                "status": "failed",
                "error": "No valid PGN games found (minimum 14 moves required per game)."
            }
            return

        move_clf, game_clf = get_models()
        analyzer = StockfishAnalyzer(engine_path=DEFAULT_STOCKFISH_PATH, depth=depth)

        game_breakdowns = []
        all_game_scores = []
        total_games = len(games)

        for idx, g in enumerate(games):
            pct = 15 + int(((idx + 1) / total_games) * 75)
            TASKS[task_id] = {
                "status": "running",
                "progress_pct": pct,
                "progress_message": f"Analyzing PGN game {idx+1} of {total_games} for {player_name}..."
            }

            # Target selected color ('white', 'black', or 'both')
            records = analyzer.analyze_game(g["game_obj"], target_color=target_color)
            if not records:
                continue

            target_elo = g["white_elo"] if target_color == "white" else g["black_elo"]
            opp_elo = g["black_elo"] if target_color == "white" else g["white_elo"]

            formatted_records = []
            for r in records:
                formatted_records.append({
                    "game_id": f"PGN Game {idx+1}",
                    "player_id": 1,
                    "player_elo": target_elo,
                    "opponent_elo": opp_elo,
                    "half_move": r["ply"],
                    "move_player": r["played_uci"],
                    "move_stockfish_1": r["best_uci"],
                    "move_stockfish_9": r["best_uci"],
                    "move_stockfish_15": r["best_uci"],
                    "eval_before": r["best_cp"],
                    "eval_after": r["played_cp"],
                    "centipawn_loss": max(0.0, float(r["best_cp"] - r["played_cp"])),
                    "normalized_centipawn_loss": max(0.0, float(win_prob(r["best_cp"]) - win_prob(r["played_cp"]))),
                    "is_used": True,
                    "is_cheating_player_game": 0
                })

            if not formatted_records:
                continue

            df_moves = pd.DataFrame(formatted_records)
            X_move = build_move_features(df_moves)

            move_probs = move_clf.predict_proba(X_move)[:, 1]
            df_game = build_game_level_features(df_moves, move_probs)

            game_feature_cols = [
                "mean_move_score", "max_move_score", "top10_mean_score",
                "max_sf_streak", "streaks_len3_plus_count", "critical_match_rate",
                "game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"
            ]

            X_game = df_game[game_feature_cols]
            game_prob = float(game_clf.predict_proba(X_game)[0, 1])
            all_game_scores.append(game_prob)

            row_game = df_game.iloc[0]
            game_breakdowns.append({
                "game_id": f"PGN Game {idx+1}",
                "event": g["event"],
                "white": g["white"],
                "black": g["black"],
                "moves_analyzed": len(records),
                "avg_cpl": round(float(row_game["game_avg_cpl"]), 1),
                "sf_match_pct": round(float(row_game["game_sf15_match_rate"] * 100.0), 1),
                "max_streak": int(row_game["max_sf_streak"]),
                "streaks_count": int(row_game["streaks_len3_plus_count"]),
                "critical_match_pct": round(float(row_game["critical_match_rate"] * 100.0), 1),
                "suspicion_prob": round(float(game_prob * 100.0), 1)
            })

        if not all_game_scores:
            TASKS[task_id] = {
                "status": "failed",
                "error": "No valid move sequences from move 21 onward were found in the PGN."
            }
            return

        avg_suspicion_prob = float(np.mean(all_game_scores))
        suspicion_percentage = round(avg_suspicion_prob * 100.0, 1)

        if suspicion_percentage >= 75.0:
            risk_level = "HIGH SUSPICION"
        elif suspicion_percentage >= 45.0:
            risk_level = "MODERATE SUSPICION"
        else:
            risk_level = "LOW / NORMAL HUMAN PLAY"

        confidence_pct = min(100.0, max(25.0, round((len(game_breakdowns) / 10.0) * 100.0, 1)))

        report = {
            "player_name": player_name,
            "total_games_analyzed": len(game_breakdowns),
            "player_elo": int(games[0]["white_elo"]) if games else 1500,
            "suspicion_score_pct": suspicion_percentage,
            "risk_level": risk_level,
            "confidence_pct": confidence_pct,
            "top_contributing_factors": [
                {
                    "feature_label": "PGN Player Targeted Analysis",
                    "description": f"Analyzed {len(game_breakdowns)} game(s) for player '{player_name}'."
                }
            ],
            "game_breakdown": game_breakdowns,
            "legal_notice": "NOTICE: This output represents statistical likelihood based on position criticality, Stockfish streaks, and move accuracy models. It is intended for research and educational purposes and does not constitute a legal or definitive verdict."
        }

        TASKS[task_id] = {
            "status": "completed",
            "progress_pct": 100,
            "result": report
        }
    except Exception as e:
        TASKS[task_id] = {
            "status": "failed",
            "error": str(e)
        }

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/parse-pgn-players", methods=["POST"])
def parse_pgn_players():
    data = request.get_json() or {}
    pgn_text = data.get("pgn_text", "").strip()
    if not pgn_text:
        return jsonify({"status": "error", "message": "PGN text is empty"}), 400

    players = GameIngestion.extract_pgn_player_names(pgn_text)
    return jsonify({"status": "success", "players": players})

@app.route("/api/analyze/chesscom", methods=["POST"])
def analyze_chesscom():
    data = request.get_json() or {}
    username = data.get("username", "").strip()
    max_games = int(data.get("max_games", 10))
    depth = int(data.get("depth", 10))

    if not username:
        return jsonify({"status": "error", "message": "Username is required"}), 400

    task_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=run_chesscom_analysis_task,
        args=(task_id, username, max_games, depth)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "task_id": task_id})

@app.route("/api/analyze/pgn", methods=["POST"])
def analyze_pgn():
    data = request.get_json() or {}
    pgn_text = data.get("pgn_text", "").strip()
    target_color = data.get("target_color", "white").lower()
    player_name = data.get("player_name", "White Player").strip()
    depth = int(data.get("depth", 10))

    if not pgn_text:
        return jsonify({"status": "error", "message": "PGN text is required"}), 400

    task_id = str(uuid.uuid4())
    thread = threading.Thread(
        target=run_pgn_analysis_task,
        args=(task_id, pgn_text, target_color, player_name, depth)
    )
    thread.daemon = True
    thread.start()

    return jsonify({"status": "started", "task_id": task_id})

@app.route("/api/task/<task_id>", methods=["GET"])
def get_task_status(task_id):
    task = TASKS.get(task_id)
    if not task:
        return jsonify({"status": "failed", "error": "Task ID not found"}), 404
    return jsonify(task)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
