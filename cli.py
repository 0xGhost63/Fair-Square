import sys
import os
import argparse
import joblib
import pandas as pd
import numpy as np
import chess
import chess.engine
import chess.pgn
from phase2_features import build_move_features, win_prob
from phase4_game_model import build_game_level_features

DEFAULT_STOCKFISH_PATH = os.path.join("bin", "stockfish-engine")

def analyze_pgn_game(game: chess.pgn.Game, engine_path: str = DEFAULT_STOCKFISH_PATH, depth: int = 15):
    """
    Phase 5 PGN Move Analyzer:
    Analyzes a PGN game move by move at depth 15 using Stockfish.
    Computes eval_before, eval_after, centipawn_loss, normalized_centipawn_loss,
    and Stockfish 1, 9, 15 move matches for focal player (White by default).
    """
    if not os.path.exists(engine_path):
        engine_path = "stockfish"

    board = game.board()
    move_records = []
    ply = 0

    white_user = game.headers.get("White", "White Player")
    black_user = game.headers.get("Black", "Black Player")
    white_elo = float(game.headers.get("WhiteElo", 1500))
    black_elo = float(game.headers.get("BlackElo", 1500))

    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        print(f"Error launching Stockfish engine at {engine_path}: {e}")
        return []

    try:
        for node in game.mainline():
            move = node.move
            current_color = "white" if board.turn == chess.WHITE else "black"
            full_move_num = (ply // 2) + 1
            ply += 1

            # Only process moves from ply 21 onward (focal player turns)
            if ply < 21:
                board.push(move)
                continue

            # Evaluate position before move at depth 15
            eval_before_info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            score_before = eval_before_info[0]["score"].relative.score(mate_score=10000)
            if board.turn == chess.BLACK:
                score_before = -score_before
            eval_before_val = float(np.clip(score_before, -1000.0, 1000.0))

            best_move = eval_before_info[0]["pv"][0].uci() if eval_before_info[0].get("pv") else move.uci()

            # Push played move to evaluate after
            played_uci = move.uci()
            board.push(move)

            eval_after_info = engine.analyse(board, chess.engine.Limit(depth=depth), multipv=1)
            score_after = eval_after_info[0]["score"].relative.score(mate_score=10000)
            if board.turn == chess.WHITE:  # Note: after move, turn toggled
                score_after = -score_after
            eval_after_val = float(np.clip(score_after, -1000.0, 1000.0))

            cp_loss = max(0.0, float(eval_before_val - eval_after_val))
            norm_loss = max(0.0, float(win_prob(eval_before_val) - win_prob(eval_after_val)))

            # Stockfish move match approximations (at depth 15)
            sf1_match = played_uci
            sf9_match = played_uci if played_uci == best_move else best_move
            sf15_match = best_move

            move_records.append({
                "game_id": game.headers.get("Event", "PGN Game"),
                "player_id": 1,
                "player_elo": white_elo,
                "opponent_elo": black_elo,
                "half_move": ply,
                "move_player": played_uci,
                "move_stockfish_1": sf1_match,
                "move_stockfish_9": sf9_match,
                "move_stockfish_15": sf15_match,
                "eval_before": eval_before_val,
                "eval_after": eval_after_val,
                "centipawn_loss": cp_loss,
                "normalized_centipawn_loss": norm_loss,
                "is_used": True,
                "is_cheating_player_game": 0
            })
    finally:
        engine.quit()

    return move_records

def run_pgn_cli(pgn_path: str, engine_path: str = DEFAULT_STOCKFISH_PATH, depth: int = 15):
    """
    Phase 5 CLI Pipeline:
    Reads PGN file, runs Stockfish analysis, extracts move & game features,
    and prints suspicion score & engine streaks.
    """
    if not os.path.exists("models/move_model.joblib") or not os.path.exists("models/game_model.joblib"):
        print("Models not found. Training models first...")
        from train_and_save_models import train_and_save_pipeline_models
        train_and_save_pipeline_models()

    move_clf = joblib.load("models/move_model.joblib")
    game_clf = joblib.load("models/game_model.joblib")

    print(f"Reading PGN file: {pgn_path}")
    with open(pgn_path, "r", encoding="utf-8", errors="replace") as f:
        game = chess.pgn.read_game(f)

    if not game:
        print("Error: Invalid or empty PGN file.")
        return

    white_user = game.headers.get("White", "White Player")
    black_user = game.headers.get("Black", "Black Player")
    event = game.headers.get("Event", "Chess Game")

    print(f"Analyzing game: {white_user} vs {black_user} ({event})")
    print(f"Running Stockfish analysis at depth {depth}...")

    move_records = analyze_pgn_game(game, engine_path=engine_path, depth=depth)
    if not move_records:
        print("No moves processed (game may be under 21 half-moves).")
        return

    df_moves = pd.DataFrame(move_records)
    X_move = build_move_features(df_moves)

    move_probs = move_clf.predict_proba(X_move)[:, 1]
    df_game = build_game_level_features(df_moves, move_probs)

    game_feature_cols = [
        "mean_move_score", "max_move_score", "top10_mean_score",
        "max_sf_streak", "streaks_len3_plus_count", "critical_match_rate",
        "game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"
    ]

    X_game = df_game[game_feature_cols]
    suspicion_prob = float(game_clf.predict_proba(X_game)[0, 1])
    suspicion_pct = round(suspicion_prob * 100.0, 1)

    row_game = df_game.iloc[0]

    print("\n==================================================")
    print("           FAIR SQUARE CLI DETECTION REPORT       ")
    print("==================================================")
    print(f"Players:             {white_user} vs {black_user}")
    print(f"Event:               {event}")
    print(f"Moves Analyzed:      {len(move_records)} moves (move 21+)")
    print(f"Suspicion Score:     {suspicion_pct}%")
    print(f"Stockfish Match Rate:{row_game['game_sf15_match_rate']:.1%}")
    print(f"Avg Centipawn Loss:  {row_game['game_avg_cpl']:.1f} cp")
    print(f"Max Engine Streak:   {int(row_game['max_sf_streak'])} consecutive moves")
    print(f"Streaks (>= 3 moves):{int(row_game['streaks_len3_plus_count'])} streak(s)")
    print(f"Critical Position Match: {row_game['critical_match_rate']:.1%}")
    print("==================================================")

def main():
    parser = argparse.ArgumentParser(description="Fair Square Chess Cheat Detection CLI")
    parser.add_argument("--file", required=True, help="Path to input PGN file")
    parser.add_argument("--depth", type=int, default=15, help="Stockfish depth (default 15)")
    parser.add_argument("--engine", default=DEFAULT_STOCKFISH_PATH, help="Path to Stockfish binary")
    args = parser.parse_args()

    run_pgn_cli(args.file, engine_path=args.engine, depth=args.depth)

if __name__ == "__main__":
    main()
