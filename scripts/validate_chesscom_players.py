"""Real-world Chess.com Validation Script.

Fetches recent rated games for real Chess.com accounts, runs the exact app engine pipeline,
and reports suspicion scores and distributions for:
(a) Real clean Chess.com players (active, non-closed accounts, Elo 1000-2000).
(b) Chess.com accounts publicly closed for fair-play violations.
"""
import argparse
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess.pgn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests

import config
from src import chesscom
from src.engine import Analyzer
from src.features import ENGINE_FEATURE_NAMES, extract_single_side_engine_features


def analyze_chesscom_user(username, model, features, n_games=20, depth=12):
    """Analyze a single Chess.com user using the app pipeline."""
    try:
        games = chesscom.fetch_recent_games(username, n=n_games)
        if not games:
            return None, "No rated games found"

        analyzer = Analyzer(depth=depth)
        game_probs = []
        game_acpls = []
        game_top1s = []

        try:
            for g in games:
                color, chesscom_rating, opp_name = chesscom.side_of(g, username)
                mapped_elo = int(round(0.88 * chesscom_rating + 350))
                mapped_elo = min(config.MAX_ELO, max(1000, mapped_elo))

                pgn_text = g.get("pgn", "")
                game_obj = chess.pgn.read_game(io.StringIO(pgn_text))
                if not game_obj:
                    continue

                feat_dict = extract_single_side_engine_features(game_obj, color, mapped_elo, analyzer)
                if not feat_dict:
                    continue

                row = [feat_dict[f] for f in features]
                prob = float(model.predict_proba([row])[0, 1]) * 100.0
                game_probs.append(prob)
                if "acpl" in feat_dict:
                    game_acpls.append(feat_dict["acpl"])
                if "top1_match" in feat_dict:
                    game_top1s.append(feat_dict["top1_match"])
        finally:
            analyzer.close()

        if not game_probs:
            return None, "No games successfully extracted"

        probs_arr = np.array(game_probs)
        # Hybrid peak aggregation used in app
        if len(probs_arr) >= 3:
            score = float(0.5 * probs_arr.mean() + 0.5 * np.percentile(probs_arr, 85))
        else:
            score = float(probs_arr.mean())

        return {
            "username": username,
            "games_scored": len(game_probs),
            "score": round(score, 2),
            "mean_prob": round(float(probs_arr.mean()), 2),
            "p85_prob": round(float(np.percentile(probs_arr, 85)), 2),
            "avg_acpl": round(float(np.mean(game_acpls)), 1) if game_acpls else None,
            "avg_top1": round(float(np.mean(game_top1s)), 1) if game_top1s else None,
            "flagged_games": int((probs_arr >= 50.0).sum()),
        }, None
    except Exception as e:
        return None, str(e)


def run_validation(clean_users, closed_users, model_path=None, n_games=20, depth=12):
    model_path = model_path or os.path.join(config.BASE_DIR, "models", "cheat_model.joblib")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    features = bundle["features"]

    print(f"\n--- Running Real-World Validation on Chess.com ({len(clean_users)} clean, {len(closed_users)} fair-play closed) ---")
    
    clean_results = []
    print("\nAnalyzing Clean Players:")
    for u in clean_users:
        res, err = analyze_chesscom_user(u, model, features, n_games=n_games, depth=depth)
        if res:
            clean_results.append(res)
            print(f"  [CLEAN] @{u:<20} | Score: {res['score']:5.1f}% | Scored: {res['games_scored']:2d} games | ACPL: {res['avg_acpl']}")
        else:
            print(f"  [CLEAN] @{u:<20} | Skipped: {err}")

    closed_results = []
    print("\nAnalyzing Fair-Play Closed Accounts:")
    for u in closed_users:
        res, err = analyze_chesscom_user(u, model, features, n_games=n_games, depth=depth)
        if res:
            closed_results.append(res)
            print(f"  [CLOSED] @{u:<20} | Score: {res['score']:5.1f}% | Scored: {res['games_scored']:2d} games | ACPL: {res['avg_acpl']}")
        else:
            print(f"  [CLOSED] @{u:<20} | Skipped: {err}")

    clean_scores = [r["score"] for r in clean_results]
    closed_scores = [r["score"] for r in closed_results]

    out_data = {
        "clean_players": clean_results,
        "closed_players": closed_results,
        "clean_scores": clean_scores,
        "closed_scores": closed_scores,
    }

    out_json = os.path.join(config.BASE_DIR, "tests", "reports", "real_world_chesscom_validation.json")
    with open(out_json, "w") as fp:
        json.dump(out_data, fp, indent=2)

    # Plot comparison chart
    plt.figure(figsize=(9, 5))
    bins = np.linspace(0, 100, 21)
    if clean_scores:
        plt.hist(clean_scores, bins=bins, alpha=0.6, label=f"Clean Active ({len(clean_scores)})", color="green", edgecolor="black")
    if closed_scores:
        plt.hist(closed_scores, bins=bins, alpha=0.6, label=f"Fair-Play Closed ({len(closed_scores)})", color="crimson", edgecolor="black")

    plt.axvline(50.0, color="black", linestyle="--", linewidth=1.5, label="1% FPR Threshold (~50%)")
    plt.xlabel("Player Suspicion Score (%)")
    plt.ylabel("Player Count")
    plt.title("Real-World Chess.com Suspicion Score Distribution")
    plt.legend(frameon=True)
    plt.tight_layout()
    plot_path = os.path.join(config.BASE_DIR, "tests", "reports", "real_world_chesscom_validation.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved validation results to {out_json} and plot to {plot_path}")
    return out_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--games", type=int, default=20)
    parser.add_argument("--depth", type=int, default=12)
    args = parser.parse_args()
