"""Loads the trained model and produces a suspicion report for a chess.com player."""
import math
import os
import time

import joblib
import numpy as np

import config
from src import chesscom
from src.features import extract_features, ENGINE_FEATURES

_bundle = None


def load_bundle(path=None):
    global _bundle
    path = path or config.MODEL_PATH
    if _bundle is None and os.path.exists(path):
        _bundle = joblib.load(path)
    return _bundle


def verdict(score):
    if score < 30:
        return "Low suspicion"
    if score < 55:
        return "Moderate suspicion"
    return "High suspicion"


def analyze_player(username, n_games):
    bundle = load_bundle()
    if bundle is None:
        raise RuntimeError("Model file not found. Train the model first (see guide.md).")
    features = bundle["features"]
    analyzer = None
    if any(f in ENGINE_FEATURES for f in features):
        from src.engine import Analyzer
        analyzer = Analyzer()
    try:
        raw = chesscom.fetch_recent_games(username, n_games)
        results = []
        for g in raw:
            color, rating, opponent = chesscom.side_of(g, username)
            game = chesscom.parse_pgn(g["pgn"])
            feats = extract_features(game, color, analyzer) if game else None
            if feats is None:
                continue
            x = np.array([[feats[f] for f in features]])
            prob = float(bundle["model"].predict_proba(x)[0, 1])
            results.append({
                "url": g.get("url", ""), "opponent": opponent, "rating": rating,
                "time_class": g.get("time_class", ""),
                "date": time.strftime("%Y-%m-%d", time.gmtime(g.get("end_time", 0))),
                "probability": round(prob * 100, 1),
            })
    finally:
        if analyzer:
            analyzer.close()
    if not results:
        raise ValueError("No usable games with clock data were found for this player.")

    avg_rating = int(np.mean([r["rating"] for r in results]))
    if avg_rating > config.MAX_ELO:
        raise ValueError(f"Average rating {avg_rating} is above the supported limit of {config.MAX_ELO}.")

    probs = np.array([r["probability"] for r in results])
    score = float(probs.mean())
    # 95 percent interval of the mean, wider when we have fewer games
    margin = 1.96 * probs.std(ddof=1) / math.sqrt(len(probs)) if len(probs) > 1 else 50.0
    return {
        "username": username, "games_requested": n_games, "games_scored": len(results),
        "avg_rating": avg_rating, "suspicion": round(score, 1),
        "ci_low": round(max(0, score - margin), 1), "ci_high": round(min(100, score + margin), 1),
        "verdict": verdict(score), "flagged_games": int((probs >= 50).sum()), "games": results,
    }
