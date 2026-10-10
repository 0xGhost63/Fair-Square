"""Loads the trained model and produces a suspicion report for a chess.com player."""
import math
import os
import time

import joblib
import numpy as np
import pandas as pd

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


def analyze_player_stream(username, n_games, depth=None):
    bundle = load_bundle()
    if bundle is None:
        raise RuntimeError("Model file not found. Train the model first (see guide.md).")
    features = bundle["features"]
    engine_depth = int(depth) if depth is not None else config.ENGINE_DEPTH
    analyzer = None
    if any(f in ENGINE_FEATURES for f in features):
        from src.engine import Analyzer
        analyzer = Analyzer(depth=engine_depth)
    try:
        results = []
        yield {"type": "step", "step": 1, "text": "Connected to Chess.com Public API..."}
        yield {"type": "step", "step": 2, "text": "Streaming PGN archives & clock comments..."}
        yield {"type": "step", "step": 3, "text": f"Evaluating move accuracy with Stockfish 16 Engine (Depth {engine_depth})..."}

        for g in chesscom.stream_usable_games(username):
            color, rating, opponent = chesscom.side_of(g, username)
            game = chesscom.parse_pgn(g["pgn"])
            feats = extract_features(game, color, analyzer) if game else None
            if feats is None:
                continue
            x = pd.DataFrame([[feats[f] for f in features]], columns=features)
            prob = float(bundle["model"].predict_proba(x)[0, 1])
            res = {
                "url": g.get("url", ""),
                "opponent": opponent,
                "rating": rating,
                "time_class": g.get("time_class", ""),
                "date": time.strftime("%Y-%m-%d", time.gmtime(g.get("end_time", 0))),
                "probability": round(prob * 100, 1),
            }
            if "acpl" in feats:
                res["acpl"] = round(feats["acpl"], 1)
            if "top1_match" in feats:
                res["top1_match"] = round(feats["top1_match"] * 100, 1)
            if "blunder_rate" in feats:
                res["blunder_rate"] = round(feats["blunder_rate"] * 100, 1)
            results.append(res)

            yield {
                "type": "progress",
                "current": len(results),
                "total": n_games,
                "opponent": opponent,
                "step": 3 if len(results) < n_games else 4,
            }
            if len(results) >= n_games:
                break
    finally:
        if analyzer:
            analyzer.close()

    if not results:
        raise ValueError("No usable games with clock data were found for this player.")

    yield {"type": "step", "step": 4, "text": "Evaluating Hybrid Engine + Timing Model..."}

    avg_rating = int(np.mean([r["rating"] for r in results]))
    if avg_rating > config.MAX_ELO:
        raise ValueError(f"Average rating {avg_rating} is above the supported limit of {config.MAX_ELO}.")

    probs = np.array([r["probability"] for r in results])
    score = float(probs.mean())
    margin = 1.96 * probs.std(ddof=1) / math.sqrt(len(probs)) if len(probs) > 1 else 50.0

    avg_acpl = round(float(np.mean([r["acpl"] for r in results if "acpl" in r])), 1) if any("acpl" in r for r in results) else None
    avg_top1 = round(float(np.mean([r["top1_match"] for r in results if "top1_match" in r])), 1) if any("top1_match" in r for r in results) else None
    avg_blunder = round(float(np.mean([r["blunder_rate"] for r in results if "blunder_rate" in r])), 1) if any("blunder_rate" in r for r in results) else None

    result_data = {
        "username": username,
        "games_requested": n_games,
        "games_scored": len(results),
        "avg_rating": avg_rating,
        "suspicion": round(score, 1),
        "ci_low": round(max(0, score - margin), 1),
        "ci_high": round(min(100, score + margin), 1),
        "verdict": verdict(score),
        "flagged_games": int((probs >= 50).sum()),
        "avg_acpl": avg_acpl,
        "avg_top1": avg_top1,
        "avg_blunder_rate": avg_blunder,
        "has_engine": analyzer is not None or "acpl" in features,
        "engine_depth": engine_depth,
        "games": results,
    }
    yield {"type": "done", "result": result_data}


def analyze_player(username, n_games, depth=None):
    final_res = None
    for item in analyze_player_stream(username, n_games, depth=depth):
        if item.get("type") == "done":
            final_res = item["result"]
    if final_res is None:
        raise RuntimeError("Analysis terminated without result.")
    return final_res
