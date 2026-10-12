"""Elo Sensitivity Test: Measures impact of shifting mapped Elo by +/- 150 points.

Evaluates how player-level FPR and Recall change when Chess.com rating mapping
has an offset error of +/- 150 Elo.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config
from src.features import ENGINE_FEATURE_NAMES
from tests.evaluate_model import PlayerSimulator, aggregate_scores


def run_elo_sensitivity(test_df, model, features, val_thresh, agg_method="mean"):
    results = {}
    shifts = [-150, 0, 150]

    for shift in shifts:
        shifted_df = test_df.copy()
        shifted_df["elo"] = np.clip(shifted_df["elo"] + shift, 1000, 2000)

        sim = PlayerSimulator(shifted_df, model, features, seed=42)
        clean_scores = sim.simulate_clean_players(n_players=3000, agg_method=agg_method)
        fpr = float(np.mean(clean_scores >= val_thresh))

        tiers = ["tier_10_30", "tier_30_50", "tier_50_70", "tier_70_100"]
        recs = {}
        for t in tiers:
            s = sim.simulate_constant_cheaters(tier=t, n_players=500, agg_method=agg_method)
            recs[t] = float(np.mean(s >= val_thresh)) if len(s) > 0 else 0.0

        part_scores = sim.simulate_partial_cheaters(n_players=800, agg_method=agg_method)
        part_rec = float(np.mean(part_scores >= val_thresh)) if len(part_scores) > 0 else 0.0

        key = f"shift_{shift:+d}" if shift != 0 else "baseline"
        results[key] = {
            "shift": shift,
            "clean_fpr": round(fpr, 4),
            "partial_cheater_recall": round(part_rec, 4),
            "tier_recalls": {k: round(v, 4) for k, v in recs.items()},
        }

    return results


if __name__ == "__main__":
    from src.train import load_and_split_data
    csv_path = os.path.join(config.BASE_DIR, "data", "engine_features.csv")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        sys.exit(1)

    train_df, val_df, test_df, _ = load_and_split_data(csv_path)
    bundle = joblib.load(os.path.join(config.BASE_DIR, "models", "cheat_model.joblib"))
    model = bundle["model"]
    features = bundle["features"]

    # Load validation threshold from evaluation_metrics.json if exists
    eval_metrics_path = os.path.join(config.BASE_DIR, "tests", "reports", "evaluation_metrics.json")
    val_thresh = 0.50
    if os.path.exists(eval_metrics_path):
        with open(eval_metrics_path) as fp:
            d = json.load(fp)
            val_thresh = d.get("val_thresh_1pct", 0.50)

    sens = run_elo_sensitivity(test_df, model, features, val_thresh)
    print(json.dumps(sens, indent=2))
