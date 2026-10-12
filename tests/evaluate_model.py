"""Evaluates cheat detection models with simulated players, bootstrap CIs, and partial cheaters.

Features:
- Thousands of fake clean players via bootstrap sampling with replacement.
- Simulates constant cheaters across assist tiers (10-30%, 30-50%, 50-70%, 70-100%).
- Simulates partial/hybrid cheaters (cheating in only 30-70% of games, legit in the rest).
- Selects 1% FPR threshold and aggregation rule on Validation split; evaluates on Test split.
- Reports Recall at 1% and 2% FPR with 95% bootstrap confidence intervals.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve

import config

ART_DIR = os.path.join(config.BASE_DIR, "tests", "artifacts")
REPORTS_DIR = os.path.join(config.BASE_DIR, "tests", "reports")
os.makedirs(REPORTS_DIR, exist_ok=True)

RATING_BANDS = [
    ("1000-1300", 1000, 1300),
    ("1300-1600", 1300, 1600),
    ("1600-1850", 1600, 1850),
    ("1850-2000", 1850, 2000),
]


def aggregate_scores(probs, method="mean"):
    """Aggregate per-game probabilities for a player into a single player score."""
    p = np.asarray(probs, dtype=float)
    if method == "mean":
        return float(np.mean(p))
    elif method == "logit_mean":
        eps = 1e-4
        clipped = np.clip(p, eps, 1.0 - eps)
        return float(np.mean(logit(clipped)))
    elif method == "hybrid_peak":
        # 50% average + 50% 85th percentile (catches players with extreme outlier games)
        return float(0.5 * np.mean(p) + 0.5 * np.percentile(p, 85))
    else:
        return float(np.mean(p))


class PlayerSimulator:
    def __init__(self, df, model, features, seed=42, include_all_clean=False):
        self.df = df.copy()
        self.model = model
        self.features = features
        self.rng = np.random.default_rng(seed)
        self.include_all_clean = include_all_clean

        # Precompute model probabilities for all sides in the pool
        X = self.df[self.features]
        self.df["prob"] = self.model.predict_proba(X)[:, 1]

        # Organize pool into clean and assist tiers per Elo band
        self.pools = {}
        for band_name, low, high in RATING_BANDS:
            band_mask = (self.df["elo"] >= low) & (self.df["elo"] < high)
            if self.include_all_clean:
                clean_sub = self.df[band_mask & (self.df["assist_rate"] == 0.0)]
            else:
                clean_sub = self.df[band_mask & (self.df["assist_rate"] == 0.0) & (self.df["is_clean_vs_clean"] == True)]

            tier_10_30 = self.df[band_mask & (self.df["assist_rate"] >= 0.10) & (self.df["assist_rate"] < 0.30)]
            tier_30_50 = self.df[band_mask & (self.df["assist_rate"] >= 0.30) & (self.df["assist_rate"] < 0.50)]
            tier_50_70 = self.df[band_mask & (self.df["assist_rate"] >= 0.50) & (self.df["assist_rate"] < 0.70)]
            tier_70_100 = self.df[band_mask & (self.df["assist_rate"] >= 0.70)]

            self.pools[band_name] = {
                "clean": clean_sub["prob"].values,
                "tier_10_30": tier_10_30["prob"].values,
                "tier_30_50": tier_30_50["prob"].values,
                "tier_50_70": tier_50_70["prob"].values,
                "tier_70_100": tier_70_100["prob"].values,
            }

    def simulate_clean_players(self, n_players=5000, agg_method="mean", n_games=None):
        scores = []
        band_names = [b[0] for b in RATING_BANDS]
        for _ in range(n_players):
            band = self.rng.choice(band_names)
            pool = self.pools[band]["clean"]
            if len(pool) == 0:
                continue
            k = n_games if n_games is not None else int(self.rng.integers(10, 51))
            sampled_probs = self.rng.choice(pool, size=k, replace=True)
            scores.append(aggregate_scores(sampled_probs, agg_method))
        return np.array(scores)

    def simulate_constant_cheaters(self, tier="tier_50_70", n_players=500, agg_method="mean", n_games=None):
        scores = []
        band_names = [b[0] for b in RATING_BANDS]
        for _ in range(n_players):
            band = self.rng.choice(band_names)
            pool = self.pools[band][tier]
            if len(pool) == 0:
                continue
            k = n_games if n_games is not None else int(self.rng.integers(10, 51))
            sampled_probs = self.rng.choice(pool, size=k, replace=True)
            scores.append(aggregate_scores(sampled_probs, agg_method))
        return np.array(scores)

    def simulate_partial_cheaters(self, n_players=1000, agg_method="mean", n_games=None, tier=None):
        """Simulates hybrid cheaters who only cheat in 30-70% of games and play cleanly in the rest."""
        scores = []
        band_names = [b[0] for b in RATING_BANDS]
        for _ in range(n_players):
            band = self.rng.choice(band_names)
            clean_pool = self.pools[band]["clean"]
            if tier is not None:
                cheat_pool = self.pools[band][tier]
            else:
                cheat_pools = [self.pools[band][t] for t in ["tier_30_50", "tier_50_70", "tier_70_100"] if len(self.pools[band][t]) > 0]
                cheat_pool = np.concatenate(cheat_pools) if cheat_pools else []

            if len(clean_pool) == 0 or len(cheat_pool) == 0:
                continue

            k = n_games if n_games is not None else int(self.rng.integers(10, 51))
            cheat_frac = float(self.rng.uniform(0.30, 0.70))
            n_cheat = int(round(k * cheat_frac))
            n_clean = k - n_cheat

            cheat_probs = self.rng.choice(cheat_pool, size=n_cheat, replace=True) if n_cheat > 0 else []
            clean_probs = self.rng.choice(clean_pool, size=n_clean, replace=True) if n_clean > 0 else []
            combined = np.concatenate([cheat_probs, clean_probs])
            self.rng.shuffle(combined)
            scores.append(aggregate_scores(combined, agg_method))
        return np.array(scores)


def tune_aggregation_and_threshold(val_df, model, features):
    """Use validation split to compare aggregation methods and determine 1% and 2% FPR thresholds."""
    sim = PlayerSimulator(val_df, model, features, seed=101)
    methods = ["mean", "logit_mean", "hybrid_peak"]
    best_method = "mean"
    best_partial_recall = -1.0
    tuning_results = {}

    print("\nTuning aggregation methods on Validation Set Simulator:")
    for m in methods:
        clean_scores = sim.simulate_clean_players(n_players=3000, agg_method=m)
        thresh_1pct = float(np.percentile(clean_scores, 99.0))
        thresh_2pct = float(np.percentile(clean_scores, 98.0))

        partial_scores = sim.simulate_partial_cheaters(n_players=800, agg_method=m)
        rec_1 = float(np.mean(partial_scores >= thresh_1pct))
        rec_2 = float(np.mean(partial_scores >= thresh_2pct))

        tuning_results[m] = {
            "thresh_1pct": thresh_1pct,
            "thresh_2pct": thresh_2pct,
            "partial_recall_at_1pct_fpr": rec_1,
            "partial_recall_at_2pct_fpr": rec_2,
        }
        print(f"  Method {m:<12}: Thresh(1% FPR)={thresh_1pct:.4f} | Partial Recall={rec_1*100:.1f}%")

        if rec_1 > best_partial_recall:
            best_partial_recall = rec_1
            best_method = m

    print(f"Selected best aggregation method: '{best_method}'\n")
    return best_method, tuning_results[best_method]["thresh_1pct"], tuning_results[best_method]["thresh_2pct"]


def evaluate_test_set(test_df, model, features, agg_method, thresh_1pct, thresh_2pct, n_bootstraps=1000):
    """Evaluate on unseen Test set: clean pool sensitivity, recall matrix across assist tiers and game counts (10, 20, 50)."""
    # 1. Distinct Clean Pool Analysis
    clean_cvc = test_df[(test_df["assist_rate"] == 0.0) & (test_df["is_clean_vs_clean"] == True)]
    all_clean = test_df[test_df["assist_rate"] == 0.0]
    distinct_clean_games = clean_cvc["game_id"].nunique()
    total_clean_games = all_clean["game_id"].nunique()

    print("--- Test Set Clean Pool Analysis ---")
    print(f"  Distinct clean-vs-clean games: {distinct_clean_games} ({len(clean_cvc)} sides)")
    print(f"  Total clean games (including clean side of assisted): {total_clean_games} ({len(all_clean)} sides)")

    # 2. FPR Sensitivity Run with Bootstrap Confidence Intervals
    sim_cvc = PlayerSimulator(test_df, model, features, seed=202, include_all_clean=False)
    sim_all = PlayerSimulator(test_df, model, features, seed=202, include_all_clean=True)

    rng = np.random.default_rng(303)

    def bootstrap_fpr(sim, thresh, n_players=5000):
        scores = sim.simulate_clean_players(n_players=n_players, agg_method=agg_method)
        observed_fpr = float(np.mean(scores >= thresh))
        boot_fprs = [np.mean(rng.choice(scores, size=len(scores), replace=True) >= thresh) for _ in range(n_bootstraps)]
        ci_lo, ci_hi = float(np.percentile(boot_fprs, 2.5)), float(np.percentile(boot_fprs, 97.5))
        return observed_fpr, ci_lo, ci_hi, scores

    fpr_cvc, fpr_cvc_lo, fpr_cvc_hi, scores_cvc = bootstrap_fpr(sim_cvc, thresh_1pct)
    fpr_all, fpr_all_lo, fpr_all_hi, scores_all = bootstrap_fpr(sim_all, thresh_1pct)

    print("\n--- Clean Player FPR Sensitivity Run (Target: 1.00%) ---")
    print(f"  Clean-vs-Clean Only FPR: {fpr_cvc*100:.2f}% (95% CI: [{fpr_cvc_lo*100:.2f}%, {fpr_cvc_hi*100:.2f}%])")
    print(f"  All Clean Sides FPR:     {fpr_all*100:.2f}% (95% CI: [{fpr_all_lo*100:.2f}%, {fpr_all_hi*100:.2f}%])")

    # 3. Recall Breakdown Matrix by Assist Tier and Game Count (10, 20, 50)
    tiers = ["tier_10_30", "tier_30_50", "tier_50_70", "tier_70_100"]
    game_counts = [10, 20, 50]

    matrix_results = {
        "full_cheaters": {},
        "partial_cheaters": {},
    }

    def bootstrap_recall(scores, thresh):
        if len(scores) == 0:
            return 0.0, 0.0, 0.0
        recs = [np.mean(rng.choice(scores, size=len(scores), replace=True) >= thresh) for _ in range(n_bootstraps)]
        return float(np.mean(recs)), float(np.percentile(recs, 2.5)), float(np.percentile(recs, 97.5))

    print("\n--- Detailed Recall Matrix at 1% FPR Threshold ---")
    for gc in game_counts:
        matrix_results["full_cheaters"][f"games_{gc}"] = {}
        matrix_results["partial_cheaters"][f"games_{gc}"] = {}

        print(f"\n[ Games Per Player: {gc} ]")
        print("  Full Cheaters (Cheating 100% of games):")
        for t in tiers:
            scores = sim_cvc.simulate_constant_cheaters(tier=t, n_players=800, agg_method=agg_method, n_games=gc)
            rec, lo, hi = bootstrap_recall(scores, thresh_1pct)
            matrix_results["full_cheaters"][f"games_{gc}"][t] = {
                "recall": round(rec, 4), "ci_95_lo": round(lo, 4), "ci_95_hi": round(hi, 4)
            }
            print(f"    {t:<14}: Recall = {rec*100:5.1f}% (95% CI: [{lo*100:5.1f}%, {hi*100:5.1f}%])")

        print("  Partial Cheaters (Cheating 30-70% of games):")
        for t in tiers:
            scores = sim_cvc.simulate_partial_cheaters(n_players=800, agg_method=agg_method, n_games=gc, tier=t)
            rec, lo, hi = bootstrap_recall(scores, thresh_1pct)
            matrix_results["partial_cheaters"][f"games_{gc}"][t] = {
                "recall": round(rec, 4), "ci_95_lo": round(lo, 4), "ci_95_hi": round(hi, 4)
            }
            print(f"    {t:<14}: Recall = {rec*100:5.1f}% (95% CI: [{lo*100:5.1f}%, {hi*100:5.1f}%])")

        # Mixed partial cheaters
        scores_mix = sim_cvc.simulate_partial_cheaters(n_players=1000, agg_method=agg_method, n_games=gc, tier=None)
        rec_mix, lo_mix, hi_mix = bootstrap_recall(scores_mix, thresh_1pct)
        matrix_results["partial_cheaters"][f"games_{gc}"]["mixed_tiers"] = {
            "recall": round(rec_mix, 4), "ci_95_lo": round(lo_mix, 4), "ci_95_hi": round(hi_mix, 4)
        }
        print(f"    {'mixed_tiers':<14}: Recall = {rec_mix*100:5.1f}% (95% CI: [{lo_mix*100:5.1f}%, {hi_mix*100:5.1f}%])")

    # Save detailed evaluation metrics
    evaluation_data = {
        "aggregation_method": agg_method,
        "val_thresh_1pct": thresh_1pct,
        "val_thresh_2pct": thresh_2pct,
        "clean_pool": {
            "distinct_clean_games_cvc": distinct_clean_games,
            "total_clean_games_all": total_clean_games,
            "fpr_cvc_only": round(fpr_cvc, 4),
            "fpr_cvc_ci_95": [round(fpr_cvc_lo, 4), round(fpr_cvc_hi, 4)],
            "fpr_all_clean": round(fpr_all, 4),
            "fpr_all_ci_95": [round(fpr_all_lo, 4), round(fpr_all_hi, 4)],
        },
        "matrix_results": matrix_results,
    }

    metrics_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(metrics_path, "w") as fp:
        json.dump(evaluation_data, fp, indent=2)

    # 4. Generate Grouped Recall Chart across 10, 20, 50 games
    plt.figure(figsize=(10, 5.5))
    x_labels = ["10-30%", "30-50%", "50-70%", "70-100%"]
    x = np.arange(len(x_labels))
    width = 0.25

    for idx, gc in enumerate(game_counts):
        vals = [matrix_results["full_cheaters"][f"games_{gc}"][t]["recall"] * 100 for t in tiers]
        errs = [
            [vals[i] - matrix_results["full_cheaters"][f"games_{gc}"][t]["ci_95_lo"] * 100 for i, t in enumerate(tiers)],
            [matrix_results["full_cheaters"][f"games_{gc}"][t]["ci_95_hi"] * 100 - vals[i] for i, t in enumerate(tiers)],
        ]
        plt.bar(x + (idx - 1) * width, vals, width, yerr=errs, capsize=4, label=f"{gc} games/player", edgecolor="black")

    plt.xticks(x, x_labels)
    plt.xlabel("Stockfish Move-Level Assist Tier")
    plt.ylabel("Recall at 1% FPR (%)")
    plt.title("Cheater Recall vs Assist Tier by Games Per Player (1% FPR)")
    plt.ylim(0, 105)
    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=True)
    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "recall_by_assist_level.png")
    plt.savefig(chart_path, dpi=130)
    plt.close()

    # 5. Overall ROC & PR Curve
    all_clean_scores = sim_cvc.simulate_clean_players(n_players=4750, agg_method=agg_method)
    all_cheat_scores = sim_cvc.simulate_partial_cheaters(n_players=250, agg_method=agg_method)
    y_true = np.array([0] * len(all_clean_scores) + [1] * len(all_cheat_scores))
    y_scores = np.concatenate([all_clean_scores, all_cheat_scores])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = float(roc_auc_score(y_true, y_scores))
    prec, rec, _ = precision_recall_curve(y_true, y_scores)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(fpr * 100, tpr * 100, color="#2563eb", linewidth=2, label=f"Player AUC = {roc_auc:.3f}")
    ax1.plot([0, 100], [0, 100], "--", color="gray")
    ax1.set_xlabel("False Positive Rate (%)")
    ax1.set_ylabel("True Positive Rate (%)")
    ax1.set_title("Simulated Player ROC Curve", fontweight="bold")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(rec * 100, prec * 100, color="#10b981", linewidth=2)
    ax2.set_xlabel("Recall (%)")
    ax2.set_ylabel("Precision (%)")
    ax2.set_title("Precision-Recall (Realistic 5% Prevalence)", fontweight="bold")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    roc_path = os.path.join(REPORTS_DIR, "player_level_roc_pr.png")
    plt.savefig(roc_path, dpi=130)
    plt.close()

    # Print summary
    print("-" * 75)
    print(f"TEST SET SIMULATION RESULTS (Operating at Val 1% FPR Threshold = {thresh_1pct:.4f}):")
    print(f"  Observed Clean FPR on Test Set: {actual_fpr_1*100:.2f}% (Target: 1.00%)")
    print("-" * 75)
    for k, v in results_by_tier.items():
        print(f"  {k:<20}: Recall@1%FPR = {v['recall_1pct_fpr']*100:.1f}% (95% CI: [{v['ci_95_lo']*100:.1f}%, {v['ci_95_hi']*100:.1f}%]) | Recall@2%FPR = {v['recall_2pct_fpr']*100:.1f}%")
    print("-" * 75)

    final_report = {
        "aggregation_method": agg_method,
        "val_thresh_1pct": thresh_1pct,
        "val_thresh_2pct": thresh_2pct,
        "test_observed_fpr_1": actual_fpr_1,
        "test_observed_fpr_2": actual_fpr_2,
        "player_level_auc": roc_auc,
        "results_by_tier": results_by_tier,
    }
    with open(os.path.join(REPORTS_DIR, "evaluation_metrics.json"), "w") as f:
        json.dump(final_report, f, indent=2)

    return final_report


def main():
    val_csv = os.path.join(ART_DIR, "val_set.csv")
    test_csv = os.path.join(ART_DIR, "test_set.csv")
    if not os.path.exists(val_csv) or not os.path.exists(test_csv):
        print("Validation or Test splits not found. Please run src.train first.")
        return

    val_df = pd.read_csv(val_csv)
    test_df = pd.read_csv(test_csv)
    bundle = joblib.load(config.MODEL_PATH)
    model = bundle["model"]
    features = bundle["features"]

    best_agg, th_1, th_2 = tune_aggregation_and_threshold(val_df, model, features)
    evaluate_test_set(test_df, model, features, best_agg, th_1, th_2)


if __name__ == "__main__":
    main()
