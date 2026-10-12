"""End-to-End Kaggle Training & Evaluation Pipeline for Fair Square.

Run this script on Kaggle (CPU accelerator, Internet ON):
    python kaggle_pipeline.py --max-games 5000

Workflow:
1. Verifies/installs Stockfish engine.
2. Performs stratified sampling of 4,000-6,000 games from dataset.csv.
3. Runs parallel single-pass MultiPV 3 engine extraction (Threads=1, Hash=32, Depth=12).
4. Saves resumable checkpoints to data/features_checkpoint.parquet every 250 games.
5. Performs leakage-free game_id group split (70% Train, 15% Val, 15% Test).
6. Runs shuffled-label check (~0.50 AUC) and computes per-feature AUC table.
7. Trains Logistic Regression and HistGradientBoosting.
8. Simulates 5,000 fake clean players and partial cheaters.
9. Tunes 1% FPR threshold and aggregation on Validation split; evaluates on Test split.
10. Exports production cheat_model.joblib and evaluation charts.
"""
import argparse
import concurrent.futures
import io
import json
import os
import shutil
import subprocess
import sys

import chess.pgn
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import config
from src.engine import Analyzer
from src.features import ENGINE_FEATURE_NAMES, extract_game_engine_features

RATING_BANDS = [
    ("1000-1300", 1000, 1300),
    ("1300-1600", 1300, 1600),
    ("1600-1850", 1600, 1850),
    ("1850-2000", 1850, 2000),
]


def ensure_stockfish():
    """Ensure Stockfish executable is available."""
    if os.path.exists(config.STOCKFISH_PATH) and os.access(config.STOCKFISH_PATH, os.X_OK):
        print(f"Stockfish verified at: {config.STOCKFISH_PATH}")
        return

    sys_sf = shutil.which("stockfish")
    if sys_sf:
        config.STOCKFISH_PATH = sys_sf
        print(f"Using system Stockfish at: {sys_sf}")
        return

    print("Stockfish not found. Attempting apt install...")
    try:
        subprocess.run(["apt-get", "update"], check=True)
        subprocess.run(["apt-get", "install", "-y", "stockfish"], check=True)
        config.STOCKFISH_PATH = shutil.which("stockfish") or "/usr/games/stockfish"
        print(f"Installed Stockfish at: {config.STOCKFISH_PATH}")
    except Exception as e:
        print(f"Failed to auto-install Stockfish: {e}. Please install manually.")


def aggregate_scores(probs, method="mean"):
    p = np.asarray(probs, dtype=float)
    if method == "mean":
        return float(np.mean(p))
    elif method == "logit_mean":
        eps = 1e-4
        return float(np.mean(logit(np.clip(p, eps, 1.0 - eps))))
    elif method == "hybrid_peak":
        return float(0.5 * np.mean(p) + 0.5 * np.percentile(p, 85))
    return float(np.mean(p))


def load_dataset_stratified(dataset_path="dataset.csv", max_games=5000, seed=42):
    print(f"\n--- Loading and Stratifying {dataset_path} ---")
    df = pd.read_csv(dataset_path, dtype=str)
    df["game_id"] = np.arange(len(df))

    # Filter Elo strictly to [1000, 2000]
    df["Elo White"] = pd.to_numeric(df["Elo White"], errors="coerce")
    df["Elo Black"] = pd.to_numeric(df["Elo Black"], errors="coerce")
    valid_elo = (
        (df["Elo White"] >= 1000) & (df["Elo White"] <= 2000) &
        (df["Elo Black"] >= 1000) & (df["Elo Black"] <= 2000)
    )
    df = df[valid_elo].copy()

    # Drop malformed odd short strings (length <= 3 that are not '0')
    w_valid = (df["Liste cheat white"] == "0") | (df["Liste cheat white"].str.len() > 3)
    b_valid = (df["Liste cheat black"] == "0") | (df["Liste cheat black"].str.len() > 3)
    df = df[w_valid & b_valid].copy()

    def calc_rate(s):
        return 0.0 if s == "0" else float(s.count("1")) / float(len(s))

    df["w_rate"] = df["Liste cheat white"].apply(calc_rate)
    df["b_rate"] = df["Liste cheat black"].apply(calc_rate)
    df["is_clean_vs_clean"] = (df["w_rate"] == 0.0) & (df["b_rate"] == 0.0)
    df["is_assisted"] = (df["w_rate"] > 0.0) | (df["b_rate"] > 0.0)

    clean_games = df[df["is_clean_vs_clean"]].copy()
    assisted_games = df[df["is_assisted"]].copy()

    assisted_games["max_rate"] = assisted_games[["w_rate", "b_rate"]].max(axis=1)
    assisted_games["rate_tier"] = pd.cut(
        assisted_games["max_rate"],
        bins=[0.0, 0.30, 0.50, 0.70, 1.01],
        labels=["tier_10_30", "tier_30_50", "tier_50_70", "tier_70_100"],
    )

    n_clean = min(len(clean_games), max_games // 2)
    n_assisted = min(len(assisted_games), max_games - n_clean)

    sampled_clean = clean_games.sample(n=n_clean, random_state=seed)
    sampled_assisted_list = []
    per_tier = n_assisted // 4
    for tier in ["tier_10_30", "tier_30_50", "tier_50_70", "tier_70_100"]:
        sub = assisted_games[assisted_games["rate_tier"] == tier]
        if len(sub) > 0:
            sampled_assisted_list.append(sub.sample(n=min(len(sub), per_tier), random_state=seed))

    sampled_assisted = pd.concat(sampled_assisted_list, ignore_index=True)
    selected = pd.concat([sampled_clean, sampled_assisted], ignore_index=True).sample(frac=1.0, random_state=seed)
    print(f"Selected {len(selected)} games ({len(sampled_clean)} clean-vs-clean, {len(sampled_assisted)} assisted).")
    return selected


def process_one_game(g):
    try:
        game = chess.pgn.read_game(io.StringIO(g["Game"]))
        if not game:
            return []
        analyzer = Analyzer(depth=config.ENGINE_DEPTH)
        try:
            wf, bf = extract_game_engine_features(game, int(g["Elo White"]), int(g["Elo Black"]), analyzer)
        finally:
            analyzer.close()

        res = []
        if wf:
            res.append({
                "game_id": g["game_id"], "side": "white",
                "label": 1 if g["w_rate"] >= 0.30 else (0 if g["w_rate"] == 0.0 else -1),
                "assist_rate": g["w_rate"], "is_clean_vs_clean": g["is_clean_vs_clean"],
                **wf,
            })
        if bf:
            res.append({
                "game_id": g["game_id"], "side": "black",
                "label": 1 if g["b_rate"] >= 0.30 else (0 if g["b_rate"] == 0.0 else -1),
                "assist_rate": g["b_rate"], "is_clean_vs_clean": g["is_clean_vs_clean"],
                **bf,
            })
        return res
    except Exception:
        return []


def run_parallel_extraction(selected_df, workers=4, chunk_size=250):
    os.makedirs("data", exist_ok=True)
    cp_path = "data/features_checkpoint.parquet"
    out_csv = "data/engine_features.csv"

    existing_rows = []
    processed_ids = set()
    if os.path.exists(cp_path):
        try:
            cp_df = pd.read_parquet(cp_path)
            processed_ids = set(cp_df["game_id"].unique())
            existing_rows = cp_df.to_dict("records")
            print(f"Resuming from checkpoint ({len(processed_ids)} games already extracted).")
        except Exception:
            pass

    to_run = selected_df[~selected_df["game_id"].isin(processed_ids)].to_dict("records")
    print(f"Processing {len(to_run)} games using {workers} worker processes...")

    all_rows = existing_rows
    if to_run:
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
            for i in range(0, len(to_run), chunk_size):
                chunk = to_run[i:i + chunk_size]
                futures = [executor.submit(process_one_game, g) for g in chunk]
                for fut in tqdm(concurrent.futures.as_completed(futures), total=len(chunk), desc=f"Chunk {i//chunk_size + 1}"):
                    res = fut.result()
                    if res:
                        all_rows.extend(res)
                # Checkpoint
                pd.DataFrame(all_rows).to_parquet(cp_path, index=False)

    final_df = pd.DataFrame(all_rows)
    final_df.to_csv(out_csv, index=False)
    print(f"Saved extracted features: {len(final_df)} rows to {out_csv}")
    return final_df


def train_models(df):
    print("\n--- Training Models with Strict Game-Level Split ---")
    mask_clean = (df["assist_rate"] == 0.0) & (df["is_clean_vs_clean"] == True)
    mask_cheat = df["assist_rate"] >= 0.30
    trainable = df[mask_clean | mask_cheat].copy()
    trainable["label"] = (trainable["assist_rate"] >= 0.30).astype(int)

    # 70% Train, 15% Val, 15% Test
    gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    tr_val_idx, te_idx = next(gss1.split(trainable, trainable["label"], trainable["game_id"]))
    tr_val_df, test_df = trainable.iloc[tr_val_idx].copy(), trainable.iloc[te_idx].copy()

    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=42)
    tr_idx, val_idx = next(gss2.split(tr_val_df, tr_val_df["label"], tr_val_df["game_id"]))
    train_df, val_df = tr_val_df.iloc[tr_idx].copy(), tr_val_df.iloc[val_idx].copy()

    features = ENGINE_FEATURE_NAMES
    X_tr, y_tr = train_df[features], train_df["label"]
    X_val, y_val = val_df[features], val_df["label"]
    X_te, y_te = test_df[features], test_df["label"]

    # Shuffled label baseline
    y_shuf = np.random.default_rng(42).permutation(y_tr)
    m_shuf = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42))
    m_shuf.fit(X_tr, y_shuf)
    shuf_auc = roc_auc_score(y_val, m_shuf.predict_proba(X_val)[:, 1])
    print(f"Shuffled-label baseline Val AUC: {shuf_auc:.4f} (expected ~0.50)")

    # Candidate models
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
        "gradient_boosting": HistGradientBoostingClassifier(max_iter=150, class_weight="balanced", random_state=42),
    }

    best_name, best_model, best_val_auc = None, None, -1.0
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        print(f"  {name:<22}: Val AUC = {val_auc:.4f}")
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_name = name
            best_model = model

    test_auc = roc_auc_score(y_te, best_model.predict_proba(X_te)[:, 1])
    print(f"Selected: {best_name} | Test Game-Level AUC = {test_auc:.4f}")

    os.makedirs("models", exist_ok=True)
    bundle = {"model": best_model, "features": features, "name": best_name, "depth": config.ENGINE_DEPTH, "max_elo": config.MAX_ELO}
    joblib.dump(bundle, "models/cheat_model.joblib")
    return best_model, val_df, test_df


def run_full_evaluation(model, val_df, test_df):
    print("\n--- Running Player Simulator with Bootstrap CIs ---")
    from tests.evaluate_model import PlayerSimulator, tune_aggregation_and_threshold, evaluate_test_set

    features = ENGINE_FEATURE_NAMES
    best_agg, th_1, th_2 = tune_aggregation_and_threshold(val_df, model, features)
    results = evaluate_test_set(test_df, model, features, best_agg, th_1, th_2)
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="dataset.csv")
    parser.add_argument("--max-games", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ensure_stockfish()
    selected_games = load_dataset_stratified(args.dataset, max_games=args.max_games)
    features_df = run_parallel_extraction(selected_games, workers=args.workers)
    model, val_df, test_df = train_models(features_df)
    run_full_evaluation(model, val_df, test_df)
    print("\nPipeline execution complete! Check models/cheat_model.joblib and tests/reports/ for results.")


if __name__ == "__main__":
    main()
