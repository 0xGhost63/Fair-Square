"""Trains move-level cheat detection models with strict game-level splitting.

Features:
- Strict game_id group splitting (zero position leakage between train/val/test).
- Clean class: Sides from clean-vs-clean games (both sides '0').
- Positive class: Sides with assist_rate >= 0.30.
- Compares Logistic Regression vs HistGradientBoosting.
- Extra validation: Shuffled-label baseline (~0.50 AUC) and per-feature AUC table.
- Strictly excludes n_moves, game length, game result, and clock times.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import config
from src.features import ENGINE_FEATURE_NAMES

ART_DIR = os.path.join(config.BASE_DIR, "tests", "artifacts")
FEATURES_CSV = os.path.join(config.BASE_DIR, "data", "engine_features.csv")


def load_and_split_data(csv_path=None, test_size=0.15, val_size=0.15, random_seed=42):
    """Load engine features table and split strictly by game_id into Train, Val, and Test."""
    path = csv_path or FEATURES_CSV
    if not os.path.exists(path):
        raise FileNotFoundError(f"Feature table not found at {path}. Run src.process_dataset first.")

    df = pd.read_csv(path).dropna(subset=ENGINE_FEATURE_NAMES)

    # Clean class: clean sides from clean-vs-clean games
    # Positive class: assisted sides with assist_rate >= 0.30
    mask_clean = (df["assist_rate"] == 0.0) & (df["is_clean_vs_clean"] == True)
    mask_assisted = df["assist_rate"] >= 0.30

    trainable_df = df[mask_clean | mask_assisted].copy()
    trainable_df["label"] = (trainable_df["assist_rate"] >= 0.30).astype(int)

    # First split: (Train + Val) vs Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_seed)
    train_val_idx, test_idx = next(gss_test.split(trainable_df, trainable_df["label"], trainable_df["game_id"]))

    train_val_df = trainable_df.iloc[train_val_idx].copy()
    test_df = trainable_df.iloc[test_idx].copy()

    # Second split: Train vs Val
    val_rel_size = val_size / (1.0 - test_size)
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_rel_size, random_state=random_seed)
    train_idx, val_idx = next(gss_val.split(train_val_df, train_val_df["label"], train_val_df["game_id"]))

    train_df = train_val_df.iloc[train_idx].copy()
    val_df = train_val_df.iloc[val_idx].copy()

    # Verify zero game overlap
    train_games = set(train_df["game_id"])
    val_games = set(val_df["game_id"])
    test_games = set(test_df["game_id"])
    assert len(train_games & val_games) == 0, "Leakage between train and validation!"
    assert len(train_games & test_games) == 0, "Leakage between train and test!"
    assert len(val_games & test_games) == 0, "Leakage between validation and test!"

    print("Data splits strictly by game_id:")
    print(f"  Train: {len(train_df)} sides ({train_df['game_id'].nunique()} games, {train_df['label'].sum()} cheaters)")
    print(f"  Val:   {len(val_df)} sides ({val_df['game_id'].nunique()} games, {val_df['label'].sum()} cheaters)")
    print(f"  Test:  {len(test_df)} sides ({test_df['game_id'].nunique()} games, {test_df['label'].sum()} cheaters)")

    return train_df, val_df, test_df, df


def compute_per_feature_auc(val_df, features):
    """Compute univariate ROC AUC for each feature to evaluate individual predictive signal."""
    scores = {}
    y_val = val_df["label"]
    for f in features:
        vals = val_df[f]
        try:
            auc = roc_auc_score(y_val, vals)
            # Invert if feature is negatively correlated (e.g. ACPL lower is more suspicious)
            if auc < 0.5:
                auc = 1.0 - auc
            scores[f] = float(round(auc, 4))
        except Exception:
            scores[f] = 0.5
    return scores


def run_shuffled_label_check(X_tr, y_tr, X_val, y_val):
    """Verify that training on permuted labels gives ~0.50 AUC, proving zero data leakage."""
    y_shuffled = np.random.default_rng(42).permutation(y_tr)
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", random_state=42))
    model.fit(X_tr, y_shuffled)
    probs = model.predict_proba(X_val)[:, 1]
    return float(round(roc_auc_score(y_val, probs), 4))


def train_and_evaluate(csv_path=None):
    os.makedirs(ART_DIR, exist_ok=True)
    os.makedirs(os.path.join(config.BASE_DIR, "models"), exist_ok=True)

    train_df, val_df, test_df, full_df = load_and_split_data(csv_path)

    features = ENGINE_FEATURE_NAMES
    X_tr, y_tr = train_df[features], train_df["label"]
    X_val, y_val = val_df[features], val_df["label"]
    X_te, y_te = test_df[features], test_df["label"]

    # 1. Per-feature AUC table
    per_feat_auc = compute_per_feature_auc(val_df, features)
    print("\nPer-feature Univariate Validation ROC AUC:")
    for f, auc_val in sorted(per_feat_auc.items(), key=lambda x: x[1], reverse=True):
        print(f"  {f:<25}: {auc_val:.4f}")

    # 2. Shuffled-label leakage check
    shuffled_auc = run_shuffled_label_check(X_tr, y_tr, X_val, y_val)
    print(f"\nShuffled-label baseline ROC AUC: {shuffled_auc:.4f} (Must be ~0.50, proving zero leakage)")

    # 3. Model Candidates
    candidates = {
        "logistic_regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            max_iter=150,
            class_weight="balanced",
            random_state=42,
        ),
    }

    val_scores = {}
    trained_models = {}
    print("\nTraining candidate models...")
    for name, model in candidates.items():
        model.fit(X_tr, y_tr)
        val_probs = model.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, val_probs))
        val_scores[name] = auc
        trained_models[name] = model
        print(f"  {name:<22}: Validation ROC AUC = {auc:.4f}")

    best_name = max(val_scores, key=val_scores.get)
    best_model = trained_models[best_name]
    print(f"\nSelected best model: {best_name} (Val AUC: {val_scores[best_name]:.4f})")

    # Evaluate best model on Test set
    test_probs = best_model.predict_proba(X_te)[:, 1]
    test_auc = float(roc_auc_score(y_te, test_probs))
    print(f"Test Set ROC AUC: {test_auc:.4f}")

    # Save artifacts
    train_df.to_csv(os.path.join(ART_DIR, "train_set.csv"), index=False)
    val_df.to_csv(os.path.join(ART_DIR, "val_set.csv"), index=False)
    test_df.to_csv(os.path.join(ART_DIR, "test_set.csv"), index=False)
    full_df.to_csv(os.path.join(ART_DIR, "full_processed_set.csv"), index=False)

    train_info = {
        "features": features,
        "selected_model": best_name,
        "validation_auc": val_scores,
        "test_auc": test_auc,
        "shuffled_label_auc": shuffled_auc,
        "per_feature_auc": per_feat_auc,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "test_rows": len(test_df),
        "depth": config.ENGINE_DEPTH,
    }
    with open(os.path.join(ART_DIR, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)

    # Save production bundle
    bundle = {
        "model": best_model,
        "features": features,
        "name": best_name,
        "depth": config.ENGINE_DEPTH,
        "max_elo": config.MAX_ELO,
    }
    joblib.dump(bundle, config.MODEL_PATH)
    print(f"Exported model bundle to {config.MODEL_PATH}")

    return best_model, train_info


if __name__ == "__main__":
    train_and_evaluate()
