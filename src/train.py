"""Trains the model. Players in the test set never appear in training (no leakage).

Usage:  python -m src.train
"""
import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from src.features import CLOCK_FEATURES, ENGINE_FEATURES

ART_DIR = os.path.join(config.BASE_DIR, "tests", "artifacts")


def main():
    df = pd.read_csv(config.DATA_PATH).dropna()
    features = CLOCK_FEATURES + [f for f in ENGINE_FEATURES if f in df.columns]
    X, y, groups = df[features], df["label"], df["player"]

    # split by player so the model is tested on people it has never seen
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(split.split(X, y, groups))
    X_tr, y_tr, g_tr = X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]

    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=5, class_weight="balanced",
                                                random_state=42, n_jobs=-1),
    }
    cv = GroupKFold(n_splits=5)
    scores = {}
    for name, model in candidates.items():
        scores[name] = float(cross_val_score(model, X_tr, y_tr, groups=g_tr, cv=cv, scoring="roc_auc").mean())
        print(f"{name}: cross-validated ROC AUC = {scores[name]:.3f}")
    best = max(scores, key=scores.get)
    model = candidates[best].fit(X_tr, y_tr)
    print("Selected model:", best)

    os.makedirs("models", exist_ok=True)
    os.makedirs(ART_DIR, exist_ok=True)
    joblib.dump({"model": model, "features": features, "name": best, "max_elo": config.MAX_ELO}, config.MODEL_PATH)
    df.iloc[test_idx].to_csv(os.path.join(ART_DIR, "test_set.csv"), index=False)
    json.dump({"cv_auc": scores, "selected": best, "train_rows": len(train_idx), "test_rows": len(test_idx)},
              open(os.path.join(ART_DIR, "train_info.json"), "w"), indent=2)
    print(f"Saved model to {config.MODEL_PATH}. Now run: python tests/evaluate_model.py")


if __name__ == "__main__":
    main()
