import os
import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from phase2_features import build_move_features
from phase4_game_model import build_game_level_features

def train_and_save_pipeline_models():
    """
    Trains the final move-level and game-level classifiers on the real tournament dataset
    and serializes them to the models/ directory for CLI inference.
    """
    os.makedirs("models", exist_ok=True)

    # 1. Load Real Dataset
    df_real = pd.read_parquet("chess_fraud.parquet")
    df_real = df_real[df_real["is_used"] == True].copy()

    # 2. Train Move-Level Model
    X_move = build_move_features(df_real)
    y_move = df_real["is_cheating_move"].values.astype(int)

    move_clf = HistGradientBoostingClassifier(random_state=42)
    move_clf.fit(X_move, y_move)
    joblib.dump(move_clf, "models/move_model.joblib")
    print("Move-level classifier saved to models/move_model.joblib")

    # 3. Predict Move Probabilities & Build Game Features
    move_probs = move_clf.predict_proba(X_move)[:, 1]
    df_game = build_game_level_features(df_real, move_probs)

    game_feature_cols = [
        "mean_move_score", "max_move_score", "top10_mean_score",
        "max_sf_streak", "streaks_len3_plus_count", "critical_match_rate",
        "game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"
    ]

    X_game = df_game[game_feature_cols]
    y_game = df_game["label"].values

    game_clf = HistGradientBoostingClassifier(random_state=42)
    game_clf.fit(X_game, y_game)
    joblib.dump(game_clf, "models/game_model.joblib")
    print("Game-level classifier saved to models/game_model.joblib")

if __name__ == "__main__":
    train_and_save_pipeline_models()
