import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from phase2_features import build_move_features
from phase3_move_model import evaluate_group_kfold

def build_game_level_features(df_moves: pd.DataFrame, move_probs: np.ndarray) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Phase 4: Aggregates move-level predictions and move features into a game-level feature matrix.
    """
    df_moves = df_moves.copy()
    df_moves["move_prob"] = move_probs

    game_records = []
    
    for game_id, group in df_moves.groupby("game_id"):
        player_id = group["player_id"].iloc[0]
        label = int(group["is_cheating_player_game"].iloc[0])
        probs = group["move_prob"].values
        sf15_matches = (group["move_player"] == group["move_stockfish_15"]).values.astype(int)

        # 1. Probabilities Aggregation
        mean_move_score = float(np.mean(probs))
        max_move_score = float(np.max(probs))
        
        # Top 10 mean score
        sorted_probs = np.sort(probs)[::-1]
        top10_mean_score = float(np.mean(sorted_probs[:min(10, len(sorted_probs))]))

        # 2. Engine Matching Streaks
        max_sf_streak = 0
        curr_streak = 0
        streaks_len3_plus_count = 0

        for match in sf15_matches:
            if match == 1:
                curr_streak += 1
                max_sf_streak = max(max_sf_streak, curr_streak)
            else:
                if curr_streak >= 3:
                    streaks_len3_plus_count += 1
                curr_streak = 0
        if curr_streak >= 3:
            streaks_len3_plus_count += 1

        # 3. Critical Position Match Rate (large eval swings or normalized loss > 0.05)
        critical_mask = (group["normalized_centipawn_loss"] >= 0.05) | (np.abs(group["eval_before"]) >= 200)
        critical_group = group[critical_mask]
        if not critical_group.empty:
            critical_match_rate = float((critical_group["move_player"] == critical_group["move_stockfish_15"]).mean())
        else:
            critical_match_rate = float((group["move_player"] == group["move_stockfish_15"]).mean())

        # 4. Centipawn Loss Averages
        game_avg_cpl = float(group["centipawn_loss"].mean())
        game_avg_norm_cpl = float(group["normalized_centipawn_loss"].mean())
        game_sf15_match_rate = float((group["move_player"] == group["move_stockfish_15"]).mean())

        game_records.append({
            "game_id": game_id,
            "player_id": player_id,
            "label": label,
            "mean_move_score": mean_move_score,
            "max_move_score": max_move_score,
            "top10_mean_score": top10_mean_score,
            "max_sf_streak": float(max_sf_streak),
            "streaks_len3_plus_count": float(streaks_len3_plus_count),
            "critical_match_rate": critical_match_rate,
            "game_avg_cpl": game_avg_cpl,
            "game_avg_norm_cpl": game_avg_norm_cpl,
            "game_sf15_match_rate": game_sf15_match_rate
        })

    df_game = pd.DataFrame(game_records)
    return df_game

def evaluate_game_level_models():
    print("=== PHASE 4: GAME-LEVEL SUSPICION SCORE MODEL ===")

    # Load Real Data
    df_real = pd.read_parquet("chess_fraud.parquet")
    df_real = df_real[df_real["is_used"] == True].copy()

    X_move_feats = build_move_features(df_real)
    y_move = df_real["is_cheating_move"].values.astype(int)
    groups_move = df_real["player_id"].values

    # Step 1: Obtain out-of-fold move probability predictions using GroupKFold
    res_move = evaluate_group_kfold(
        X_move_feats, y_move, groups_move,
        HistGradientBoostingClassifier, {"random_state": 42}
    )
    move_probs = res_move["oof_probs"]

    # Step 2: Build game-level feature matrix
    df_game = build_game_level_features(df_real, move_probs)

    y_game = df_game["label"].values
    groups_game = df_game["player_id"].values

    # Baseline Model: Using ONLY game_avg_cpl, game_avg_norm_cpl, game_sf15_match_rate
    baseline_cols = ["game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"]
    X_baseline = df_game[baseline_cols]

    res_game_baseline = evaluate_group_kfold(
        X_baseline, y_game, groups_game,
        HistGradientBoostingClassifier, {"random_state": 42}
    )

    # Full Model: Using all advanced aggregated features
    full_cols = [
        "mean_move_score", "max_move_score", "top10_mean_score",
        "max_sf_streak", "streaks_len3_plus_count", "critical_match_rate",
        "game_avg_cpl", "game_avg_norm_cpl", "game_sf15_match_rate"
    ]
    X_full = df_game[full_cols]

    res_game_full = evaluate_group_kfold(
        X_full, y_game, groups_game,
        HistGradientBoostingClassifier, {"random_state": 42}
    )

    print("\n1. Baseline Game Classifier (Average Match Rate & CPL Only):")
    print(f"   Precision: {res_game_baseline['precision']:.4f}")
    print(f"   Recall:    {res_game_baseline['recall']:.4f}")
    print(f"   F1 Score:  {res_game_baseline['f1_score']:.4f}")
    print(f"   ROC-AUC:   {res_game_baseline['roc_auc']:.4f}")

    print("\n2. Advanced Game Classifier (Top-10 Suspicion, Engine Streaks & Critical Match):")
    print(f"   Precision: {res_game_full['precision']:.4f}")
    print(f"   Recall:    {res_game_full['recall']:.4f}")
    print(f"   F1 Score:  {res_game_full['f1_score']:.4f}")
    print(f"   ROC-AUC:   {res_game_full['roc_auc']:.4f}")

    return res_game_full, full_cols

if __name__ == "__main__":
    from typing import Tuple
    evaluate_game_level_models()
