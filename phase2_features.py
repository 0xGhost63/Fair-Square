import numpy as np
import pandas as pd

def win_prob(eval_cp: float) -> float:
    """
    Computes win probability from centipawn evaluation using Lichess logistic formula:
    p(e) = 1 / (1 + exp(-0.00368208 * e))
    """
    cp_clamped = np.clip(eval_cp, -1000.0, 1000.0)
    return 1.0 / (1.0 + np.exp(-0.00368208 * cp_clamped))

def build_move_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enhanced Move Feature Extractor:
    Computes move-level features including Elo-relative expected CPL scaling.
    """
    feats = pd.DataFrame(index=df.index)

    # Engine move match indicators
    feats["sf1_match"] = (df["move_player"] == df["move_stockfish_1"]).astype(float)
    feats["sf9_match"] = (df["move_player"] == df["move_stockfish_9"]).astype(float)
    feats["sf15_match"] = (df["move_player"] == df["move_stockfish_15"]).astype(float)

    eval_before = df["eval_before"].clip(-1000.0, 1000.0)
    eval_after = df["eval_after"].clip(-1000.0, 1000.0)

    feats["eval_before"] = eval_before
    feats["eval_after"] = eval_after
    feats["position_difficulty"] = np.abs(eval_before)

    if "centipawn_loss" in df.columns:
        cpl = df["centipawn_loss"].clip(lower=0.0)
    else:
        cpl = (eval_before - eval_after).clip(lower=0.0)

    if "normalized_centipawn_loss" in df.columns:
        norm_cpl = df["normalized_centipawn_loss"].clip(lower=0.0)
    else:
        norm_cpl = (win_prob(eval_before) - win_prob(eval_after)).clip(lower=0.0)

    feats["centipawn_loss"] = cpl
    feats["normalized_centipawn_loss"] = norm_cpl
    feats["is_zero_cpl"] = (cpl <= 5.0).astype(float)
    feats["is_blunder"] = (cpl >= 100.0).astype(float)
    feats["is_norm_blunder"] = (norm_cpl >= 0.15).astype(float)

    # Elo-relative baseline scaling
    elo = df["player_elo"].fillna(1500.0)
    opp_elo = df["opponent_elo"].fillna(1500.0)
    feats["player_elo"] = elo
    feats["elo_diff"] = elo - opp_elo

    expected_cpl = np.clip(80.0 - (elo - 1000.0) * 0.035, 12.0, 80.0)
    feats["cpl_diff_expected"] = cpl - expected_cpl
    feats["cpl_ratio_expected"] = cpl / (expected_cpl + 1e-4)

    feats["half_move"] = df["half_move"].fillna(21.0)
    feats["late_game"] = (df["half_move"] >= 40.0).astype(float)

    return feats

if __name__ == "__main__":
    df_real = pd.read_parquet("chess_fraud.parquet")
    df_real = df_real[df_real["is_used"] == True].copy()
    X_feats = build_move_features(df_real)
    print("Enhanced Move Features Shape:", X_feats.shape)
