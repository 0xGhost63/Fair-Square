import numpy as np
import pandas as pd
from typing import List, Dict, Any

def cp_to_win_probability(cp: float) -> float:
    """
    Converts centipawn score to Win Probability (0.0 to 1.0) using standard logistic formula.
    """
    cp_clamped = max(-1000.0, min(1000.0, float(cp)))
    return 1.0 / (1.0 + 10.0 ** (-cp_clamped / 400.0))

class FeatureExtractor:
    @staticmethod
    def extract_game_features(move_records: List[Dict[str, Any]], player_elo: float = 1500.0) -> Dict[str, float]:
        """
        Extracts comprehensive game-level feature vector including position criticality and move time clock features.
        """
        if not move_records:
            return FeatureExtractor._get_default_features(player_elo)

        df = pd.DataFrame(move_records)

        # 1. Basic Loss Metrics
        df["best_wp"] = df["best_cp"].apply(cp_to_win_probability)
        df["played_wp"] = df["played_cp"].apply(cp_to_win_probability)
        df["second_best_wp"] = df["second_best_cp"].apply(cp_to_win_probability)

        df["wpl"] = (df["best_wp"] - df["played_wp"]).clip(lower=0.0)
        df["cpl"] = (df["best_cp"] - df["played_cp"]).clip(lower=0.0)

        df["is_t1"] = (df["played_rank"] == 1).astype(float)
        df["is_t3"] = (df["played_rank"] <= 3).astype(float)
        df["is_near_zero_cpl"] = (df["cpl"] <= 10.0).astype(float)
        df["is_blunder"] = (df["wpl"] > 0.15).astype(float)
        df["is_inaccuracy"] = (df["wpl"] > 0.05).astype(float)

        # 2. Position Criticality Definition
        df["criticality_wp"] = (df["best_wp"] - df["second_best_wp"]).clip(lower=0.0)

        df["criticality_band"] = "low"
        df.loc[df["criticality_wp"] >= 0.05, "criticality_band"] = "medium"
        df.loc[df["criticality_wp"] >= 0.15, "criticality_band"] = "high"

        # 3. Game Phase Breakdown
        df["phase"] = "middlegame"
        df.loc[df["move_number"] <= 15, "phase"] = "opening"
        df.loc[df["move_number"] >= 35, "phase"] = "endgame"

        # Aggregation
        total_moves = len(df)
        avg_wpl = float(df["wpl"].mean())
        std_wpl = float(df["wpl"].std()) if total_moves > 1 else 0.0
        avg_cpl = float(df["cpl"].mean())
        t1_match_rate = float(df["is_t1"].mean())
        t3_match_rate = float(df["is_t3"].mean())
        near_zero_cpl_pct = float(df["is_near_zero_cpl"].mean())
        blunder_rate = float(df["is_blunder"].mean())

        # Engine Move Streaks
        max_t1_streak = 0
        curr_streak = 0
        for val in df["is_t1"]:
            if val == 1.0:
                curr_streak += 1
                max_t1_streak = max(max_t1_streak, curr_streak)
            else:
                curr_streak = 0

        # Phase-specific WPL & T1 match rates
        mg_df = df[df["phase"] == "middlegame"]
        eg_df = df[df["phase"] == "endgame"]

        wpl_middlegame = float(mg_df["wpl"].mean()) if not mg_df.empty else avg_wpl
        wpl_endgame = float(eg_df["wpl"].mean()) if not eg_df.empty else avg_wpl

        # Criticality Band Features
        low_crit_df = df[df["criticality_band"] == "low"]
        med_crit_df = df[df["criticality_band"] == "medium"]
        high_crit_df = df[df["criticality_band"] == "high"]

        wpl_low_crit = float(low_crit_df["wpl"].mean()) if not low_crit_df.empty else avg_wpl
        wpl_high_crit = float(high_crit_df["wpl"].mean()) if not high_crit_df.empty else avg_wpl

        t1_low_crit = float(low_crit_df["is_t1"].mean()) if not low_crit_df.empty else t1_match_rate
        t1_high_crit = float(high_crit_df["is_t1"].mean()) if not high_crit_df.empty else t1_match_rate

        blunder_high_crit = float(high_crit_df["is_blunder"].mean()) if not high_crit_df.empty else blunder_rate

        criticality_sensitivity_ratio = wpl_high_crit / (wpl_low_crit + 1e-4)
        criticality_t1_delta = t1_high_crit - t1_low_crit

        # 4. Move Time / Clock Consistency Features (FR-4)
        has_time_data = "move_time_sec" in df.columns and df["move_time_sec"].notnull().any()
        if has_time_data:
            valid_times = df["move_time_sec"].dropna()
            avg_move_time = float(valid_times.mean())
            std_move_time = float(valid_times.std()) if len(valid_times) > 1 else 1.0
            move_time_cv = std_move_time / (avg_move_time + 1e-4)

            # Correlation between move time and criticality gap
            time_df = df.dropna(subset=["move_time_sec"])
            if len(time_df) > 3 and time_df["move_time_sec"].std() > 1e-4 and time_df["criticality_wp"].std() > 1e-4:
                time_crit_corr = float(time_df["move_time_sec"].corr(time_df["criticality_wp"]))
                if np.isnan(time_crit_corr):
                    time_crit_corr = 0.15
            else:
                time_crit_corr = 0.15

            # Fast Critical Move Ratio (% high crit moves played in < 3s)
            high_crit_time_df = time_df[time_df["criticality_band"] == "high"]
            if not high_crit_time_df.empty:
                fast_critical_pct = float((high_crit_time_df["move_time_sec"] < 3.0).mean())
            else:
                fast_critical_pct = 0.10
        else:
            avg_move_time = 10.0
            std_move_time = 6.0
            move_time_cv = 0.60
            time_crit_corr = 0.25
            fast_critical_pct = 0.10

        return {
            "player_elo": float(player_elo),
            "total_analyzed_moves": float(total_moves),
            "avg_wpl": avg_wpl,
            "std_wpl": std_wpl,
            "avg_cpl": avg_cpl,
            "t1_match_rate": t1_match_rate,
            "t3_match_rate": t3_match_rate,
            "near_zero_cpl_pct": near_zero_cpl_pct,
            "blunder_rate": blunder_rate,
            "max_t1_streak": float(max_t1_streak),
            "wpl_middlegame": wpl_middlegame,
            "wpl_endgame": wpl_endgame,
            "wpl_low_crit": wpl_low_crit,
            "wpl_high_crit": wpl_high_crit,
            "t1_low_crit": t1_low_crit,
            "t1_high_crit": t1_high_crit,
            "blunder_high_crit": blunder_high_crit,
            "criticality_sensitivity_ratio": criticality_sensitivity_ratio,
            "criticality_t1_delta": criticality_t1_delta,
            "high_crit_move_pct": float(len(high_crit_df)) / float(total_moves),
            "avg_move_time": avg_move_time,
            "std_move_time": std_move_time,
            "move_time_cv": move_time_cv,
            "time_crit_corr": time_crit_corr,
            "fast_critical_pct": fast_critical_pct
        }

    @staticmethod
    def _get_default_features(player_elo: float = 1500.0) -> Dict[str, float]:
        return {
            "player_elo": float(player_elo),
            "total_analyzed_moves": 0.0,
            "avg_wpl": 0.08,
            "std_wpl": 0.10,
            "avg_cpl": 45.0,
            "t1_match_rate": 0.45,
            "t3_match_rate": 0.70,
            "near_zero_cpl_pct": 0.35,
            "blunder_rate": 0.12,
            "max_t1_streak": 3.0,
            "wpl_middlegame": 0.09,
            "wpl_endgame": 0.08,
            "wpl_low_crit": 0.06,
            "wpl_high_crit": 0.12,
            "t1_low_crit": 0.50,
            "t1_high_crit": 0.35,
            "blunder_high_crit": 0.20,
            "criticality_sensitivity_ratio": 2.0,
            "criticality_t1_delta": -0.15,
            "high_crit_move_pct": 0.20,
            "avg_move_time": 10.0,
            "std_move_time": 6.0,
            "move_time_cv": 0.60,
            "time_crit_corr": 0.25,
            "fast_critical_pct": 0.10
        }
