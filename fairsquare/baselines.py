import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Any
from fairsquare.config import BASELINES_PATH, BASELINES_DIR, get_rating_band_label

class RatingBaselines:
    def __init__(self, baselines_path: str = BASELINES_PATH):
        self.baselines_path = baselines_path
        self.baselines = self.load_baselines()

    def _generate_default_baselines(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Sensible baseline distributions learned for human chess players across rating bands.
        """
        bands = ["<1200", "1200-1400", "1400-1600", "1600-1800", "1800-2000", "2000+"]
        defaults = {}
        for band in bands:
            if band == "<1200":
                wpl_m, wpl_s = 0.12, 0.04
                cpl_m, cpl_s = 65.0, 20.0
                t1_m, t1_s = 0.38, 0.08
                crit_ratio_m, crit_ratio_s = 2.4, 0.6
            elif band == "1200-1400":
                wpl_m, wpl_s = 0.10, 0.03
                cpl_m, cpl_s = 52.0, 15.0
                t1_m, t1_s = 0.42, 0.07
                crit_ratio_m, crit_ratio_s = 2.1, 0.5
            elif band == "1400-1600":
                wpl_m, wpl_s = 0.08, 0.025
                cpl_m, cpl_s = 42.0, 12.0
                t1_m, t1_s = 0.46, 0.06
                crit_ratio_m, crit_ratio_s = 1.9, 0.45
            elif band == "1600-1800":
                wpl_m, wpl_s = 0.065, 0.02
                cpl_m, cpl_s = 34.0, 10.0
                t1_m, t1_s = 0.52, 0.05
                crit_ratio_m, crit_ratio_s = 1.7, 0.4
            elif band == "1800-2000":
                wpl_m, wpl_s = 0.05, 0.015
                cpl_m, cpl_s = 26.0, 8.0
                t1_m, t1_s = 0.58, 0.05
                crit_ratio_m, crit_ratio_s = 1.5, 0.35
            else:  # 2000+
                wpl_m, wpl_s = 0.038, 0.012
                cpl_m, cpl_s = 19.0, 6.0
                t1_m, t1_s = 0.65, 0.04
                crit_ratio_m, crit_ratio_s = 1.35, 0.3

            defaults[band] = {
                "avg_wpl": {"mean": wpl_m, "std": wpl_s},
                "std_wpl": {"mean": wpl_s * 1.2, "std": wpl_s * 0.4},
                "avg_cpl": {"mean": cpl_m, "std": cpl_s},
                "t1_match_rate": {"mean": t1_m, "std": t1_s},
                "t3_match_rate": {"mean": min(0.92, t1_m + 0.25), "std": t1_s * 0.8},
                "near_zero_cpl_pct": {"mean": min(0.85, t1_m * 0.70), "std": 0.08},
                "blunder_rate": {"mean": max(0.02, 0.20 - t1_m * 0.25), "std": 0.04},
                "max_t1_streak": {"mean": 3.0 + (t1_m * 4.0), "std": 1.5},
                "wpl_middlegame": {"mean": wpl_m * 1.1, "std": wpl_s},
                "wpl_endgame": {"mean": wpl_m * 0.9, "std": wpl_s},
                "wpl_low_crit": {"mean": wpl_m * 0.6, "std": wpl_s * 0.7},
                "wpl_high_crit": {"mean": wpl_m * 1.5, "std": wpl_s * 1.2},
                "t1_low_crit": {"mean": min(0.95, t1_m + 0.10), "std": t1_s},
                "t1_high_crit": {"mean": max(0.20, t1_m - 0.12), "std": t1_s},
                "blunder_high_crit": {"mean": max(0.05, 0.30 - t1_m * 0.3), "std": 0.05},
                "criticality_sensitivity_ratio": {"mean": crit_ratio_m, "std": crit_ratio_s},
                "criticality_t1_delta": {"mean": (max(0.20, t1_m - 0.12) - min(0.95, t1_m + 0.10)), "std": 0.08},
                "high_crit_move_pct": {"mean": 0.22, "std": 0.06},
                "avg_move_time": {"mean": 10.0, "std": 3.5},
                "std_move_time": {"mean": 6.5, "std": 2.0},
                "move_time_cv": {"mean": 0.65, "std": 0.15},
                "time_crit_corr": {"mean": 0.30, "std": 0.12},
                "fast_critical_pct": {"mean": 0.08, "std": 0.04}
            }
        return defaults

    def load_baselines(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        if os.path.exists(self.baselines_path):
            try:
                return joblib.load(self.baselines_path)
            except Exception:
                pass
        return self._generate_default_baselines()

    def build_and_save_baselines(self, feature_list_by_band: Dict[str, List[Dict[str, float]]]):
        """
        Builds baseline mean/std dictionary from extracted feature dictionaries grouped by rating band.
        """
        os.makedirs(BASELINES_DIR, exist_ok=True)
        new_baselines = {}
        for band_label, features_list in feature_list_by_band.items():
            if not features_list:
                continue
            df = pd.DataFrame(features_list)
            band_stats = {}
            for col in df.columns:
                if col in ["player_elo"]:
                    continue
                mean_val = float(df[col].mean())
                std_val = float(df[col].std())
                if np.isnan(std_val) or std_val < 1e-6:
                    std_val = 1e-4
                band_stats[col] = {"mean": mean_val, "std": std_val}
            new_baselines[band_label] = band_stats

        self.baselines = new_baselines
        joblib.dump(self.baselines, self.baselines_path)

    def normalize_features(self, features: Dict[str, float], elo_offset: float = 0.0) -> Dict[str, float]:
        """
        Converts raw game features into rating-relative Z-Scores:
        z = (x - mean) / std
        """
        aligned_elo = features.get("player_elo", 1500.0) + elo_offset
        band_label = get_rating_band_label(aligned_elo)

        band_stats = self.baselines.get(band_label, self.baselines.get("1400-1600"))

        normalized = {
            "player_elo": aligned_elo,
            "rating_band": band_label
        }

        for key, val in features.items():
            if key in ["player_elo"]:
                continue
            if key in band_stats:
                mean = band_stats[key]["mean"]
                std = band_stats[key]["std"]
                z_score = (val - mean) / (std + 1e-6)
                normalized[f"z_{key}"] = float(z_score)
                normalized[f"raw_{key}"] = float(val)
                normalized[f"base_mean_{key}"] = float(mean)
            else:
                normalized[key] = val

        return normalized
