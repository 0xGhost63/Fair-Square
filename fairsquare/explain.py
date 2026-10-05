from typing import Dict, List, Any

class FeatureExplainer:
    FEATURE_NAME_MAP = {
        "avg_wpl": "Average Win Probability Loss per move",
        "avg_cpl": "Average Centipawn Loss per move",
        "t1_match_rate": "Top 1 engine move match rate",
        "t3_match_rate": "Top 3 engine candidate match rate",
        "near_zero_cpl_pct": "Percentage of near-zero loss moves (CPL <= 10)",
        "blunder_rate": "Blunder rate (WPL > 0.15)",
        "max_t1_streak": "Maximum consecutive Top 1 engine move streak",
        "wpl_high_crit": "Loss in high criticality positions",
        "t1_high_crit": "Top 1 match rate in critical positions",
        "blunder_high_crit": "Blunder rate in critical positions",
        "criticality_sensitivity_ratio": "Criticality Sensitivity Ratio (High vs Low criticality loss)",
        "criticality_t1_delta": "Top 1 accuracy shift in critical positions",
        "move_time_cv": "Move time variation coefficient (uniform timing metric)",
        "time_crit_corr": "Correlation between move time and position criticality",
        "fast_critical_pct": "Fast critical move ratio (critical moves played under 3s)"
    }

    @staticmethod
    def explain_player_features(normalized_features: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Extracts key driving factors contributing to elevated suspicion score.
        Compares observed values against expected baseline for rating band.
        """
        explanations = []
        band_label = normalized_features.get("rating_band", "1500")

        # Feature Z-Score factors
        factor_keys = [
            ("t1_match_rate", "z_t1_match_rate", True),
            ("max_t1_streak", "z_max_t1_streak", True),
            ("near_zero_cpl_pct", "z_near_zero_cpl_pct", True),
            ("t1_high_crit", "z_t1_high_crit", True),
            ("avg_wpl", "z_avg_wpl", False),
            ("wpl_high_crit", "z_wpl_high_crit", False),
            ("criticality_sensitivity_ratio", "z_criticality_sensitivity_ratio", False),
            ("move_time_cv", "z_move_time_cv", False),          # Unnaturally low move time CV is suspicious
            ("time_crit_corr", "z_time_crit_corr", False),      # Low/negative time-crit correlation is suspicious
            ("fast_critical_pct", "z_fast_critical_pct", True), # High fast critical ratio is suspicious
            ("blunder_rate", "z_blunder_rate", False)
        ]

        for key, z_key, higher_is_suspicious in factor_keys:
            z_val = normalized_features.get(z_key, 0.0)
            raw_val = normalized_features.get(f"raw_{key}", 0.0)
            base_mean = normalized_features.get(f"base_mean_{key}", 0.0)

            abs_z = abs(z_val)
            if abs_z >= 1.0:  # Significant deviation from rating baseline
                is_suspicious_dir = (z_val > 0) if higher_is_suspicious else (z_val < 0)
                readable_name = FeatureExplainer.FEATURE_NAME_MAP.get(key, key)

                if is_suspicious_dir:
                    description = (
                        f"{readable_name} is {raw_val:.3f}, which is {abs_z:.1f} standard deviations "
                        f"beyond the expected baseline ({base_mean:.3f}) for rating band {band_label}."
                    )
                else:
                    description = (
                        f"{readable_name} is {raw_val:.3f}, within human variance relative to baseline "
                        f"({base_mean:.3f}) for rating band {band_label}."
                    )

                explanations.append({
                    "feature_key": key,
                    "feature_label": readable_name,
                    "raw_value": round(float(raw_val), 4),
                    "baseline_value": round(float(base_mean), 4),
                    "z_score": round(float(z_val), 2),
                    "is_suspicious_signal": is_suspicious_dir,
                    "description": description
                })

        # Sort explanations by absolute Z-Score magnitude
        explanations.sort(key=lambda x: abs(x["z_score"]), reverse=True)
        return explanations
