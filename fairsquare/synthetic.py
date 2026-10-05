import copy
import random
from typing import List, Dict, Any

class SyntheticCheaterGenerator:
    """
    Generates realistic engine-assisted game examples from honest human game move records.
    Simulates diverse cheating styles including move overrides and clock anomalies.
    """
    CHEATING_PROFILES = [
        "FULL_ENGINE",
        "TOGGLE_SELECTIVE",
        "CRITICAL_ONLY",
        "HUMANIZED_NOISE",
        "CLIMAX_MIDDLEGAME_CHEAT",
        "BLUNDER_AVOIDANCE_CHEAT",
        "TIME_UNIFORM_CHEAT"
    ]

    @staticmethod
    def generate_cheated_game(
        honest_move_records: List[Dict[str, Any]],
        profile: str = "TOGGLE_SELECTIVE",
        cheat_rate: float = 0.40,
        random_seed: int = 42
    ) -> List[Dict[str, Any]]:
        """
        Creates a synthetic cheated game by overriding selected move records with engine candidates
        and simulating move timing patterns.
        """
        if not honest_move_records:
            return []

        rng = random.Random(random_seed)
        cheated_records = copy.deepcopy(honest_move_records)
        current_eval_deficit = 0.0

        for record in cheated_records:
            move_num = record.get("move_number", 15)
            best_cp = record.get("best_cp", 0.0)
            candidates = record.get("candidates", [])
            cp_gap = record.get("cp_gap", 0.0)
            criticality_wp = record.get("best_wp", 0.5) - record.get("second_best_wp", 0.5)

            should_cheat = False

            if profile == "FULL_ENGINE":
                should_cheat = (rng.random() < 0.95)
            elif profile == "TOGGLE_SELECTIVE":
                should_cheat = (rng.random() < cheat_rate)
            elif profile == "CRITICAL_ONLY":
                if cp_gap >= 80.0 or criticality_wp >= 0.12:
                    should_cheat = (rng.random() < 0.85)
                else:
                    should_cheat = False
            elif profile == "HUMANIZED_NOISE":
                should_cheat = (rng.random() < 0.60)
            elif profile == "CLIMAX_MIDDLEGAME_CHEAT":
                if 15 <= move_num <= 35:
                    should_cheat = (rng.random() < 0.80)
                else:
                    should_cheat = (rng.random() < 0.10)
            elif profile == "BLUNDER_AVOIDANCE_CHEAT":
                if current_eval_deficit > 100.0 or criticality_wp >= 0.15:
                    should_cheat = (rng.random() < 0.90)
                else:
                    should_cheat = (rng.random() < 0.15)
            elif profile == "TIME_UNIFORM_CHEAT":
                should_cheat = (rng.random() < cheat_rate)

            if should_cheat and candidates:
                if profile == "HUMANIZED_NOISE" and len(candidates) > 1 and rng.random() < 0.40:
                    chosen_cand = candidates[1]
                    record["played_rank"] = 2
                else:
                    chosen_cand = candidates[0]
                    record["played_rank"] = 1

                record["played_uci"] = chosen_cand["move_uci"]
                record["played_san"] = chosen_cand["move_san"]
                record["played_cp"] = chosen_cand["cp"]
                current_eval_deficit = max(0.0, current_eval_deficit - 50.0)
            else:
                played_cp = record.get("played_cp", best_cp)
                current_eval_deficit += max(0.0, best_cp - played_cp)

            # Simulate clock anomalies for uniform cheat profiles
            if profile in ["TIME_UNIFORM_CHEAT", "FULL_ENGINE"]:
                # Uniform move time between 7.5s and 9.5s regardless of complexity
                record["move_time_sec"] = float(rng.uniform(7.5, 9.5))

        return cheated_records
