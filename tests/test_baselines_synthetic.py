import unittest
from fairsquare.baselines import RatingBaselines
from fairsquare.synthetic import SyntheticCheaterGenerator

class TestBaselinesAndSynthetic(unittest.TestCase):
    def test_baseline_normalization(self):
        baselines = RatingBaselines()
        raw_feats = {
            "player_elo": 1500,
            "avg_wpl": 0.08,
            "t1_match_rate": 0.46,
            "criticality_sensitivity_ratio": 1.9
        }
        norm = baselines.normalize_features(raw_feats, elo_offset=100)
        self.assertEqual(norm["rating_band"], "1600-1800")
        self.assertIn("z_avg_wpl", norm)
        self.assertIn("z_t1_match_rate", norm)

    def test_synthetic_cheater_generator(self):
        sample_records = [
            {
                "best_cp": 100,
                "played_cp": 0,
                "best_wp": 0.64,
                "second_best_wp": 0.50,
                "played_rank": 3,
                "candidates": [
                    {"move_uci": "e2e4", "move_san": "e4", "cp": 100},
                    {"move_uci": "d2d4", "move_san": "d4", "cp": 50}
                ]
            }
        ]
        cheated = SyntheticCheaterGenerator.generate_cheated_game(sample_records, profile="FULL_ENGINE")
        self.assertEqual(cheated[0]["played_rank"], 1)
        self.assertEqual(cheated[0]["played_uci"], "e2e4")

if __name__ == "__main__":
    unittest.main()
