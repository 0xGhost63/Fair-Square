import unittest
from fairsquare.features import FeatureExtractor, cp_to_win_probability

class TestFeatures(unittest.TestCase):
    def test_win_probability_conversion(self):
        # 0 cp should equal 0.5 win probability
        self.assertAlmostEqual(cp_to_win_probability(0.0), 0.5, places=3)
        # Positive cp should be > 0.5
        self.assertGreater(cp_to_win_probability(100.0), 0.5)
        # Negative cp should be < 0.5
        self.assertLess(cp_to_win_probability(-100.0), 0.5)

    def test_feature_extraction(self):
        sample_records = [
            {
                "move_number": 12,
                "played_cp": 50,
                "best_cp": 50,
                "second_best_cp": 0,
                "played_rank": 1,
                "cp_gap": 50
            },
            {
                "move_number": 13,
                "played_cp": -100,
                "best_cp": 50,
                "second_best_cp": 20,
                "played_rank": 3,
                "cp_gap": 30
            }
        ]
        feats = FeatureExtractor.extract_game_features(sample_records, player_elo=1500)
        self.assertEqual(feats["total_analyzed_moves"], 2)
        self.assertIn("criticality_sensitivity_ratio", feats)
        self.assertIn("t1_match_rate", feats)
        self.assertIn("wpl_high_crit", feats)

if __name__ == "__main__":
    unittest.main()
