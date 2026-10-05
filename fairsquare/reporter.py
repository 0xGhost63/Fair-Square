import numpy as np
from typing import List, Dict, Any
from fairsquare.features import FeatureExtractor
from fairsquare.baselines import RatingBaselines
from fairsquare.model import FairSquareClassifier
from fairsquare.explain import FeatureExplainer

class AnalysisReporter:
    def __init__(self, baselines: RatingBaselines = None, classifier: FairSquareClassifier = None):
        self.baselines = baselines or RatingBaselines()
        self.classifier = classifier or FairSquareClassifier()

    def generate_player_report(
        self,
        player_name: str,
        games_analysis_data: List[Dict[str, Any]],
        rating_offset: float = 100.0
    ) -> Dict[str, Any]:
        """
        Aggregates game-level analysis results into one per-player report.
        Calculates suspicion score, confidence rating, top contributing factors, and individual game metrics.
        """
        if not games_analysis_data:
            return {
                "player_name": player_name,
                "status": "error",
                "message": "No valid games provided for analysis."
            }

        game_scores = []
        game_reports = []
        all_raw_features = []

        total_games = len(games_analysis_data)
        user_elo = 1500.0

        for game_info in games_analysis_data:
            move_records = game_info.get("move_records", [])
            elo = float(game_info.get("target_elo", 1500.0))
            user_elo = elo

            # Extract raw game features
            raw_feats = FeatureExtractor.extract_game_features(move_records, player_elo=elo)
            all_raw_features.append(raw_feats)

            # Normalize features against rating band baselines
            norm_feats = self.baselines.normalize_features(raw_feats, elo_offset=rating_offset)

            # Predict suspicion probability
            prob = self.classifier.predict_game_probability(norm_feats)
            game_scores.append(prob)

            game_reports.append({
                "game_id": game_info.get("id", f"Game {len(game_reports)+1}"),
                "event": game_info.get("event", "Standard Game"),
                "white": game_info.get("white", "White"),
                "black": game_info.get("black", "Black"),
                "suspicion_prob": round(float(prob * 100.0), 1),
                "moves_analyzed": len(move_records),
                "avg_wpl": round(float(raw_feats["avg_wpl"]), 4),
                "t1_match_rate": round(float(raw_feats["t1_match_rate"] * 100.0), 1),
                "criticality_ratio": round(float(raw_feats["criticality_sensitivity_ratio"]), 2)
            })

        # Aggregation across all games
        avg_suspicion_prob = float(np.mean(game_scores))
        suspicion_percentage = round(avg_suspicion_prob * 100.0, 1)

        # Calculate Confidence Score based on sample size
        # 40+ games = High Confidence (100%), 10 games = 50%
        confidence_pct = min(100.0, max(20.0, round((total_games / 40.0) * 100.0, 1)))

        # Average normalized feature vector across games
        avg_raw_feats = {}
        for k in all_raw_features[0].keys():
            avg_raw_feats[k] = float(np.mean([g[k] for g in all_raw_features]))

        avg_norm_feats = self.baselines.normalize_features(avg_raw_feats, elo_offset=rating_offset)
        feature_explanations = FeatureExplainer.explain_player_features(avg_norm_feats)

        # Risk Classification
        if suspicion_percentage >= 75.0:
            risk_level = "HIGH SUSPICION"
        elif suspicion_percentage >= 45.0:
            risk_level = "MODERATE SUSPICION"
        else:
            risk_level = "LOW / NORMAL HUMAN PLAY"

        notice = (
            "NOTICE: This output represents statistical likelihood based on position criticality "
            "and move accuracy models. It is intended for research and educational purposes. "
            "It does not constitute a definitive or legal verdict of cheating."
        )

        return {
            "player_name": player_name,
            "total_games_analyzed": total_games,
            "player_elo": int(user_elo),
            "rating_offset": rating_offset,
            "aligned_rating_band": avg_norm_feats.get("rating_band", "1500-1700"),
            "suspicion_score_pct": suspicion_percentage,
            "risk_level": risk_level,
            "confidence_pct": confidence_pct,
            "top_contributing_factors": feature_explanations[:5],
            "game_breakdown": game_reports,
            "legal_notice": notice
        }
