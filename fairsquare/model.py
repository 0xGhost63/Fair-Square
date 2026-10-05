import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, precision_score, recall_score, f1_score
from fairsquare.config import MODEL_PATH, MODELS_DIR

FEATURE_COLUMNS = [
    "z_avg_wpl",
    "z_std_wpl",
    "z_avg_cpl",
    "z_t1_match_rate",
    "z_t3_match_rate",
    "z_near_zero_cpl_pct",
    "z_blunder_rate",
    "z_max_t1_streak",
    "z_wpl_middlegame",
    "z_wpl_endgame",
    "z_wpl_low_crit",
    "z_wpl_high_crit",
    "z_t1_low_crit",
    "z_t1_high_crit",
    "z_blunder_high_crit",
    "z_criticality_sensitivity_ratio",
    "z_criticality_t1_delta",
    "z_high_crit_move_pct",
    "z_move_time_cv",
    "z_time_crit_corr",
    "z_fast_critical_pct"
]

class FairSquareClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.model: Optional[CalibratedClassifierCV] = None
        self.threshold: float = 0.85
        self.load_model()

    def train(
        self,
        X_df: pd.DataFrame,
        y: np.ndarray,
        target_fpr: float = 0.01
    ) -> Dict[str, float]:
        """
        Trains and calibrates the classifier on normalized feature matrix X_df and binary labels y (0=Honest, 1=Cheated).
        Determines decision threshold targeting a fixed low False Positive Rate (e.g., 1%).
        """
        # Feature columns selection
        X = X_df[FEATURE_COLUMNS].fillna(0.0).values

        # Base classifier: Gradient Boosting
        base_clf = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        )

        # Calibrated classifier for accurate probability output
        calibrated_clf = CalibratedClassifierCV(estimator=base_clf, cv=5)
        calibrated_clf.fit(X, y)
        self.model = calibrated_clf

        # Calculate probabilities
        probs = self.model.predict_proba(X)[:, 1]

        # Calculate ROC & FPR threshold
        fpr, tpr, thresholds = roc_curve(y, probs)
        roc_auc = float(roc_auc_score(y, probs))

        # Find threshold corresponding to target_fpr (e.g., 1%)
        idx = np.argmin(np.abs(fpr - target_fpr))
        raw_thresh = float(thresholds[idx]) if idx < len(thresholds) else 0.85
        if np.isinf(raw_thresh) or np.isnan(raw_thresh):
            raw_thresh = 0.85
        self.threshold = float(np.clip(raw_thresh, 0.50, 0.95))

        # Evaluate at selected threshold
        preds = (probs >= self.threshold).astype(int)
        prec = float(precision_score(y, preds, zero_division=0))
        rec = float(recall_score(y, preds, zero_division=0))
        f1 = float(f1_score(y, preds, zero_division=0))
        actual_fpr = float(np.mean(preds[y == 0] == 1))

        self.save_model()

        return {
            "roc_auc": roc_auc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "operating_threshold": self.threshold,
            "target_fpr": target_fpr,
            "actual_fpr": actual_fpr
        }

    def predict_game_probability(self, normalized_features: Dict[str, float]) -> float:
        """
        Predicts suspicion probability (0.0 to 1.0) for a single game.
        """
        if self.model is None:
            # Fallback heuristic calculation if model file not trained yet
            return self._heuristic_predict(normalized_features)

        row = [normalized_features.get(col, 0.0) for col in FEATURE_COLUMNS]
        X = np.array([row])
        prob = float(self.model.predict_proba(X)[0, 1])
        return prob

    def _heuristic_predict(self, norm_feats: Dict[str, float]) -> float:
        """
        Deterministic statistical fallback score when model file is absent.
        Combines Z-Scores of Criticality Sensitivity Ratio and T1 Match Rate.
        """
        z_t1 = norm_feats.get("z_t1_match_rate", 0.0)
        z_csr = norm_feats.get("z_criticality_sensitivity_ratio", 0.0)
        z_wpl_crit = norm_feats.get("z_wpl_high_crit", 0.0)

        # Higher T1 match rate (positive z) + Lower WPL in high criticality (negative z) => higher suspicion
        score_val = (z_t1 * 0.4) - (z_csr * 0.4) - (z_wpl_crit * 0.3)
        prob = 1.0 / (1.0 + np.exp(-score_val))
        return float(prob)

    def save_model(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump({"model": self.model, "threshold": self.threshold}, self.model_path)

    def load_model(self):
        if os.path.exists(self.model_path):
            try:
                data = joblib.load(self.model_path)
                self.model = data.get("model")
                self.threshold = data.get("threshold", 0.85)
            except Exception:
                self.model = None
