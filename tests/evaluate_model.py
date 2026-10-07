"""Model performance report. Run after training:  python tests/evaluate_model.py

Reads the held-out test set (players never seen in training) and writes
charts and a written report into tests/reports/.
"""
import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score, roc_curve)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

OUT = os.path.join(config.BASE_DIR, "tests", "reports")
os.makedirs(OUT, exist_ok=True)


def save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, name), dpi=130)
    plt.close()


def player_level_auc(df, probs, n_games, trials=200, seed=1):
    """Mimics the app: average n games per player, then check how well that average separates classes."""
    rng = np.random.default_rng(seed)
    df = df.assign(prob=probs)
    groups = {p: g for p, g in df.groupby("player") if len(g) >= n_games}
    if len({g.label.iloc[0] for g in groups.values()}) < 2:
        return None
    names = list(groups)
    scores, labels = [], []
    for _ in range(trials):
        p = names[rng.integers(len(names))]
        g = groups[p].sample(n_games, random_state=int(rng.integers(1_000_000)))
        scores.append(g.prob.mean())
        labels.append(g.label.iloc[0])
    return roc_auc_score(labels, scores) if len(set(labels)) > 1 else None


def main():
    bundle = joblib.load(config.MODEL_PATH)
    df = pd.read_csv(os.path.join(config.BASE_DIR, "tests", "artifacts", "test_set.csv"))
    X, y = df[bundle["features"]], df["label"]
    probs = bundle["model"].predict_proba(X)[:, 1]
    pred = (probs >= 0.5).astype(int)

    metrics = {
        "model": bundle["name"], "test_rows": len(df), "test_players": int(df.player.nunique()),
        "cheater_rows": int(y.sum()),
        "accuracy": accuracy_score(y, pred), "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred), "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, probs), "pr_auc": average_precision_score(y, probs),
        "brier_score": brier_score_loss(y, probs),
    }
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    metrics["false_positive_rate"] = fp / max(fp + tn, 1)
    metrics["player_level_auc"] = {str(n): player_level_auc(df, probs, n) for n in (5, 10, 20)}

    # charts
    fpr, tpr, _ = roc_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(fpr, tpr, label=f"AUC {metrics['roc_auc']:.3f}"); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate"); plt.title("ROC curve"); plt.legend(); save("roc_curve.png")

    prec, rec, _ = precision_recall_curve(y, probs)
    plt.figure(figsize=(5, 4)); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"Precision recall (AP {metrics['pr_auc']:.3f})"); save("precision_recall.png")

    plt.figure(figsize=(4.5, 4)); plt.imshow([[tn, fp], [fn, tp]], cmap="Purples")
    for (i, j), v in np.ndenumerate([[tn, fp], [fn, tp]]):
        plt.text(j, i, int(v), ha="center", va="center", fontsize=14)
    plt.xticks([0, 1], ["Pred clean", "Pred cheat"]); plt.yticks([0, 1], ["Clean", "Cheat"]); plt.title("Confusion matrix"); save("confusion_matrix.png")

    plt.figure(figsize=(6, 4)); plt.hist(probs[y == 0], bins=25, alpha=0.7, label="Clean"); plt.hist(probs[y == 1], bins=25, alpha=0.7, label="Cheater")
    plt.xlabel("Predicted probability"); plt.ylabel("Games"); plt.title("Score distribution"); plt.legend(); save("score_distribution.png")

    frac_pos, mean_pred = calibration_curve(y, probs, n_bins=8)
    plt.figure(figsize=(5, 4)); plt.plot(mean_pred, frac_pos, "o-"); plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("Predicted probability"); plt.ylabel("Observed cheater rate"); plt.title("Calibration"); save("calibration.png")

    model = bundle["model"]
    imp = getattr(model, "feature_importances_", None)
    if imp is None:                                    # logistic regression pipeline
        imp = np.abs(model[-1].coef_[0])
    order = np.argsort(imp)
    plt.figure(figsize=(6, 5)); plt.barh(np.array(bundle["features"])[order], np.array(imp)[order])
    plt.title("Feature importance"); save("feature_importance.png")

    json.dump(metrics, open(os.path.join(OUT, "metrics.json"), "w"), indent=2, default=float)
    lines = [
        "# Model performance report", "",
        f"Model: {metrics['model']}. Test set: {metrics['test_rows']} games from {metrics['test_players']} players the model never saw.", "",
        f"Accuracy {metrics['accuracy']:.3f}, precision {metrics['precision']:.3f}, recall {metrics['recall']:.3f}, F1 {metrics['f1']:.3f}.",
        f"ROC AUC {metrics['roc_auc']:.3f}, PR AUC {metrics['pr_auc']:.3f}, Brier score {metrics['brier_score']:.3f}.",
        f"False positive rate {metrics['false_positive_rate']:.3f}. This is the share of clean games wrongly flagged and is the number to watch.", "",
        "Player level AUC (average of n games per player, like the app does): "
        + ", ".join(f"{n} games: {v:.3f}" if v else f"{n} games: not enough data" for n, v in metrics["player_level_auc"].items()), "",
        "Charts saved next to this file: roc_curve.png, precision_recall.png, confusion_matrix.png, score_distribution.png, calibration.png, feature_importance.png.", "",
    ]
    open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
