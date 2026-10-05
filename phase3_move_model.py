import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from phase2_features import build_move_features

def evaluate_group_kfold(X, y, groups, model_cls, model_params=None):
    """
    Evaluates a move-level classifier using GroupKFold grouped by player_id.
    Prevents data leakage across games of the same player.
    """
    if model_params is None:
        model_params = {}

    gkf = GroupKFold(n_splits=5)
    oof_preds = np.zeros(len(y))
    oof_probs = np.zeros(len(y))

    for train_idx, val_idx in gkf.split(X, y, groups):
        X_train, y_train = X.iloc[train_idx], y[train_idx]
        X_val, y_val = X.iloc[val_idx], y[val_idx]

        clf = model_cls(**model_params)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_val)[:, 1]
        preds = (probs >= 0.50).astype(int)

        oof_probs[val_idx] = probs
        oof_preds[val_idx] = preds

    prec = precision_score(y, oof_preds, zero_division=0)
    rec = recall_score(y, oof_preds, zero_division=0)
    f1 = f1_score(y, oof_preds, zero_division=0)
    roc_auc = roc_auc_score(y, oof_probs)

    return {
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "oof_probs": oof_probs
    }

def prepare_synthetic_samples(df_synth):
    """
    Converts synthetic dataset into labeled fair (label 0) and assisted (label 1) move rows.
    - Fair example uses move_player and eval_after.
    - Assisted example uses move_stockfish_15 and eval_stockfish_15.
    """
    # 1. Fair moves (label 0)
    df_fair = df_synth.copy()
    df_fair["label"] = 0

    # 2. Assisted moves (label 1)
    df_assisted = df_synth.copy()
    df_assisted["move_player"] = df_assisted["move_stockfish_15"]
    df_assisted["eval_after"] = df_assisted["eval_stockfish_15"]
    df_assisted["centipawn_loss"] = 0.0
    df_assisted["normalized_centipawn_loss"] = 0.0
    df_assisted["label"] = 1

    df_combined = pd.concat([df_fair, df_assisted], ignore_index=True)
    X_synth = build_move_features(df_combined)
    y_synth = df_combined["label"].values

    return X_synth, y_synth

def run_phase3_experiments():
    print("=== PHASE 3: MOVE-LEVEL MODEL EVALUATION ===")

    # 1. Load Real Dataset
    df_real = pd.read_parquet("chess_fraud.parquet")
    df_real = df_real[df_real["is_used"] == True].copy()

    X_real = build_move_features(df_real)
    y_real = df_real["is_cheating_move"].values.astype(int)
    groups_real = df_real["player_id"].values

    # Model 1: Logistic Regression Baseline (Real Data GroupKFold)
    res_lr = evaluate_group_kfold(
        X_real, y_real, groups_real,
        LogisticRegression, {"max_iter": 1000, "random_state": 42}
    )
    print("\n1. Logistic Regression (GroupKFold on Real Data):")
    print(f"   Precision: {res_lr['precision']:.4f}")
    print(f"   Recall:    {res_lr['recall']:.4f}")
    print(f"   F1 Score:  {res_lr['f1_score']:.4f}")
    print(f"   ROC-AUC:   {res_lr['roc_auc']:.4f}")

    # Model 2: HistGradientBoosting (Real Data GroupKFold)
    res_hgb = evaluate_group_kfold(
        X_real, y_real, groups_real,
        HistGradientBoostingClassifier, {"random_state": 42}
    )
    print("\n2. HistGradientBoosting (GroupKFold on Real Data):")
    print(f"   Precision: {res_hgb['precision']:.4f}")
    print(f"   Recall:    {res_hgb['recall']:.4f}")
    print(f"   F1 Score:  {res_hgb['f1_score']:.4f}")
    print(f"   ROC-AUC:   {res_hgb['roc_auc']:.4f}")

    # Model 3: Pretrain on Synthetic Data -> Evaluate / Fine-tune on Real Data
    print("\n3. Synthetic Pretraining Strategy:")
    df_synth_train = pd.read_parquet("synth_train.parquet")
    df_synth_train = df_synth_train[df_synth_train["is_used"] == True].copy()

    # Subsample synthetic data to keep training fast
    df_synth_sample = df_synth_train.sample(n=50000, random_state=42)
    X_synth, y_synth = prepare_synthetic_samples(df_synth_sample)

    synth_clf = HistGradientBoostingClassifier(random_state=42)
    synth_clf.fit(X_synth, y_synth)

    # Evaluate synthetic pretrained model directly on real data
    synth_probs_real = synth_clf.predict_proba(X_real)[:, 1]
    synth_preds_real = (synth_probs_real >= 0.50).astype(int)

    prec_synth = precision_score(y_real, synth_preds_real, zero_division=0)
    rec_synth = recall_score(y_real, synth_preds_real, zero_division=0)
    f1_synth = f1_score(y_real, synth_preds_real, zero_division=0)
    auc_synth = roc_auc_score(y_real, synth_probs_real)

    print("   Zero-shot Synthetic Model evaluated on Real Data:")
    print(f"   Precision: {prec_synth:.4f}")
    print(f"   Recall:    {rec_synth:.4f}")
    print(f"   F1 Score:  {f1_synth:.4f}")
    print(f"   ROC-AUC:   {auc_synth:.4f}")

    return res_hgb, synth_clf

if __name__ == "__main__":
    run_phase3_experiments()
