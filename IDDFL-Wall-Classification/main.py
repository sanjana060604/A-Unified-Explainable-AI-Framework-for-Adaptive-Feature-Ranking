"""
Cumulative LGBM + SHAP Feature Pruning Pipeline (with model serialization)
==========================================================================
Iteration  i  uses:
    - Data    : split_01.csv + split_02.csv + ... + split_i.csv   (concatenated)
    - Features: surviving feature set from iteration i-1

Each iteration:
    1. Train LGBM on the cumulative data using the current feature set.
    2. Compute Train + Test accuracy + precision / recall / F1.
    3. Run SHAP TreeExplainer; rank features by mean |SHAP|.
    4. Drop features with shap_importance <= 10% of the TOP feature.
    5. Pass surviving features to the next iteration.
    6. Save the per-iteration model bundle (model + label encoder +
       feature columns + class names) to disk.

At the end:
    - The FINAL bundle (iter20, all cumulative data, final feature set)
      is saved as model/final_model.pkl  for inference later.

Outputs in Cumulative_SHAP/:
    iter01/  ...  iter20/
        analysis_summary.json / .txt
        feature_importance_ranked.csv
        confusion_matrix.csv / .png
        shap_bar_plot.png
        shap_summary_plot.png
    FINAL_pipeline_history.csv
    FINAL_surviving_features.csv
    FINAL_pipeline_report.txt
    model/                                <-- NEW
        iter01_model.pkl ... iter20_model.pkl
        final_model.pkl                   (FULL bundle - use this for inference)
        final_model_only.pkl              (raw LGBMClassifier)
        inference_example.py
        manifest.json
"""

import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
import shap
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

# ==================== CONFIGURATION ====================
splits_dir       = "new_data_splits"
output_root      = "Cumulative_SHAP"
model_dir        = os.path.join(output_root, "model")    # <-- model subfolder
n_iterations     = 20
threshold_pct    = 0.10
test_size        = 0.20
random_state     = 42
top_n_for_plot   = 30

LGBM_PARAMS = dict(
    objective         = "multiclass",
    n_estimators      = 200,
    learning_rate     = 0.05,
    num_leaves        = 31,
    max_depth         = -1,
    min_child_samples = 50,
    reg_lambda        = 1.0,
    reg_alpha         = 0.0,
    subsample         = 0.85,
    subsample_freq    = 1,
    colsample_bytree  = 0.85,
    random_state      = random_state,
    n_jobs            = -1,
    verbose           = -1,
)

os.makedirs(output_root, exist_ok=True)
os.makedirs(model_dir,   exist_ok=True)


# ==================== HELPER: SHAP IMPORTANCE ====================
def compute_shap_importance(shap_values, feature_names):
    if isinstance(shap_values, list):
        shap_list = shap_values
        mean_abs = np.mean([np.abs(sv).mean(axis=0) for sv in shap_list], axis=0)
    elif hasattr(shap_values, "ndim") and shap_values.ndim == 3:
        shap_list = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
        mean_abs = np.abs(shap_values).mean(axis=(0, 2))
    else:
        shap_list = shap_values
        mean_abs = np.abs(shap_values).mean(axis=0)
    return mean_abs, shap_list


# ==================== MAIN LOOP ====================
print("\n" + "=" * 72)
print("  CUMULATIVE LGBM + SHAP FEATURE PRUNING PIPELINE (model save)")
print("  Iteration i  ->  data = split_01..split_i (concatenated)")
print(f"  Drop rule: shap_importance <= {int(threshold_pct*100)}% of top feature")
print(f"  Models saved to: {model_dir}/")
print("=" * 72)

current_features = None
cumulative_dfs   = []
history          = []

# Track the most recent fitted artifacts so the final bundle has them
last_model         = None
last_label_encoder = None
last_feature_cols  = None
last_class_names   = None
last_test_acc      = None
last_train_acc     = None
last_f1_macro      = None
last_n_train       = None
last_n_test        = None
last_iter          = None

for i in range(1, n_iterations + 1):
    tag      = f"{i:02d}"
    csv_path = os.path.join(splits_dir, f"split_{tag}.csv")
    out_dir  = os.path.join(output_root, f"iter{tag}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*72}")
    print(f"  ITERATION {tag}  ->  {out_dir}")
    print(f"{'='*72}")

    # ----- Load + accumulate -----
    df_new = pd.read_csv(csv_path)
    cumulative_dfs.append(df_new)
    df_cum = pd.concat(cumulative_dfs, ignore_index=True)
    csvs_used = [f"split_{j:02d}.csv" for j in range(1, i + 1)]
    print(f"  Cumulative rows: {len(df_cum):,}  "
          f"(using {len(csvs_used)} csv: split_01..split_{tag})")

    # ----- Apply current feature set -----
    y_raw = df_cum["Class"].values
    X = df_cum.drop(columns=["Class", "Filename"], errors="ignore")
    if current_features is not None:
        keep = [f for f in current_features if f in X.columns]
        X = X[keep]
    feature_names = X.columns.tolist()
    n_feat = len(feature_names)
    print(f"  Features in use: {n_feat}")

    # ----- Encode + split -----
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    class_names = le.classes_.tolist()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"  Train rows: {len(X_train):,}   Test rows: {len(X_test):,}")

    # ----- Train LGBM -----
    model = lgb.LGBMClassifier(num_class=len(class_names), **LGBM_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        eval_metric="multi_logloss",
        callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(0)],
    )

    # ----- Evaluate -----
    y_pred_train = model.predict(X_train)
    y_pred_test  = model.predict(X_test)
    train_acc = accuracy_score(y_train, y_pred_train)
    test_acc  = accuracy_score(y_test,  y_pred_test)
    prec_m    = precision_score(y_test, y_pred_test, average="macro", zero_division=0)
    rec_m     = recall_score(y_test,    y_pred_test, average="macro", zero_division=0)
    f1_m      = f1_score(y_test,        y_pred_test, average="macro", zero_division=0)
    print(f"  Train acc: {train_acc:.4f}   Test acc: {test_acc:.4f}   F1-macro: {f1_m:.4f}")

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_test, y_pred_test)
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_df.to_csv(os.path.join(out_dir, "confusion_matrix.csv"))
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Iter {tag} - Confusion Matrix  (test_acc={test_acc:.4f})")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "confusion_matrix.png"), dpi=180)
    plt.close()

    # ----- SHAP -----
    print("  Running SHAP TreeExplainer ...")
    explainer = shap.TreeExplainer(model)
    shap_raw  = explainer.shap_values(X_test)
    mean_abs_shap, shap_list = compute_shap_importance(shap_raw, feature_names)

    importance_df = pd.DataFrame({
        "feature":         feature_names,
        "shap_importance": mean_abs_shap,
    }).sort_values("shap_importance", ascending=False).reset_index(drop=True)
    importance_df.insert(0, "rank", range(1, len(importance_df) + 1))
    importance_df["pct_of_top"] = (
        importance_df["shap_importance"] / importance_df["shap_importance"].iloc[0]
    )
    importance_df.to_csv(os.path.join(out_dir, "feature_importance_ranked.csv"), index=False)

    # SHAP plots
    try:
        plt.figure()
        shap.summary_plot(
            shap_list, X_test, plot_type="bar",
            class_names=class_names, feature_names=feature_names,
            show=False, max_display=min(top_n_for_plot, n_feat),
        )
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_bar_plot.png"), dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  ! shap_bar_plot failed: {e}")
        plt.close()
    try:
        plt.figure()
        per_class_total = [np.abs(sv).sum() for sv in shap_list]
        best_cls = int(np.argmax(per_class_total))
        shap.summary_plot(
            shap_list[best_cls], X_test, feature_names=feature_names,
            show=False, max_display=min(top_n_for_plot, n_feat),
        )
        plt.title(f"Iter {tag} - SHAP Summary  (class: {class_names[best_cls]})")
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, "shap_summary_plot.png"), dpi=180, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"  ! shap_summary_plot failed: {e}")
        plt.close()

    # ----- Apply 10% pruning -----
    top_imp   = importance_df["shap_importance"].iloc[0]
    threshold = top_imp * threshold_pct
    survivors = importance_df.loc[
        importance_df["shap_importance"] > threshold, "feature"
    ].tolist()
    dropped = importance_df.loc[
        importance_df["shap_importance"] <= threshold, "feature"
    ].tolist()
    print(f"  Top feature: {importance_df.iloc[0]['feature']}  "
          f"(importance={top_imp:.5f})")
    print(f"  Threshold (10%): {threshold:.5f}")
    print(f"  Dropped: {len(dropped)}   Surviving for next iter: {len(survivors)}")

    # ----- Save per-iteration model bundle -----
    per_iter_bundle = {
        "model":           model,
        "label_encoder":   le,
        "feature_columns": feature_names,    # exact order the model expects
        "class_names":     class_names,
        "iteration":       i,
        "csvs_used":       csvs_used,
        "n_train_rows":    int(len(X_train)),
        "n_test_rows":     int(len(X_test)),
        "n_features":      n_feat,
        "train_accuracy":  float(train_acc),
        "test_accuracy":   float(test_acc),
        "f1_macro":        float(f1_m),
        "top_feature":     importance_df.iloc[0]["feature"],
        "lgbm_params":     LGBM_PARAMS,
    }
    bundle_path = os.path.join(model_dir, f"iter{tag}_model.pkl")
    joblib.dump(per_iter_bundle, bundle_path)
    print(f"  Saved bundle -> {bundle_path}")

    # Track most-recent artifacts for the FINAL bundle
    last_model         = model
    last_label_encoder = le
    last_feature_cols  = feature_names
    last_class_names   = class_names
    last_test_acc      = float(test_acc)
    last_train_acc     = float(train_acc)
    last_f1_macro      = float(f1_m)
    last_n_train       = int(len(X_train))
    last_n_test        = int(len(X_test))
    last_iter          = i

    # ----- Per-iteration summary files -----
    summary = {
        "iteration":             i,
        "csvs_used":             csvs_used,
        "n_rows_cumulative":     int(len(df_cum)),
        "n_train_rows":          int(len(X_train)),
        "n_test_rows":           int(len(X_test)),
        "n_features_used":       n_feat,
        "features_used":         feature_names,
        "train_accuracy":        float(train_acc),
        "test_accuracy":         float(test_acc),
        "precision_macro":       float(prec_m),
        "recall_macro":          float(rec_m),
        "f1_macro":              float(f1_m),
        "top_feature":           importance_df.iloc[0]["feature"],
        "top_feature_importance":float(top_imp),
        "pruning_threshold":     float(threshold),
        "threshold_pct_of_top":  threshold_pct,
        "n_dropped":             len(dropped),
        "n_surviving":           len(survivors),
        "surviving_features":    survivors,
        "dropped_features":      dropped,
        "model_pkl":             bundle_path,
    }
    with open(os.path.join(out_dir, "analysis_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "analysis_summary.txt"), "w") as f:
        f.write(f"Iteration {tag} - Cumulative LGBM + SHAP Analysis\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"CSVs used (cumulative):     {', '.join(csvs_used)}\n")
        f.write(f"Cumulative rows:            {len(df_cum):,}\n")
        f.write(f"Train rows:                 {len(X_train):,}\n")
        f.write(f"Test  rows:                 {len(X_test):,}\n")
        f.write(f"# Features used:            {n_feat}\n\n")
        f.write(f"Train accuracy:             {train_acc:.4f}\n")
        f.write(f"Test  accuracy:             {test_acc:.4f}\n")
        f.write(f"Precision (macro):          {prec_m:.4f}\n")
        f.write(f"Recall    (macro):          {rec_m:.4f}\n")
        f.write(f"F1-score  (macro):          {f1_m:.4f}\n\n")
        f.write(f"Top feature:                {importance_df.iloc[0]['feature']}\n")
        f.write(f"Top feature SHAP value:     {top_imp:.6f}\n")
        f.write(f"Pruning threshold (10%):    {threshold:.6f}\n")
        f.write(f"# Dropped this iteration:   {len(dropped)}\n")
        f.write(f"# Surviving for next iter:  {len(survivors)}\n")
        f.write(f"Model bundle:               {bundle_path}\n\n")
        f.write("Surviving features (passed to next iteration):\n")
        for ft in survivors:
            f.write(f"  - {ft}\n")
        f.write("\nDropped features (this iteration):\n")
        for ft in dropped:
            f.write(f"  - {ft}\n")

    history.append({
        "iteration":       i,
        "cumulative_rows": int(len(df_cum)),
        "n_features_in":   n_feat,
        "train_acc":       float(train_acc),
        "test_acc":        float(test_acc),
        "precision_macro": float(prec_m),
        "recall_macro":    float(rec_m),
        "f1_macro":        float(f1_m),
        "top_feature":     importance_df.iloc[0]["feature"],
        "top_importance":  float(top_imp),
        "n_dropped":       len(dropped),
        "n_surviving":     len(survivors),
    })

    if len(survivors) == 0:
        print("  ! No features survived. Stopping early.")
        break

    current_features = survivors


# ==================== FINAL REPORT ====================
final_features = current_features
history_df     = pd.DataFrame(history)
history_df.to_csv(os.path.join(output_root, "FINAL_pipeline_history.csv"), index=False)

pd.DataFrame({"final_surviving_feature": final_features}).to_csv(
    os.path.join(output_root, "FINAL_surviving_features.csv"), index=False
)

with open(os.path.join(output_root, "FINAL_pipeline_report.txt"), "w") as f:
    f.write("CUMULATIVE LGBM + SHAP FEATURE PRUNING - FINAL REPORT\n")
    f.write("=" * 72 + "\n\n")
    f.write(f"Iterations executed:     {len(history)}\n")
    f.write(f"Drop rule:               shap_importance <= 10% of top (per iteration)\n")
    f.write(f"Final # surviving feats: {len(final_features)}\n\n")
    f.write("Per-iteration history:\n")
    f.write(history_df.to_string(index=False))
    f.write("\n\nFinal surviving features:\n")
    for ft in final_features:
        f.write(f"  - {ft}\n")


# ==================== SAVE FINAL MODEL BUNDLE ====================
print("\n" + "=" * 72)
print("  Saving FINAL model bundle for inference ...")
print("=" * 72)

final_bundle = {
    "model":           last_model,
    "label_encoder":   last_label_encoder,
    "feature_columns": last_feature_cols,     # exact training column order
    "class_names":     last_class_names,
    "lgbm_params":     LGBM_PARAMS,
    "trained_on":      f"cumulative split_01..split_{last_iter:02d}.csv",
    "iteration":       last_iter,
    "train_accuracy":  last_train_acc,
    "test_accuracy":   last_test_acc,
    "f1_macro":        last_f1_macro,
    "n_train_rows":    last_n_train,
    "n_test_rows":     last_n_test,
    "notes":           ("Final model from the LAST iteration of the cumulative "
                        "LGBM+SHAP pruning pipeline. Reorder inputs to "
                        "feature_columns before predicting."),
}
final_bundle_path = os.path.join(model_dir, "final_model.pkl")
final_model_only  = os.path.join(model_dir, "final_model_only.pkl")
joblib.dump(final_bundle, final_bundle_path)
joblib.dump(last_model,   final_model_only)
print(f"  Saved bundle: {final_bundle_path}")
print(f"  Saved model:  {final_model_only}")

# Inference example
inference_example = f'''"""
Inference example using the final saved model.
Run from the parent folder of '{output_root}'.
"""
import joblib
import pandas as pd

# 1) Load the bundle
bundle      = joblib.load(r"{final_bundle_path}")
model       = bundle["model"]
le          = bundle["label_encoder"]
feat_cols   = bundle["feature_columns"]
class_names = bundle["class_names"]

print(f"Loaded model with {{len(feat_cols)}} features and {{len(class_names)}} classes")
print(f"Classes: {{class_names}}")
print("Expected feature columns:")
for f in feat_cols:
    print(f"  - {{f}}")

# 2) Load new data (using last split as a demo)
new_df = pd.read_csv(r"{splits_dir}/split_20.csv")

# 3) Drop label-related cols + reorder to match training
X_new = new_df.drop(columns=["Class", "Filename"], errors="ignore")
missing = [c for c in feat_cols if c not in X_new.columns]
if missing:
    raise ValueError(f"Missing features in new data: {{missing}}")
X_new = X_new[feat_cols]    # CRITICAL: same column order

# 4) Predict
y_int = model.predict(X_new)
y_str = le.inverse_transform(y_int)
proba = model.predict_proba(X_new)

print("\\nFirst 10 predictions:", y_str[:10].tolist())

# 5) Probabilities
proba_df = pd.DataFrame(proba, columns=class_names)
print("\\nFirst 5 rows of class probabilities:")
print(proba_df.head())
'''
inference_path = os.path.join(model_dir, "inference_example.py")
with open(inference_path, "w") as f:
    f.write(inference_example)
print(f"  Saved inference example: {inference_path}")

# Manifest
manifest = {
    "final_model_bundle": os.path.basename(final_bundle_path),
    "final_model_only":   os.path.basename(final_model_only),
    "per_iter_models":    [f"iter{i:02d}_model.pkl" for i in range(1, last_iter + 1)],
    "inference_example":  os.path.basename(inference_path),
    "n_features_final":   len(last_feature_cols),
    "class_names":        last_class_names,
    "final_iteration":    last_iter,
}
with open(os.path.join(model_dir, "manifest.json"), "w") as f:
    json.dump(manifest, f, indent=2)


print("\n" + "=" * 72)
print("  PIPELINE COMPLETE")
print("=" * 72)
print(f"  Iterations executed:     {len(history)}")
print(f"  Final # surviving feats: {len(final_features)}")
print(f"  Train acc range: {history_df['train_acc'].min()*100:.2f}% - "
      f"{history_df['train_acc'].max()*100:.2f}%")
print(f"  Test  acc range: {history_df['test_acc'].min()*100:.2f}% - "
      f"{history_df['test_acc'].max()*100:.2f}%")
print(f"  All outputs in: {output_root}/")
print(f"  All models  in: {model_dir}/")
print(f"  History CSV   : {output_root}/FINAL_pipeline_history.csv")
print("=" * 72)
