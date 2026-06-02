"""
=============================================================
  STEP 2 — 10-Fold Random Forest + SHAP Feature Ranking
  Dataset : AWR2243 mmWave radar — Road Surface Classification
  Author  : Sanjana Arigela, University of Agder, 2026

  PURPOSE
  -------
  Train a Random Forest (RF) classifier using file-level
  stratified 10-fold cross-validation and compute SHAP
  (SHapley Additive exPlanations) feature importance for
  each fold. SHAP tells us HOW MUCH each spectral feature
  contributes to the model's prediction decisions.

  WHY SHAP?
  ---------
  Standard RF feature importance (Gini impurity) is biased
  toward high-cardinality features and ignores interaction
  effects. SHAP uses cooperative game theory (Shapley values)
  to fairly attribute each feature's marginal contribution
  to every individual prediction, then averages across the
  test set for global importance.

  WHAT THIS SCRIPT DOES
  ---------------------
  For each of the 10 folds:
    1. Load test split (split_kk.csv) and combine other 9 as train
    2. Train RF with 200 trees (balanced class weights, no OOB)
    3. Compute train accuracy and test accuracy
    4. Run TreeSHAP on 300 test samples with 300 background samples
    5. Average |SHAP| across all classes → global feature importance
    6. Save per-fold results (metrics, SHAP ranking, plots)

  After all 10 folds:
    - Average SHAP importance across folds
    - Save combined plots and summary report

  INPUT FILES
  -----------
  splits/split_01.csv ... split_10.csv  (created by step1_create_splits.py)
  obj1_poolB_features_shuffled.csv      (full feature matrix)

  OUTPUT STRUCTURE
  ----------------
  poolB_SHAP_10fold/
    run_01/ ... run_10/          per-fold detailed results
      run_XX_metrics.txt         train/test accuracy + per-class breakdown
      run_XX_SHAP_feature_ranking.csv  SHAP scores for this fold
      run_XX_confusion_matrix.png
      run_XX_SHAP_bar.png
      run_XX_SHAP_beeswarm.png
      run_XX_per_class_accuracy.png
      run_XX_SHAP_direction.png
      run_XX_precision_recall_f1.png
    10fold_summary_report.txt    10-fold averaged results
    10fold_SHAP_averaged_ranking.csv
    10fold_accuracy_curve.png    test accuracy per fold with ±1σ band
    10fold_SHAP_bar_averaged.png averaged SHAP with error bars
    10fold_SHAP_stability_heatmap.png
    10fold_SHAP_rank_stability.png
    10fold_confusion_matrix_combined.png
    (+ more combined plots)

  HOW TO RUN
  ----------
    python step2_10fold_SHAP.py
    (Run step1_create_splits.py first to generate split files)

  DEPENDENCIES
  ------------
    pip install scikit-learn shap pandas numpy matplotlib seaborn
=============================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')           # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import shap

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
#  Change these paths/settings to adapt to a new dataset
# ══════════════════════════════════════════════════════════════

BASE_DIR   = os.getcwd()                        # working directory
FULL_CSV   = os.path.join(BASE_DIR, 'obj1_poolB_features_shuffled.csv')
SPLITS_DIR = os.path.join(BASE_DIR, 'splits')  # created by step1
ROOT_OUT   = os.path.join(BASE_DIR, 'poolB_SHAP_10fold')
os.makedirs(ROOT_OUT, exist_ok=True)

N_TREES   = 200    # number of trees in the RF ensemble
K_FOLDS   = 10     # number of cross-validation folds
SHAP_BG   = 300    # background samples used to build SHAP explainer
SHAP_TEST = 300    # test samples explained per fold
SEED      = 42     # random seed for reproducibility

# Feature colour map for consistent visualisation
FEAT_COLORS = {
    'SP_dc_component'  : '#E74C3C', 'SP_spec_entropy'  : '#E67E22',
    'SP_spec_flatness' : '#D4AC0D', 'SP_spec_centroid' : '#2ECC71',
    'SP_spec_spread'   : '#1ABC9C', 'SP_spec_skew'     : '#3498DB',
    'SP_spec_kurt'     : '#9B59B6', 'SP_band_ratio_L'  : '#E91E63',
    'SP_band_ratio_M'  : '#795548', 'SP_band_ratio_H'  : '#607D8B',
    'SP_peak_freq'     : '#F39C12', 'SP_spec_rolloff'  : '#16A085',
    'SP_spec_flux'     : '#8E44AD', 'SP_mean_abs_deriv': '#C0392B',
    'SP_zero_cross_rate':'#2980B9',
}
PALETTE_8 = ['#E74C3C','#3498DB','#2ECC71','#F39C12',
             '#9B59B6','#1ABC9C','#E91E63','#607D8B']

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10, 'axes.titlesize': 12,
    'axes.labelsize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.alpha': 0.3,
    'axes.spines.top': False, 'axes.spines.right': False,
})


# ══════════════════════════════════════════════════════════════
#  HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════

def savefig(path):
    """Save and close the current matplotlib figure."""
    plt.savefig(path, bbox_inches='tight', dpi=150)
    plt.close()


def feat_color(feature_name):
    """Return the assigned colour for a given feature name."""
    return FEAT_COLORS.get(feature_name, '#95A5A6')


# ══════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATA AND SPLIT FILES
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("  STEP 2 — 10-Fold RF + SHAP Feature Ranking")
print("  Road Surface Classification | Pool B Spectral Features")
print("=" * 65)
t_total = time.time()

print(f"\n[1/4]  Loading data and split files ...")

# Verify split files exist (created by step1_create_splits.py)
if not os.path.isdir(SPLITS_DIR):
    raise FileNotFoundError(
        f"splits/ folder not found.\nRun step1_create_splits.py first.")

split_files = []
for k in range(1, K_FOLDS + 1):
    fp = os.path.join(SPLITS_DIR, f'split_{k:02d}.csv')
    if not os.path.isfile(fp):
        raise FileNotFoundError(
            f"Missing: {fp}\nRun step1_create_splits.py first.")
    split_files.append(fp)

print(f"       Found all {K_FOLDS} split files ✓")

# Load full feature matrix to identify columns
df_full = pd.read_csv(FULL_CSV)
print(f"       Full data: {df_full.shape[0]:,} rows × {df_full.shape[1]} columns")

# Detect the label column (first non-numeric column or named 'label')
label_col = 'label' if 'label' in df_full.columns else \
            df_full.select_dtypes(include='object').columns[0]

# Select only SP_* spectral features (Pool B)
skip = {label_col, 'file_id', 'frame_id', 'index', 'Unnamed: 0'}
feat_cols = [c for c in df_full.columns
             if c not in skip
             and pd.api.types.is_numeric_dtype(df_full[c])]
n_feats = len(feat_cols)

# Encode string class labels to integers (required by sklearn)
le = LabelEncoder()
le.fit(df_full[label_col].values)
class_names = le.classes_.tolist()
n_classes   = len(class_names)

print(f"       Features ({n_feats}): {feat_cols}")
print(f"       Classes  ({n_classes}): {class_names}")

# Pre-load all split CSVs into memory to avoid repeated disk reads
split_dfs = []
for k, fp in enumerate(split_files):
    sp = pd.read_csv(fp)
    split_dfs.append(sp)
    print(f"       split_{k+1:02d}.csv : {len(sp):,} rows")

# Sanity checks: every row covered exactly once, no overlaps
total = sum(len(s) for s in split_dfs)
assert total == len(df_full), \
    f"Split rows ({total}) ≠ full data ({len(df_full)})!"

print(f"\n       Verifying zero overlap across splits ...")
all_keys = []
for sp in split_dfs:
    keys = list(zip(sp[label_col], sp['file_id'], sp['frame_id']))
    all_keys.extend(keys)
assert len(all_keys) == len(set(all_keys)), "OVERLAP DETECTED!"
print(f"       Zero overlap: PASSED ✓")
print(f"       Total rows  : {total:,} ✓")


# ══════════════════════════════════════════════════════════════
#  STEP 2: 10-FOLD RF TRAINING + SHAP COMPUTATION
# ══════════════════════════════════════════════════════════════

print(f"\n[2/4]  Running {K_FOLDS} folds ...")
print(f"       Train = 9 splits combined  |  Test = 1 split")
print(f"       SHAP background={SHAP_BG} samples, "
      f"test={SHAP_TEST} samples per fold")

# Accumulators across folds
fold_results = []                                   # per-fold metrics dict
shap_accum   = np.zeros((n_classes, K_FOLDS, n_feats))  # |SHAP| per class/fold
cm_accum     = np.zeros((n_classes, n_classes), dtype=int)  # aggregate confusion matrix

for k in range(K_FOLDS):
    run_num = k + 1
    run_dir = os.path.join(ROOT_OUT, f'run_{run_num:02d}')
    os.makedirs(run_dir, exist_ok=True)
    prefix  = os.path.join(run_dir, f'run_{run_num:02d}')

    t0 = time.time()
    print(f"\n  {'='*55}")
    print(f"  Fold {run_num}/{K_FOLDS}  →  run_{run_num:02d}/")
    print(f"  Test : split_{run_num:02d}.csv  |  "
          f"Train: remaining {K_FOLDS-1} splits")

    # ── Build train/test DataFrames ───────────────────────────
    # Test: current fold; Train: union of all other folds
    df_test  = split_dfs[k].copy()
    df_train = pd.concat(
        [split_dfs[j] for j in range(K_FOLDS) if j != k],
        ignore_index=True)

    # Convert to numpy arrays (NaN-safe float64)
    X_train = np.nan_to_num(df_train[feat_cols].values.astype(np.float64))
    y_train = le.transform(df_train[label_col].values)
    X_test  = np.nan_to_num(df_test[feat_cols].values.astype(np.float64))
    y_test  = le.transform(df_test[label_col].values)

    print(f"  Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows")

    # ── Train Random Forest ───────────────────────────────────
    # oob_score=False: we use the held-out test split instead
    # class_weight='balanced': corrects for any class imbalance
    rf = RandomForestClassifier(
        n_estimators=N_TREES,
        criterion='gini',
        max_depth=20,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',   # standard RF: sqrt(F) features per split
        bootstrap=True,
        oob_score=False,
        class_weight='balanced',
        n_jobs=-1,
        random_state=SEED,
        verbose=0
    )
    rf.fit(X_train, y_train)

    # ── Compute train and test accuracy ───────────────────────
    y_train_pred = rf.predict(X_train)
    train_acc_k  = accuracy_score(y_train, y_train_pred)
    train_f1_k   = f1_score(y_train, y_train_pred,
                             average='macro', zero_division=0)

    y_pred     = rf.predict(X_test)
    test_acc_k = accuracy_score(y_test, y_pred)
    test_f1_k  = f1_score(y_test, y_pred,
                           average='macro', zero_division=0)

    # Per-class metrics
    cm_k  = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    rep_k = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0)
    pc_k  = np.where(cm_k.sum(axis=1) > 0,
                     cm_k.diagonal() / cm_k.sum(axis=1), 0.0)
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(n_classes), zero_division=0)
    cm_accum += cm_k

    print(f"  RF  {time.time()-t0:.1f}s  "
          f"Train={train_acc_k*100:.2f}%  "
          f"Test={test_acc_k*100:.2f}%  "
          f"F1={test_f1_k:.4f}")

    # ── Compute SHAP values ───────────────────────────────────
    # TreeSHAP uses the interventional perturbation scheme:
    # it replaces a feature with a value sampled from the
    # background distribution rather than conditioning the tree
    # path, giving unbiased marginal contribution estimates.

    np.random.seed(k * 100 + SEED)
    # Sample background and explanation sets randomly
    bg_idx = np.random.choice(
        len(X_train), min(SHAP_BG, len(X_train)), replace=False)
    te_idx = np.random.choice(
        len(X_test), min(SHAP_TEST, len(X_test)), replace=False)
    X_bg = X_train[bg_idx]
    X_te = X_test[te_idx]

    t1  = time.time()
    exp = shap.TreeExplainer(
        rf,
        data=X_bg,
        feature_perturbation='interventional',  # correct for correlated features
        feature_names=feat_cols
    )
    sv_raw = exp.shap_values(X_te, check_additivity=False)

    # sv_raw is either a list (one array per class) or a 3D array
    if isinstance(sv_raw, list):
        sv_list = sv_raw                              # already list of [N×F] arrays
    elif isinstance(sv_raw, np.ndarray) and sv_raw.ndim == 3:
        sv_list = [sv_raw[:, :, c] for c in range(n_classes)]
    else:
        sv_list = [sv_raw, -sv_raw]

    print(f"  SHAP {time.time()-t1:.1f}s  "
          f"(background={SHAP_BG}, explained={SHAP_TEST})")

    # Store mean |SHAP| per class for this fold
    for ci in range(n_classes):
        shap_accum[ci, k, :] = np.abs(sv_list[ci]).mean(axis=0)

    # Global importance = mean |SHAP| averaged across all classes
    global_imp_k = np.mean(
        [np.abs(sv).mean(axis=0) for sv in sv_list], axis=0)

    # Rank features by descending global importance
    order_k = np.argsort(global_imp_k)[::-1]
    shap_df_k = pd.DataFrame({
        'rank'         : range(1, n_feats + 1),
        'feature'      : [feat_cols[i] for i in order_k],
        'mean_abs_shap': global_imp_k[order_k],
    })

    # ── Save per-fold outputs ─────────────────────────────────

    # Classification report text file
    with open(f'{prefix}_metrics.txt', 'w') as f:
        f.write("=" * 60 + f"\n  Fold {run_num:02d} / {K_FOLDS}\n"
                f"  Test split : split_{run_num:02d}.csv\n"
                + "=" * 60 + "\n\n")
        f.write(f"  Train rows : {len(X_train):,}\n")
        f.write(f"  Test rows  : {len(X_test):,}\n\n")
        # NOTE: accuracy format matches what run_unc_poolB_10fold.py reads
        f.write(f"  Train Accuracy : {train_acc_k:.4f}  "
                f"({train_acc_k*100:.2f}%)\n")
        f.write(f"  Test  Accuracy : {test_acc_k:.4f}  "
                f"({test_acc_k*100:.2f}%)\n")
        f.write(f"  Train Macro F1 : {train_f1_k:.4f}\n")
        f.write(f"  Test  Macro F1 : {test_f1_k:.4f}\n\n")
        f.write("  Per-Class Test Accuracy:\n")
        for cn, ac in zip(class_names, pc_k):
            f.write(f"    {cn:<24}: {ac*100:.2f}%\n")
        f.write(f"\n{rep_k}\n")
        f.write("  SHAP Feature Ranking:\n")
        f.write(f"  {'Rank':<5}{'Feature':<25}{'SHAP':>10}\n"
                + "-" * 42 + "\n")
        for _, row in shap_df_k.iterrows():
            f.write(f"  #{int(row['rank']):<4}{row['feature']:<25}"
                    f"  {row['mean_abs_shap']:.6f}\n")

    # SHAP ranking CSV (used by run_reliability_weighted_v2.py)
    shap_df_k.to_csv(f'{prefix}_SHAP_feature_ranking.csv', index=False)

    # ── Plot: Confusion matrix ────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 8))
    rs  = cm_k.sum(axis=1, keepdims=True)
    pct = np.where(rs > 0, cm_k / rs * 100, 0)
    sns.heatmap(pct, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                linewidths=0.5, linecolor='grey',
                cbar_kws={'label': 'Row %'}, ax=ax, vmin=0, vmax=100)
    for i in range(n_classes):
        for j in range(n_classes):
            ax.text(j + 0.5, i + 0.75, f'n={cm_k[i,j]}',
                    ha='center', va='center', fontsize=6, color='#333')
    ax.set_title(f'Fold {run_num:02d} Confusion Matrix  |  '
                 f'Train={train_acc_k*100:.1f}%  '
                 f'Test={test_acc_k*100:.1f}%',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticklabels(class_names, rotation=35, ha='right', fontsize=8)
    ax.set_yticklabels(class_names, rotation=0, fontsize=8)
    plt.tight_layout()
    savefig(f'{prefix}_confusion_matrix.png')

    # ── Plot: SHAP bar chart ──────────────────────────────────
    shap_s = shap_df_k.sort_values('mean_abs_shap', ascending=True)
    fig, ax = plt.subplots(figsize=(11, 6))
    bars = ax.barh(shap_s['feature'], shap_s['mean_abs_shap'],
                   color=[feat_color(f) for f in shap_s['feature']],
                   edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, shap_s['mean_abs_shap']):
        ax.text(val + shap_s['mean_abs_shap'].max() * 0.012,
                bar.get_y() + bar.get_height() / 2,
                f'{val:.5f}', va='center', fontsize=8)
    ax.set_xlabel('Mean |SHAP Value|  (higher = more important)', fontsize=11)
    ax.set_title(f'SHAP Feature Importance — Fold {run_num:02d}\n'
                 f'Train={train_acc_k*100:.1f}%  '
                 f'Test={test_acc_k*100:.1f}%',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0, shap_s['mean_abs_shap'].max() * 1.25)
    plt.tight_layout()
    savefig(f'{prefix}_SHAP_bar.png')

    # ── Plot: SHAP beeswarm (class-averaged SHAP distribution) ──
    sv_avg_k = np.mean(np.stack(sv_list, axis=0), axis=0)
    fig = plt.figure(figsize=(11, 7))
    shap.summary_plot(sv_avg_k, X_te,
                      feature_names=feat_cols,
                      plot_type='dot',
                      show=False,
                      max_display=n_feats,
                      color_bar_label='Feature Value')
    plt.title(f'SHAP Beeswarm — Fold {run_num:02d}  (mean over classes)',
              fontsize=12, fontweight='bold', pad=10)
    plt.tight_layout()
    savefig(f'{prefix}_SHAP_beeswarm.png')

    # ── Plot: Per-class accuracy bar ──────────────────────────
    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(class_names, pc_k * 100,
                  color=PALETTE_8[:n_classes],
                  edgecolor='white', lw=0.5)
    for bar, val in zip(bars, pc_k * 100):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.8,
                f'{val:.1f}%', ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.axhline(test_acc_k * 100, color='black', ls='--', lw=1.3,
               label=f'Test Overall={test_acc_k*100:.1f}%')
    ax.axhline(train_acc_k * 100, color='#3498DB', ls=':', lw=1.3,
               label=f'Train Overall={train_acc_k*100:.1f}%')
    ax.set_ylim(0, 115)
    ax.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax.set_title(f'Per-Class Test Accuracy — Fold {run_num:02d}',
                 fontsize=12, fontweight='bold')
    ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=8)
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(f'{prefix}_per_class_accuracy.png')

    # ── Plot: SHAP direction per class ────────────────────────
    # Red bars = pushes prediction toward this class
    # Blue bars = pushes prediction away from this class
    n_r = int(np.ceil(n_classes / 2))
    fig, axes = plt.subplots(n_r, 2, figsize=(14, n_r * 3))
    for ci, cname in enumerate(class_names):
        ax    = axes.flatten()[ci]
        sv_m  = sv_list[ci].mean(axis=0)       # mean signed SHAP per feature
        order = np.argsort(np.abs(sv_m))[::-1] # sort by magnitude
        sv_o  = sv_m[order]
        names_o = [feat_cols[i] for i in order]
        ax.bar(range(n_feats), sv_o,
               color=['#E74C3C' if v >= 0 else '#3498DB' for v in sv_o],
               edgecolor='white', lw=0.4)
        ax.set_xticks(range(n_feats))
        ax.set_xticklabels(names_o, rotation=45, ha='right', fontsize=7)
        ax.axhline(0, color='black', lw=0.7, ls='--')
        ax.set_title(cname, fontsize=9, fontweight='bold',
                     color=PALETTE_8[ci % len(PALETTE_8)])
        ax.set_ylabel('Mean SHAP', fontsize=8)
    for idx in range(n_classes, n_r * 2):
        axes.flatten()[idx].axis('off')
    fig.suptitle(f'SHAP Direction — Fold {run_num:02d}\n'
                 'Red = pushes toward class  |  Blue = pushes away',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(f'{prefix}_SHAP_direction.png')

    # ── Plot: Precision / Recall / F1 ─────────────────────────
    x_pos = np.arange(n_classes)
    w     = 0.26
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x_pos - w, prec_pc * 100, w, label='Precision',
           color='#3498DB', edgecolor='white')
    ax.bar(x_pos,     rec_pc  * 100, w, label='Recall',
           color='#2ECC71', edgecolor='white')
    ax.bar(x_pos + w, f1_pc   * 100, w, label='F1',
           color='#E74C3C', edgecolor='white')
    for xg, vals in [(x_pos-w, prec_pc),
                     (x_pos,   rec_pc),
                     (x_pos+w, f1_pc)]:
        for xp, val in zip(xg, vals):
            ax.text(xp, val * 100 + 1, f'{val*100:.1f}',
                    ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=8)
    ax.set_ylabel('Score (%)', fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_title(f'Precision / Recall / F1 — Fold {run_num:02d}',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(f'{prefix}_precision_recall_f1.png')

    n_saved = len([f for f in os.listdir(run_dir)
                   if os.path.isfile(os.path.join(run_dir, f))])
    print(f"  Saved {n_saved} files → run_{run_num:02d}/")

    fold_results.append({
        'fold'         : run_num,
        'train_rows'   : len(X_train),
        'test_rows'    : len(X_test),
        'train_acc'    : train_acc_k,
        'test_acc'     : test_acc_k,
        'train_f1'     : train_f1_k,
        'test_f1'      : test_f1_k,
        'per_class_acc': pc_k.tolist(),
        'shap_global'  : global_imp_k.tolist(),
    })


# ══════════════════════════════════════════════════════════════
#  STEP 3: AGGREGATE ACROSS ALL 10 FOLDS
# ══════════════════════════════════════════════════════════════

print(f"\n[3/4]  Aggregating {K_FOLDS} folds ...")

train_accs = np.array([r['train_acc'] for r in fold_results])
test_accs  = np.array([r['test_acc']  for r in fold_results])
test_f1s   = np.array([r['test_f1']   for r in fold_results])

mean_train = train_accs.mean(); std_train = train_accs.std()
mean_test  = test_accs.mean();  std_test  = test_accs.std()
mean_f1    = test_f1s.mean();   std_f1    = test_f1s.std()

# Per-class accuracy matrix [K × n_classes]
pc_matrix = np.array([r['per_class_acc'] for r in fold_results])
mean_pc   = pc_matrix.mean(axis=0)
std_pc    = pc_matrix.std(axis=0)

# SHAP: average |SHAP| per class, then average across classes and folds
mean_shap_per_class = shap_accum.mean(axis=1)   # [n_classes × n_feats]
mean_shap_global    = mean_shap_per_class.mean(axis=0)  # [n_feats]
std_shap_global     = np.array(
    [r['shap_global'] for r in fold_results]).std(axis=0)

# Rank by descending importance
order_desc  = np.argsort(mean_shap_global)[::-1]
shap_avg_df = pd.DataFrame({
    'rank'          : range(1, n_feats + 1),
    'feature'       : [feat_cols[i] for i in order_desc],
    'mean_abs_shap' : mean_shap_global[order_desc],
    'std_abs_shap'  : std_shap_global[order_desc],
})

print(f"\n  10-FOLD SHAP SUMMARY")
print(f"  Train Accuracy : {mean_train*100:.2f}% ± {std_train*100:.2f}%")
print(f"  Test  Accuracy : {mean_test*100:.2f}% ± {std_test*100:.2f}%")
print(f"  Test  Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"\n  SHAP Ranking (10-fold averaged):")
for _, row in shap_avg_df.iterrows():
    cv = row['std_abs_shap'] / (row['mean_abs_shap'] + 1e-10) * 100
    print(f"    #{int(row['rank']):<3} {row['feature']:<25} "
          f"{row['mean_abs_shap']:.5f} ±{row['std_abs_shap']:.5f}  "
          f"CV={cv:.1f}%")


# ══════════════════════════════════════════════════════════════
#  STEP 4: SAVE COMBINED OUTPUTS
# ══════════════════════════════════════════════════════════════

print(f"\n[4/4]  Saving combined outputs ...")

# Save averaged SHAP ranking CSV
shap_avg_df.to_csv(
    os.path.join(ROOT_OUT, '10fold_SHAP_averaged_ranking.csv'), index=False)

# Save per-fold metrics CSV
fold_df_out = pd.DataFrame([{
    'fold'            : r['fold'],
    'split_file'      : f"split_{r['fold']:02d}.csv",
    'train_rows'      : r['train_rows'],
    'test_rows'       : r['test_rows'],
    'train_accuracy'  : r['train_acc'],
    'test_accuracy'   : r['test_acc'],
    'train_f1'        : r['train_f1'],
    'test_f1'         : r['test_f1'],
    **{f"acc_{cn.replace(' ','_')}": r['per_class_acc'][ci]
       for ci, cn in enumerate(class_names)}
} for r in fold_results])
fold_df_out.to_csv(
    os.path.join(ROOT_OUT, '10fold_all_fold_metrics.csv'), index=False)

# Summary text report
with open(os.path.join(ROOT_OUT, '10fold_summary_report.txt'), 'w') as f:
    f.write("=" * 65 + "\n")
    f.write("  10-Fold RF + SHAP — Pool B Spectral Features\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"  Features    : {feat_cols}\n")
    f.write(f"  RF Trees    : {N_TREES}\n\n")
    f.write(f"  Train Accuracy : {mean_train*100:.4f}% ± "
            f"{std_train*100:.4f}%\n")
    f.write(f"  Test  Accuracy : {mean_test*100:.4f}% ± "
            f"{std_test*100:.4f}%\n")
    f.write(f"  Test  Macro F1 : {mean_f1:.4f} ± {std_f1:.4f}\n\n")
    f.write("  SHAP Ranking (10-fold avg):\n")
    for _, row in shap_avg_df.iterrows():
        cv = row['std_abs_shap'] / (row['mean_abs_shap'] + 1e-10) * 100
        f.write(f"    #{int(row['rank']):<3} {row['feature']:<25} "
                f"{row['mean_abs_shap']:.6f} ±{row['std_abs_shap']:.6f} "
                f"CV={cv:.1f}%\n")

print("  Saved: 10fold_summary_report.txt")
print("  Saved: 10fold_SHAP_averaged_ranking.csv")
print("  Saved: 10fold_all_fold_metrics.csv")

# ── Combined plot: 10-fold test accuracy curve ────────────────
fx    = [r['fold'] for r in fold_results]
x_lbl = [f'run_{k+1:02d}' for k in range(K_FOLDS)]

fig, ax = plt.subplots(figsize=(13, 5))
ax.plot(fx, test_accs * 100, 'o-', color='#E74C3C', lw=2, ms=8,
        label='Test Accuracy')
ax.fill_between(fx,
                [(mean_test - std_test) * 100] * K_FOLDS,
                [(mean_test + std_test) * 100] * K_FOLDS,
                alpha=0.12, color='#E74C3C',
                label=f'±1σ ({mean_test*100:.2f}±{std_test*100:.2f}%)')
ax.axhline(mean_test * 100, color='#E74C3C', ls='--', lw=1, alpha=0.6)
ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Test Accuracy (%)', fontsize=12)
ax.set_title(f'10-Fold Test Accuracy — Pool B\n'
             f'Mean={mean_test*100:.2f}%  F1={mean_f1:.4f}',
             fontsize=12, fontweight='bold')
ax.set_xticks(fx)
ax.set_xticklabels(x_lbl, rotation=25, ha='right', fontsize=9)
ax.set_ylim(0, 108)
ax.legend(fontsize=9)
plt.tight_layout()
savefig(os.path.join(ROOT_OUT, '10fold_accuracy_curve.png'))
print("  Saved: 10fold_accuracy_curve.png")

# ── Combined plot: averaged SHAP bar with error bars ──────────
shap_s = shap_avg_df.sort_values('mean_abs_shap', ascending=True)
fig, ax = plt.subplots(figsize=(12, 7))
ax.barh(shap_s['feature'], shap_s['mean_abs_shap'],
        color=[feat_color(f) for f in shap_s['feature']],
        edgecolor='white', lw=0.5,
        xerr=shap_s['std_abs_shap'],
        error_kw={'ecolor': '#333', 'capsize': 4, 'lw': 1.2})
for _, row in shap_s.iterrows():
    ax.text(row['mean_abs_shap'] + shap_s['mean_abs_shap'].max() * 0.015,
            list(shap_s['feature']).index(row['feature']),
            f"{row['mean_abs_shap']:.5f}",
            va='center', fontsize=8.5)
ax.set_xlabel('Mean |SHAP| averaged over 10 folds', fontsize=11)
ax.set_title(f'SHAP Feature Ranking — 10-Fold Averaged\n'
             f'Test Acc={mean_test*100:.2f}%±{std_test*100:.2f}%',
             fontsize=12, fontweight='bold')
ax.set_xlim(0, shap_s['mean_abs_shap'].max() * 1.32)
plt.tight_layout()
savefig(os.path.join(ROOT_OUT, '10fold_SHAP_bar_averaged.png'))
print("  Saved: 10fold_SHAP_bar_averaged.png")

# ── Combined plot: SHAP stability heatmap ────────────────────
shap_per_fold = np.array([r['shap_global'] for r in fold_results])
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(shap_per_fold, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=feat_cols,
            yticklabels=[f'run_{k+1:02d}' for k in range(K_FOLDS)],
            linewidths=0.3, linecolor='grey',
            cbar_kws={'label': 'Mean |SHAP|'}, ax=ax)
ax.set_title('SHAP Stability — Mean |SHAP| per Feature per Fold',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xticklabels(feat_cols, rotation=35, ha='right', fontsize=8)
plt.tight_layout()
savefig(os.path.join(ROOT_OUT, '10fold_SHAP_stability_heatmap.png'))
print("  Saved: 10fold_SHAP_stability_heatmap.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
elapsed = time.time() - t_total
print(f"\n{'='*65}")
print(f"  ALL DONE  ({elapsed/60:.1f} min)")
print(f"{'='*65}")
print(f"  Train : {mean_train*100:.2f}% ± {std_train*100:.2f}%")
print(f"  Test  : {mean_test*100:.2f}% ± {std_test*100:.2f}%")
print(f"  F1    : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"\n  Output: {ROOT_OUT}/")
