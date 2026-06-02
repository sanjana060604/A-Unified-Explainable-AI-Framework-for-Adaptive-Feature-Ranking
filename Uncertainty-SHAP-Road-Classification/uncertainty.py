"""
=============================================================
  Uncertainty Quantification (UQ) — Pool B Spectral Features
  Dataset : AWR2243 mmWave radar — Road Surface Classification
  Author  : Sanjana Arigela, University of Agder, 2026

  PURPOSE
  -------
  Measure the EPISTEMIC UNCERTAINTY contribution of each
  spectral feature using a permutation-based method applied
  to the trained Random Forest ensemble.

  WHAT IS EPISTEMIC UNCERTAINTY?
  -------------------------------
  Epistemic uncertainty reflects the model's lack of
  knowledge — it is high when the model's ensemble of trees
  disagrees about a prediction. It can be reduced by
  collecting more or better data (unlike aleatoric uncertainty
  which is irreducible noise in the signal itself).

  For a Random Forest, epistemic uncertainty per sample x is:
      u(x) = (1/C) * sum_c Var_{t=1..T} [ P_t(y=c | x) ]
  where P_t is the class probability from tree t.
  High variance across trees = high epistemic uncertainty.

  WHAT IS UQ CHANGE (permutation importance for uncertainty)?
  -----------------------------------------------------------
  For each feature i:
    1. Permute feature i in the UQ subsample (shuffles its values,
       destroying any information it carries)
    2. Re-measure the mean uncertainty on the permuted data
    3. UQ_change_i = mean_uncertainty_permuted - baseline_uncertainty

  A LARGE POSITIVE UQ_change means:
    → Permuting feature i raised the model's uncertainty a lot
    → The feature was actively STABILISING the model's predictions
    → It is an epistemically important feature

  A NEAR-ZERO or NEGATIVE UQ_change means:
    → Permuting the feature barely changed uncertainty
    → The feature was not contributing to prediction stability

  This UQ signal is COMPLEMENTARY to SHAP:
    SHAP   = discriminative importance (how much does it change predictions?)
    UQ     = epistemic reliability    (how much does it stabilise predictions?)

  HOW THIS SCRIPT FITS IN THE PIPELINE
  -------------------------------------
  step1_create_splits.py  →  splits/
  step2_10fold_SHAP.py    →  poolB_SHAP_10fold/   (SHAP per fold)
  run_unc_poolB_10fold.py →  unc_poolB_10fold/    (UQ per fold)  ← YOU ARE HERE
  run_reliability_weighted_v2.py  →  reads BOTH outputs above

  INPUT FILES
  -----------
  splits/split_01.csv ... split_10.csv
  obj1_poolB_features_shuffled.csv

  OUTPUT STRUCTURE
  ----------------
  unc_poolB_10fold/
    run_01/ ... run_10/              per-fold results
      run_XX_uq_scores.csv           UQ_change per feature
      run_XX_per_class_unc.csv       uncertainty per predicted class
      run_XX_classification_report.txt  accuracy + UQ baseline
      run_XX_uq_bar.png
      run_XX_per_class_unc.png
      run_XX_confusion_matrix.png
      run_XX_violin_unc.png
      run_XX_per_class_accuracy.png
    summary/                         10-fold aggregated results
      poolB_summary_uq.csv
      poolB_accuracy_summary.csv
      poolB_mean_uq_bar.png
      poolB_rank_stability_heatmap.png
      poolB_rank_movement.png
      poolB_uq_trajectory.png
      (+ more summary plots)

  HOW TO RUN
  ----------
    python run_unc_poolB_10fold.py
    (Run step1_create_splits.py first)

  DEPENDENCIES
  ------------
    pip install scikit-learn pandas numpy matplotlib seaborn
=============================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support
)
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

BASE_DIR   = os.getcwd()
FULL_CSV   = os.path.join(BASE_DIR, 'obj1_poolB_features_shuffled.csv')
SPLITS_DIR = os.path.join(BASE_DIR, 'splits')
ROOT_OUT   = os.path.join(BASE_DIR, 'unc_poolB_10fold')
SUM_DIR    = os.path.join(ROOT_OUT, 'summary')
os.makedirs(SUM_DIR, exist_ok=True)

N_TREES    = 200   # trees in the RF — same as SHAP script for fair comparison
K_FOLDS    = 10
UQ_N       = 2000  # number of test samples used for UQ estimation per fold
              # (larger = more accurate uncertainty estimate, but slower)
UQ_REPEATS = 10   # permutation repeats per feature per fold
              # (more repeats = lower variance in UQ_change estimate)
SEED       = 42

POOL_PREFIX = 'SP_'    # feature name prefix for Pool B

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
PALETTE_8  = ['#E74C3C','#3498DB','#2ECC71','#F39C12',
              '#9B59B6','#1ABC9C','#E91E63','#607D8B']
POOL_COLOR = '#3498DB'

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

def savefig(directory, name):
    """Save and close current matplotlib figure."""
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()


def feat_color(feature_name):
    return FEAT_COLORS.get(feature_name, '#95A5A6')


def epistemic_uncertainty_rf(clf, X):
    """
    Compute per-sample epistemic uncertainty for a Random Forest.

    Method: variance of per-tree class probabilities, averaged across classes.
        u(x) = (1/C) * sum_c  Var_{t}[ P_t(y=c | x) ]

    A high value means the ensemble of trees disagrees — the model
    does not 'know' how to classify this sample confidently.

    Parameters
    ----------
    clf : fitted RandomForestClassifier
    X   : array [N, F]  samples to evaluate

    Returns
    -------
    uncertainty : array [N]  scalar uncertainty per sample
    """
    # tree_probs: [n_trees, N, n_classes]
    tree_probs = np.array([t.predict_proba(X) for t in clf.estimators_])
    # Variance across trees for each sample and class, then mean over classes
    return tree_probs.var(axis=0).mean(axis=1)   # [N]


def permutation_uncertainty(clf, X, base_unc, n_repeats=10, seed=42):
    """
    Permutation-based UQ feature importance.

    For each feature i:
      1. Permute feature i across all UQ samples (destroys its information)
      2. Re-compute mean uncertainty on the permuted data
      3. UQ_change_i = mean(permuted_unc) - baseline_unc

    Positive UQ_change = feature was reducing uncertainty (stabilising)
    Negative UQ_change = feature was adding noise (or irrelevant to stability)

    Parameters
    ----------
    clf       : fitted RandomForestClassifier
    X         : array [N, F]  UQ subsample
    base_unc  : array [N]     baseline per-sample uncertainty (pre-computed)
    n_repeats : int           how many times to permute each feature
    seed      : int           random seed

    Returns
    -------
    unc_change : array [F]  mean UQ_change per feature across repeats
    unc_std    : array [F]  standard deviation of UQ_change across repeats
    """
    rng     = np.random.default_rng(seed)
    n_feat  = X.shape[1]
    base_m  = base_unc.mean()        # scalar baseline mean uncertainty

    unc_change = np.zeros(n_feat)
    unc_std    = np.zeros(n_feat)

    for fi in range(n_feat):
        reps = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, fi] = rng.permutation(Xp[:, fi])  # destroy feature fi
            reps.append(epistemic_uncertainty_rf(clf, Xp).mean())
        reps = np.array(reps)
        unc_change[fi] = reps.mean() - base_m   # positive = was stabilising
        unc_std[fi]    = reps.std()

    return unc_change, unc_std


# ══════════════════════════════════════════════════════════════
#  STEP 1: LOAD DATA AND SPLITS
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("  Uncertainty Quantification — Pool B Spectral Features")
print(f"  UQ_N={UQ_N} samples  |  UQ_REPEATS={UQ_REPEATS} per feature")
print("=" * 65)
t_total = time.time()

if not os.path.isdir(SPLITS_DIR):
    raise FileNotFoundError(
        "splits/ not found. Run step1_create_splits.py first.")

split_files = []
for k in range(1, K_FOLDS + 1):
    fp = os.path.join(SPLITS_DIR, f'split_{k:02d}.csv')
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"Missing: {fp}")
    split_files.append(fp)

df_full   = pd.read_csv(FULL_CSV)
label_col = 'label' if 'label' in df_full.columns else \
            df_full.select_dtypes(include='object').columns[0]
skip      = {label_col, 'file_id', 'frame_id', 'index', 'Unnamed: 0'}

# Only keep Pool B spectral features (SP_* prefix)
feat_cols = [c for c in df_full.columns
             if c not in skip
             and pd.api.types.is_numeric_dtype(df_full[c])
             and c.startswith(POOL_PREFIX)]
n_feats   = len(feat_cols)

le          = LabelEncoder()
le.fit(df_full[label_col].values)
class_names = le.classes_.tolist()
n_classes   = len(class_names)

print(f"\nPool B features ({n_feats}): {feat_cols}")
print(f"Classes ({n_classes}): {class_names}")

split_dfs = [pd.read_csv(fp) for fp in split_files]
print(f"All splits loaded  (each ≈ {len(split_dfs[0]):,} rows)")


# ══════════════════════════════════════════════════════════════
#  STEP 2: 10-FOLD LOOP — TRAIN RF, COMPUTE BASELINE UQ,
#           THEN PERMUTATION UQ PER FEATURE
# ══════════════════════════════════════════════════════════════

print(f"\nRunning {K_FOLDS} folds ...")

all_run_results = []
all_uq_change   = []   # [K × F] UQ_change values
all_uq_std_mat  = []   # [K × F] standard deviation of UQ_change

for k in range(K_FOLDS):
    run_num = k + 1
    run_dir = os.path.join(ROOT_OUT, f'run_{run_num:02d}')
    os.makedirs(run_dir, exist_ok=True)
    prefix  = f'run_{run_num:02d}'

    t0 = time.time()
    print(f"\n  Fold {run_num}/{K_FOLDS}  →  run_{run_num:02d}/")

    # Build train/test splits (file-level, no temporal leakage)
    df_test  = split_dfs[k].copy()
    df_train = pd.concat(
        [split_dfs[j] for j in range(K_FOLDS) if j != k],
        ignore_index=True)

    X_train = np.nan_to_num(df_train[feat_cols].values.astype(np.float64))
    y_train = le.transform(df_train[label_col].values)
    X_test  = np.nan_to_num(df_test[feat_cols].values.astype(np.float64))
    y_test  = le.transform(df_test[label_col].values)

    # ── Train RF ──────────────────────────────────────────────
    # Same hyperparameters as step2_10fold_SHAP.py for consistency
    rf = RandomForestClassifier(
        n_estimators=N_TREES, criterion='gini', max_depth=20,
        min_samples_split=20, min_samples_leaf=10,
        max_features='sqrt', bootstrap=True,
        oob_score=False, class_weight='balanced',
        n_jobs=-1, random_state=SEED, verbose=0)
    rf.fit(X_train, y_train)

    y_train_pred = rf.predict(X_train)
    y_pred       = rf.predict(X_test)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc  = accuracy_score(y_test,  y_pred)
    macro_f1  = f1_score(y_test, y_pred, average='macro', zero_division=0)
    cm_k      = confusion_matrix(y_test, y_pred, labels=range(n_classes))
    rep_k     = classification_report(
        y_test, y_pred, target_names=class_names, zero_division=0)
    pc_acc    = np.where(cm_k.sum(axis=1) > 0,
                         cm_k.diagonal() / cm_k.sum(axis=1), 0.0)
    prec_pc, rec_pc, f1_pc, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=range(n_classes), zero_division=0)

    print(f"  RF  {time.time()-t0:.1f}s  "
          f"Train={train_acc*100:.2f}%  "
          f"Test={test_acc*100:.2f}%  F1={macro_f1:.4f}")

    # ── Baseline epistemic uncertainty ────────────────────────
    # Use a random subsample of test data to speed up computation
    # while still getting a reliable uncertainty estimate
    np.random.seed(run_num * 100 + SEED)
    uq_idx   = np.random.choice(
        len(X_test), min(UQ_N, len(X_test)), replace=False)
    X_uq     = X_test[uq_idx]
    y_uq     = y_test[uq_idx]

    # Baseline: uncertainty before any permutation
    base_unc      = epistemic_uncertainty_rf(rf, X_uq)
    base_mean_unc = base_unc.mean()
    print(f"  Baseline mean uncertainty = {base_mean_unc:.6f}")

    # Per-class baseline uncertainty
    # (shows which surface classes the model is most uncertain about)
    tree_probs_b = np.array([t.predict_proba(X_uq)
                              for t in rf.estimators_])
    var_by_class = tree_probs_b.var(axis=0)   # [N × n_classes]
    pred_uq_int  = rf.predict(X_uq)
    pc_unc = {}
    for ci, cn in enumerate(class_names):
        mask = (pred_uq_int == ci)
        pc_unc[cn] = float(var_by_class[mask].mean()) if mask.sum() > 1 else 0.0

    # ── Permutation UQ importance ─────────────────────────────
    t1 = time.time()
    unc_change, unc_std_arr = permutation_uncertainty(
        rf, X_uq, base_unc,
        n_repeats=UQ_REPEATS, seed=run_num * 7)
    print(f"  UQ   {time.time()-t1:.1f}s  "
          f"(top feature: {feat_cols[np.argmax(unc_change)]}  "
          f"UQ_change={unc_change.max():.6f})")

    all_uq_change.append(unc_change.copy())
    all_uq_std_mat.append(unc_std_arr.copy())

    # Rank features: highest UQ_change = rank 1 (most stabilising)
    ranks = pd.Series(unc_change, index=feat_cols) \
              .rank(ascending=False).astype(int)

    # Sorted UQ DataFrame for this fold
    run_uq_df = pd.DataFrame({
        'feature'   : feat_cols,
        'unc_change': unc_change,
        'unc_std'   : unc_std_arr,
        'rank'      : ranks.values,
    }).sort_values('unc_change', ascending=False).reset_index(drop=True)
    run_uq_df['final_rank'] = run_uq_df.index + 1

    # ── Save per-fold CSVs and report ─────────────────────────
    run_uq_df.to_csv(
        os.path.join(run_dir, f'{prefix}_uq_scores.csv'), index=False)

    pd.DataFrame({'class': list(pc_unc.keys()),
                  'mean_unc': list(pc_unc.values())}) \
      .to_csv(os.path.join(run_dir, f'{prefix}_per_class_unc.csv'),
              index=False)

    # Classification report — read by run_reliability_weighted_v2.py
    # IMPORTANT: format must match exactly for the accuracy parser to work
    with open(os.path.join(run_dir,
              f'{prefix}_classification_report.txt'), 'w') as f:
        f.write(f"Pool B — Spectral Shape Features\n")
        f.write(f"Split : split_{run_num:02d}.csv\n\n")
        f.write(f"Train Accuracy         : {train_acc:.4f}  "
                f"({train_acc*100:.2f}%)\n")
        f.write(f"Test  Accuracy         : {test_acc:.4f}  "
                f"({test_acc*100:.2f}%)\n")
        f.write(f"Macro F1               : {macro_f1:.4f}\n")
        f.write(f"Baseline Uncertainty   : {base_mean_unc:.6f}\n\n")
        f.write(rep_k + "\n")
        f.write("UQ Feature Ranking:\n")
        for _, row in run_uq_df.iterrows():
            f.write(f"  #{int(row['final_rank']):<4}{row['feature']:<25}"
                    f"  {row['unc_change']:>10.6f}  "
                    f"±{row['unc_std']:.6f}\n")

    # ── Per-fold plots ────────────────────────────────────────

    # UQ bar: features ranked by UQ_change, error bars = std across repeats
    fig, ax = plt.subplots(figsize=(12, max(5, n_feats * 0.5)))
    ax.barh(range(n_feats), run_uq_df['unc_change'],
            xerr=run_uq_df['unc_std'],
            color=[POOL_COLOR if v >= 0 else '#E74C3C'
                   for v in run_uq_df['unc_change']],
            edgecolor='white', lw=0.5,
            error_kw=dict(ecolor='#555', capsize=3, lw=1))
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(run_uq_df['feature'], fontsize=9)
    ax.invert_yaxis()
    ax.axvline(0, color='black', lw=0.8)
    ax.set_xlabel('UQ_change  '
                  '(positive = feature stabilised predictions)')
    ax.set_title(f'Epistemic UQ Importance — Fold {run_num:02d}\n'
                 f'Train={train_acc*100:.1f}%  '
                 f'Test={test_acc*100:.1f}%  '
                 f'Base_unc={base_mean_unc:.6f}',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_uq_bar.png')

    # Per-class uncertainty bar
    fig, ax = plt.subplots(figsize=(12, 5))
    cvals = [pc_unc[cn] for cn in class_names]
    ax.bar(range(len(class_names)), cvals,
           color=PALETTE_8[:len(class_names)],
           edgecolor='white')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=25, ha='right', fontsize=9)
    ax.set_ylabel('Mean Epistemic Uncertainty')
    ax.set_title(f'Per-Class Epistemic Uncertainty — Fold {run_num:02d}\n'
                 'Higher = model less certain about this surface class')
    for i, v in enumerate(cvals):
        ax.text(i, v * 1.02, f'{v:.6f}', ha='center', fontsize=8)
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_per_class_unc.png')

    # Violin plot: per-sample uncertainty distribution per class
    pred_labels = rf.predict(X_uq)
    unc_by_class = [base_unc[pred_labels == ci]
                    if (pred_labels == ci).sum() > 1
                    else np.array([0.0])
                    for ci in range(n_classes)]
    fig, ax = plt.subplots(figsize=(12, 5))
    vp = ax.violinplot(unc_by_class, positions=range(n_classes),
                       showmedians=True, showextrema=True)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(PALETTE_8[i % len(PALETTE_8)])
        body.set_alpha(0.7)
    ax.set_xticks(range(n_classes))
    ax.set_xticklabels(class_names, rotation=20, ha='right', fontsize=9)
    ax.set_ylabel('Epistemic Uncertainty (per sample)')
    ax.set_title(f'Uncertainty Distribution per Class — Fold {run_num:02d}\n'
                 f'Wider violin = more spread in how uncertain the model is',
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_violin_unc.png')

    n_saved = len([f for f in os.listdir(run_dir)
                   if os.path.isfile(os.path.join(run_dir, f))])
    print(f"  Saved {n_saved} files → run_{run_num:02d}/")

    all_run_results.append({
        'split_id'   : f'split_{run_num:02d}',
        'train_acc'  : train_acc,
        'test_acc'   : test_acc,
        'macro_f1'   : macro_f1,
        'base_unc'   : base_mean_unc,
        'unc_change' : unc_change.copy(),
        'unc_std'    : unc_std_arr.copy(),
        'ranks'      : ranks.values.copy(),
        'per_cls_acc': {cn: pc_acc[ci]
                        for ci, cn in enumerate(class_names)},
        'per_cls_unc': pc_unc,
    })


# ══════════════════════════════════════════════════════════════
#  STEP 3: AGGREGATE ACROSS ALL 10 FOLDS
# ══════════════════════════════════════════════════════════════

print(f"\nAggregating {K_FOLDS} folds ...")

n_runs     = len(all_run_results)
train_accs = np.array([r['train_acc'] for r in all_run_results])
test_accs  = np.array([r['test_acc']  for r in all_run_results])
f1s        = np.array([r['macro_f1']  for r in all_run_results])
base_uncs  = np.array([r['base_unc']  for r in all_run_results])

uq_mat    = np.stack(all_uq_change)   # [K × F]
uqs_mat   = np.stack(all_uq_std_mat)  # [K × F]
rank_mat  = np.stack(
    [r['ranks'] for r in all_run_results])             # [K × F]

mean_uq   = uq_mat.mean(axis=0)
std_uq    = uq_mat.std(axis=0)
mean_rank = rank_mat.mean(axis=0)
std_rank  = rank_mat.std(axis=0)

sort_idx   = np.argsort(-mean_uq)
feat_sorted = [feat_cols[i] for i in sort_idx]

mean_train = train_accs.mean(); std_train = train_accs.std()
mean_test  = test_accs.mean();  std_test  = test_accs.std()
mean_f1    = f1s.mean();        std_f1    = f1s.std()
mean_bu    = base_uncs.mean();  std_bu    = base_uncs.std()

print(f"\n  10-FOLD UQ SUMMARY")
print(f"  Train Accuracy  : {mean_train*100:.2f}% ± {std_train*100:.2f}%")
print(f"  Test  Accuracy  : {mean_test*100:.2f}% ± {std_test*100:.2f}%")
print(f"  Macro F1        : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Base Uncertainty: {mean_bu:.6f} ± {std_bu:.6f}")
print(f"\n  UQ Ranking (10-fold avg):")
for i, fi in enumerate(sort_idx):
    cv = std_uq[fi] / (mean_uq[fi] + 1e-12) * 100
    print(f"    #{i+1:<3} {feat_cols[fi]:<25} "
          f"{mean_uq[fi]:.6f} ±{std_uq[fi]:.6f}  CV={cv:.1f}%")


# ══════════════════════════════════════════════════════════════
#  STEP 4: SAVE SUMMARY OUTPUTS
# ══════════════════════════════════════════════════════════════

print(f"\nSaving summary outputs ...")

# Summary UQ CSV (used by run_reliability_weighted_v2.py via contribution_10fold)
sum_df = pd.DataFrame({
    'feature'   : feat_cols,
    'mean_uq'   : mean_uq,
    'std_uq'    : std_uq,
    'cv_uq'     : std_uq / (mean_uq + 1e-12),
    'mean_rank' : mean_rank,
    'std_rank'  : std_rank,
}).sort_values('mean_uq', ascending=False).reset_index(drop=True)
sum_df['final_rank'] = sum_df.index + 1
sum_df.to_csv(os.path.join(SUM_DIR, 'poolB_summary_uq.csv'), index=False)

# Accuracy summary CSV
acc_df = pd.DataFrame({
    'split'    : [r['split_id'] for r in all_run_results] + ['MEAN','STD'],
    'train_acc': list(train_accs) + [mean_train, std_train],
    'test_acc' : list(test_accs)  + [mean_test,  std_test],
    'macro_f1' : list(f1s)        + [mean_f1,    std_f1],
    'base_unc' : list(base_uncs)  + [mean_bu,    std_bu],
})
acc_df.to_csv(os.path.join(SUM_DIR, 'poolB_accuracy_summary.csv'), index=False)

print("  Saved: summary/poolB_summary_uq.csv")
print("  Saved: summary/poolB_accuracy_summary.csv")

# ── Summary plot: mean UQ_change bar ─────────────────────────
fig, ax = plt.subplots(figsize=(12, max(5, n_feats * 0.5)))
ax.barh(range(n_feats), sum_df['mean_uq'],
        xerr=sum_df['std_uq'],
        color=[POOL_COLOR if v >= 0 else '#E74C3C'
               for v in sum_df['mean_uq']],
        edgecolor='white', lw=0.5,
        error_kw=dict(ecolor='#555', capsize=4, lw=1.2))
ax.set_yticks(range(n_feats))
ax.set_yticklabels(sum_df['feature'], fontsize=9)
ax.invert_yaxis()
ax.axvline(0, color='black', lw=0.8)
ax.set_xlabel('Mean UQ_change ± std  (across 10 folds)\n'
              'Positive = feature stabilised predictions = epistemically important')
ax.set_title(f'Epistemic UQ Feature Ranking — 10-Fold Averaged\n'
             f'Train={mean_train*100:.2f}%  '
             f'Test={mean_test*100:.2f}% ± {std_test*100:.2f}%',
             fontsize=12, fontweight='bold')
plt.tight_layout()
savefig(SUM_DIR, 'poolB_mean_uq_bar.png')
print("  Saved: summary/poolB_mean_uq_bar.png")

# ── Summary plot: rank stability heatmap ─────────────────────
rank_sorted = rank_mat[:, sort_idx]
fig, ax = plt.subplots(figsize=(max(14, n_feats * 0.9),
                                max(5, n_runs * 0.55)))
sns.heatmap(rank_sorted, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=feat_sorted,
            yticklabels=[r['split_id'] for r in all_run_results],
            linewidths=0.3, linecolor='grey',
            cbar_kws={'label': 'Rank (1=most stabilising)'},
            ax=ax, vmin=1, vmax=n_feats)
ax.set_title(f'UQ Rank Stability (10 folds)\n'
             f'Green=rank 1 (most stabilising)  Red=lowest rank',
             fontsize=12, fontweight='bold', pad=10)
ax.set_xticklabels(feat_sorted, rotation=40, ha='right', fontsize=8)
plt.tight_layout()
savefig(SUM_DIR, 'poolB_rank_stability_heatmap.png')
print("  Saved: summary/poolB_rank_stability_heatmap.png")

# ── FINAL SUMMARY ─────────────────────────────────────────────
elapsed = time.time() - t_total
print(f"\n{'='*65}")
print(f"  ALL DONE  ({elapsed/60:.1f} min)")
print(f"{'='*65}")
print(f"  Train : {mean_train*100:.2f}% ± {std_train*100:.2f}%")
print(f"  Test  : {mean_test*100:.2f}% ± {std_test*100:.2f}%")
print(f"  F1    : {mean_f1:.4f} ± {std_f1:.4f}")
print(f"  Base Uncertainty : {mean_bu:.6f} ± {std_bu:.6f}")
print(f"\n  Output: {ROOT_OUT}/")
