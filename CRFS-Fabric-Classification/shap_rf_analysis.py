"""
=============================================================================
RF + SHAP Analysis — Fabric Material Classification
Dataset 2: TI IWR2243 Radar | Pool B Spectral Features
=============================================================================

PURPOSE
-------
Trains a Random Forest classifier on the 15 Pool B spectral features
extracted from fabric radar measurements, computes SHAP importance
values for each feature, and evaluates cumulative accuracy by adding
features one-by-one in SHAP rank order (top-1 → top-15).

Run this script TWICE — once for each experimental condition:
  1. Translational condition  → set CSV_PATH to translational CSV
  2. Rotational condition     → set CSV_PATH to rotational CSV

FABRIC CLASSES
--------------
  cotton, fiber sofa, leather, polyster, rain coat

POOL B — 15 SPECTRAL FEATURES
------------------------------
  SP_dc_component, SP_spec_entropy, SP_spec_flatness, SP_spec_centroid,
  SP_spec_spread, SP_spec_skew, SP_spec_kurt, SP_band_ratio_L,
  SP_band_ratio_M, SP_band_ratio_H, SP_peak_freq, SP_spec_rolloff,
  SP_spec_flux, SP_mean_abs_deriv, SP_zero_cross_rate

OUTPUTS  (saved to OUT_DIR)
-------
  shap_ranking.csv              — per-feature SHAP importance + rank
  cumulative_accuracy.csv       — accuracy at each step top-1 to top-15
  shap_bar.png                  — SHAP importance bar chart
  per_class_heatmap.png         — per-class SHAP heatmap
  confusion_matrix.png          — normalised confusion matrix
  cumulative_accuracy.png       — cumulative accuracy curve

USAGE
-----
  # For translational condition:
  python shap_rf_analysis.py

  # To switch to rotational, change CSV_PATH below.

=============================================================================
"""

import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)
import shap

warnings.filterwarnings('ignore')

# =============================================================================
#  CONFIGURATION — Edit these paths before running
# =============================================================================

# Path to the feature CSV for the condition you want to analyse.
# Change this line to switch between translational and rotational.
#
#   Translational: 'path/to/translational_features_shuffled.csv'
#   Rotational   : 'path/to/rotational_features_shuffled.csv'

CSV_PATH = 'translational_features_shuffled.csv'   # ← CHANGE THIS PATH

# Label for output filenames and titles (change to 'rotational' if needed)
CONDITION_LABEL = 'translational'   # ← CHANGE TO 'rotational' IF NEEDED

# Output folder — results are saved here
OUT_DIR = os.path.join(os.getcwd(), f'results_{CONDITION_LABEL}')
os.makedirs(OUT_DIR, exist_ok=True)

# Random Forest hyperparameters (same as used in the paper)
SEED      = 42
RF_PARAMS = dict(
    n_estimators    = 200,
    max_depth       = 20,
    min_samples_leaf= 4,
    n_jobs          = -1,
    random_state    = SEED
)

# Number of test samples used for SHAP computation (subset for speed)
SHAP_SAMPLE_SIZE = 2000

# =============================================================================
#  POOL B — 15 SPECTRAL FEATURES (feature names as in CSV)
# =============================================================================

POOL_B = [
    'SP_dc_component',      # f1  — DC component (near-field reflectivity)
    'SP_spec_entropy',      # f2  — Spectral entropy
    'SP_spec_flatness',     # f3  — Spectral flatness (Wiener entropy)
    'SP_spec_centroid',     # f4  — Spectral centroid
    'SP_spec_spread',       # f5  — Spectral spread
    'SP_spec_skew',         # f6  — Spectral skewness
    'SP_spec_kurt',         # f7  — Spectral kurtosis
    'SP_band_ratio_L',      # f8  — Low-band energy ratio
    'SP_band_ratio_M',      # f9  — Mid-band energy ratio
    'SP_band_ratio_H',      # f10 — High-band energy ratio
    'SP_peak_freq',         # f11 — Peak frequency bin
    'SP_spec_rolloff',      # f12 — Spectral rolloff (85% energy)
    'SP_spec_flux',         # f13 — Spectral flux (inter-chirp change)
    'SP_mean_abs_deriv',    # f14 — Mean absolute derivative
    'SP_zero_cross_rate',   # f15 — Zero-crossing rate
]
N_FEATS = len(POOL_B)

# =============================================================================
#  HELPER — normalise SHAP values (handles different shap output formats)
# =============================================================================

def normalise_shap(shap_values, n_classes, n_samples, n_features):
    """
    Converts shap_values to a list of 2D arrays [n_samples x n_features],
    one per class. Handles both old (list) and new (3D array) SHAP formats.
    """
    if isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            # Shape could be (n_samples, n_features, n_classes)
            if shap_values.shape == (n_samples, n_features, n_classes):
                return [shap_values[:, :, c] for c in range(n_classes)]
            # or (n_classes, n_samples, n_features)
            if shap_values.shape == (n_classes, n_samples, n_features):
                return [shap_values[c] for c in range(n_classes)]
        if shap_values.ndim == 2:
            return [shap_values, -shap_values]
    if isinstance(shap_values, list):
        return shap_values
    raise ValueError(f"Cannot parse SHAP output with shape {np.array(shap_values).shape}")

# =============================================================================
#  STEP 1 — LOAD DATA
# =============================================================================

print("=" * 65)
print(f"  RF + SHAP Analysis  |  {CONDITION_LABEL.capitalize()} Condition")
print("=" * 65)

if not os.path.isfile(CSV_PATH):
    raise FileNotFoundError(
        f"CSV not found: {CSV_PATH}\n"
        f"Update CSV_PATH at the top of this script."
    )

print(f"\nLoading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  Shape   : {df.shape}")

# Verify all Pool B features are present
missing = [f for f in POOL_B if f not in df.columns]
if missing:
    raise ValueError(f"Missing Pool B features in CSV: {missing}")

# Build feature matrix and encoded labels
X  = np.nan_to_num(
        df[POOL_B].values.astype(np.float32),
        nan=0., posinf=0., neginf=0.
     )
le          = LabelEncoder()
y           = le.fit_transform(df['label'].values)
class_names = list(le.classes_)
n_classes   = len(class_names)

print(f"  Features: {N_FEATS}")
print(f"  Samples : {len(y)}")
print(f"  Classes : {class_names}")

# =============================================================================
#  STEP 2 — TRAIN / TEST SPLIT (80 / 20, stratified)
# =============================================================================

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y,
    test_size   = 0.20,
    random_state= SEED,
    stratify    = y
)
print(f"\n  Train : {len(y_tr)}  |  Test : {len(y_te)}")

# =============================================================================
#  STEP 3 — TRAIN RANDOM FOREST ON ALL 15 POOL B FEATURES
# =============================================================================

print(f"\nTraining Random Forest ...")
t0  = time.time()
clf = RandomForestClassifier(**RF_PARAMS)
clf.fit(X_tr, y_tr)
print(f"  Training time : {time.time() - t0:.1f} s")

y_pred_full = clf.predict(X_te)
acc_full    = accuracy_score(y_te, y_pred_full)
f1_full     = f1_score(y_te, y_pred_full, average='macro')

print(f"  Test accuracy : {acc_full:.4f}  ({acc_full*100:.2f}%)")
print(f"  Macro F1      : {f1_full:.4f}")
print(f"\n  Classification Report:")
print(classification_report(y_te, y_pred_full, target_names=class_names))

# =============================================================================
#  STEP 4 — COMPUTE SHAP IMPORTANCE
# =============================================================================

print(f"\nComputing SHAP values (sample size = {SHAP_SAMPLE_SIZE}) ...")
rng      = np.random.default_rng(SEED)
shap_idx = rng.choice(len(X_te),
                       size=min(SHAP_SAMPLE_SIZE, len(X_te)),
                       replace=False)
X_shap = X_te[shap_idx]

explainer  = shap.TreeExplainer(clf)
shap_raw   = explainer.shap_values(X_shap)
shap_list  = normalise_shap(shap_raw, n_classes,
                             len(X_shap), N_FEATS)

# Global importance = mean |SHAP| across all classes and samples
# Shape of each shap_list[c]: (n_samples, n_features)
global_imp = np.mean(
    np.stack([np.abs(sv).mean(axis=0) for sv in shap_list]),
    axis=0
)   # shape: (n_features,)

# Per-class mean |SHAP|: shape (n_classes, n_features)
per_class_imp = np.stack(
    [np.abs(sv).mean(axis=0) for sv in shap_list]
)

# Sort features by global importance (descending = rank 1 is most important)
shap_rank_order = np.argsort(global_imp)[::-1]
ranked_features = [POOL_B[i] for i in shap_rank_order]
ranked_shap     = global_imp[shap_rank_order]

print(f"\n  SHAP ranking ({CONDITION_LABEL}):")
print(f"  {'Rank':<5} {'Feature':<28} {'Mean |SHAP|':>12}")
print(f"  {'─'*4} {'─'*27} {'─'*12}")
for rank, (feat, imp) in enumerate(
        zip(ranked_features, ranked_shap), start=1):
    print(f"  {rank:<5} {feat:<28} {imp:>12.5f}")

# =============================================================================
#  STEP 5 — CUMULATIVE ACCURACY IN SHAP RANK ORDER
#           (add features top-1 → top-15)
# =============================================================================

print(f"\nCumulative accuracy (top-1 → top-15 in SHAP rank order) ...")
cum_accuracies = []
cum_f1         = []

for step in range(1, N_FEATS + 1):
    top_k_idx = shap_rank_order[:step]    # indices of top-k features
    clf_k     = RandomForestClassifier(**RF_PARAMS)
    clf_k.fit(X_tr[:, top_k_idx], y_tr)
    y_pred_k  = clf_k.predict(X_te[:, top_k_idx])
    acc_k     = accuracy_score(y_te, y_pred_k)
    f1_k      = f1_score(y_te, y_pred_k, average='macro')
    cum_accuracies.append(acc_k)
    cum_f1.append(f1_k)
    print(f"  Step {step:>2}  +{ranked_features[step-1]:<28}  "
          f"acc={acc_k:.4f}  f1={f1_k:.4f}")

# =============================================================================
#  STEP 6 — SAVE RESULTS (CSV)
# =============================================================================

# SHAP ranking CSV
shap_df = pd.DataFrame({
    'rank'      : range(1, N_FEATS + 1),
    'feature'   : ranked_features,
    'shap_importance': ranked_shap.round(5),
    'condition' : CONDITION_LABEL,
})
shap_df.to_csv(os.path.join(OUT_DIR, 'shap_ranking.csv'), index=False)
print(f"\n  Saved: shap_ranking.csv")

# Cumulative accuracy CSV
cum_df = pd.DataFrame({
    'step'      : range(1, N_FEATS + 1),
    'feature'   : ranked_features,
    'accuracy'  : cum_accuracies,
    'macro_f1'  : cum_f1,
    'condition' : CONDITION_LABEL,
})
cum_df.to_csv(os.path.join(OUT_DIR, 'cumulative_accuracy.csv'), index=False)
print(f"  Saved: cumulative_accuracy.csv")

# =============================================================================
#  STEP 7 — PLOTS
# =============================================================================

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 9,
    'ytick.labelsize'  : 10,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

# ── Plot 1: SHAP importance bar chart ────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
colors  = plt.cm.viridis(np.linspace(0.85, 0.2, N_FEATS))
ax.barh(
    [f.replace('SP_', '') for f in ranked_features[::-1]],
    ranked_shap[::-1],
    color=colors, edgecolor='white', height=0.7
)
for i, v in enumerate(ranked_shap[::-1]):
    ax.text(v + 0.0003, i, f'{v:.5f}',
            va='center', fontsize=8, color='#222222')
ax.set_xlabel('Mean |SHAP| value (global importance)', fontsize=11)
ax.set_title(
    f'Pool B SHAP Feature Importance — {CONDITION_LABEL.capitalize()} Condition',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'shap_bar.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: shap_bar.png")

# ── Plot 2: Per-class SHAP heatmap ────────────────────────────
fig, ax = plt.subplots(figsize=(13, 5))
heatmap_df = pd.DataFrame(
    per_class_imp,
    index  = class_names,
    columns= POOL_B
)
# Reorder columns by global SHAP rank
heatmap_df = heatmap_df[[ranked_features[i] for i in range(N_FEATS)]]
heatmap_df.columns = [c.replace('SP_', '') for c in heatmap_df.columns]
sns.heatmap(
    heatmap_df,
    ax=ax, cmap='YlOrRd', annot=True, fmt='.4f',
    linewidths=0.5, cbar_kws={'label': 'Mean |SHAP|'}
)
ax.set_title(
    f'Per-Class SHAP Importance — {CONDITION_LABEL.capitalize()} Condition\n'
    f'(columns ordered by global SHAP rank)',
    fontsize=12, fontweight='bold'
)
ax.set_xlabel('Pool B Feature (SHAP rank order)')
ax.set_ylabel('Fabric Class')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'per_class_heatmap.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: per_class_heatmap.png")

# ── Plot 3: Confusion matrix ──────────────────────────────────
cm      = confusion_matrix(y_te, y_pred_full)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax_i, (data, title) in enumerate([
        (cm,      f'Counts  (acc = {acc_full:.4f})'),
        (cm_norm, f'Normalised  (acc = {acc_full:.4f})')]):
    a2 = axes[ax_i]
    im = a2.imshow(data, cmap='Blues',
                   vmin=0, vmax=(1 if ax_i == 1 else None))
    a2.set_xticks(range(n_classes))
    a2.set_xticklabels(class_names, rotation=35, ha='right', fontsize=9)
    a2.set_yticks(range(n_classes))
    a2.set_yticklabels(class_names, fontsize=9)
    plt.colorbar(im, ax=a2)
    fmt = '.2f' if ax_i == 1 else 'd'
    for ci in range(n_classes):
        for cj in range(n_classes):
            val = data[ci, cj]
            a2.text(cj, ci, format(val, fmt),
                    ha='center', va='center', fontsize=9,
                    color='white'
                    if val > (0.5 if ax_i == 1 else cm.max() * 0.5)
                    else 'black')
    a2.set_xlabel('Predicted'); a2.set_ylabel('True')
    a2.set_title(title)
fig.suptitle(
    f'Confusion Matrix — {CONDITION_LABEL.capitalize()} Condition '
    f'| Pool B (all 15 features)',
    fontsize=12, fontweight='bold'
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'confusion_matrix.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: confusion_matrix.png")

# ── Plot 4: Cumulative accuracy curve ────────────────────────
steps  = np.arange(1, N_FEATS + 1)
labels = [f.replace('SP_', '') for f in ranked_features]
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(steps, cum_accuracies,
        color='#4C72B0', lw=2.5, marker='o', ms=8, zorder=4,
        label=f'Test accuracy  [final = {cum_accuracies[-1]:.4f}]')
ax.fill_between(steps, 0, cum_accuracies, alpha=0.08, color='#4C72B0')
for s, v in zip(steps, cum_accuracies):
    ax.text(s, v + 0.012, f'{v:.4f}',
            ha='center', va='bottom', fontsize=7.5,
            color='#1a3a6e', fontweight='bold')
ax.set_xticks(steps)
ax.set_xticklabels(
    [f'{s}\n{lbl}' for s, lbl in zip(steps, labels)],
    fontsize=8, rotation=30, ha='right'
)
ax.set_xlabel('Pool B features added in SHAP rank order', fontsize=11)
ax.set_ylabel('Classification Accuracy', fontsize=11)
ax.set_xlim(0.5, N_FEATS + 0.5)
ax.set_ylim(max(0.3, min(cum_accuracies) - 0.06),
            min(1.02, max(cum_accuracies) + 0.04))
ax.set_title(
    f'Cumulative Accuracy — {CONDITION_LABEL.capitalize()} Condition\n'
    f'Pool B | Features added in SHAP rank order',
    fontsize=12, fontweight='bold', pad=12
)
ax.legend(fontsize=10, loc='lower right', framealpha=0.93)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'cumulative_accuracy.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: cumulative_accuracy.png")

# =============================================================================
#  SUMMARY
# =============================================================================

print("\n" + "=" * 65)
print(f"  SUMMARY — {CONDITION_LABEL.capitalize()} Condition")
print("=" * 65)
print(f"  CSV             : {CSV_PATH}")
print(f"  Samples         : {len(y)}")
print(f"  Full model acc  : {acc_full:.4f}  ({acc_full*100:.2f}%)")
print(f"  Full model F1   : {f1_full:.4f}")
print(f"\n  Top-5  features : acc = {cum_accuracies[4]:.4f}")
print(f"  Top-10 features : acc = {cum_accuracies[9]:.4f}")
print(f"  All 15 features : acc = {cum_accuracies[14]:.4f}")
print(f"\n  SHAP rank 1     : {ranked_features[0]}  ({ranked_shap[0]:.5f})")
print(f"  SHAP rank 2     : {ranked_features[1]}  ({ranked_shap[1]:.5f})")
print(f"  SHAP rank 3     : {ranked_features[2]}  ({ranked_shap[2]:.5f})")
print(f"\n  Output folder   : {OUT_DIR}")
print("  DONE.")
