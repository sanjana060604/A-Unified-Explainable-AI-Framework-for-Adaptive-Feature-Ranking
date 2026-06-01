"""
=============================================================================
Harmonic Rank Fusion — Fabric Material Classification
Combining Translational + Rotational SHAP Rankings
=============================================================================

PURPOSE
-------
This script fuses the SHAP-based feature rankings from two experimental
conditions (translational and rotational) into a single, condition-invariant
feature ranking using the harmonic mean of rank scores.

The harmonic mean is used because it penalises features that are highly
ranked in only ONE condition — a feature must be important in BOTH conditions
to receive a high combined score.

WORKFLOW
--------
  Step 1:  Read SHAP rankings from both conditions
             translational: results_translational/shap_ranking.csv
             rotational   : results_rotational/shap_ranking.csv

  Step 2:  Extract rank position of each feature in each condition
             trans_rank[f]  = rank of feature f in translational SHAP ranking
             rotat_rank[f]  = rank of feature f in rotational SHAP ranking

  Step 3:  Convert rank → score  (rank 1 = best = score 1.0)
             score = (N - rank + 1) / N      where N = 15

  Step 4:  Compute harmonic mean score
             S_harm[f] = 2 × trans_score[f] × rotat_score[f]
                           / (trans_score[f] + rotat_score[f])

  Step 5:  Sort by S_harm descending → final condition-invariant ranking

  Step 6:  Re-accumulate cumulative accuracy in the harmonic rank order
           using the average accuracy gain from both conditions

  Step 7:  Plot feature vs accuracy (single curve = harmonic ordering)
           and compare with condition-specific curves

INPUTS  (produced by shap_rf_analysis.py)
------
  results_translational/shap_ranking.csv        — SHAP ranks, translational
  results_translational/cumulative_accuracy.csv — step-wise accuracy, translational
  results_rotational/shap_ranking.csv           — SHAP ranks, rotational
  results_rotational/cumulative_accuracy.csv    — step-wise accuracy, rotational

OUTPUTS  (saved to results_harmonic_fusion/)
-------
  harmonic_ranking.csv          — final ranked list with S_harm scores
  harmonic_accuracy_curve.png   — single curve: features vs accuracy (harmonic order)
  comparison_curves.png         — 3 curves: translational, harmonic, rotational

=============================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# =============================================================================
#  CONFIGURATION — paths to results from shap_rf_analysis.py
# =============================================================================

# Folder containing results from translational run
TRANS_SHAP_CSV = os.path.join('results_translational', 'shap_ranking.csv')
TRANS_ACC_CSV  = os.path.join('results_translational', 'cumulative_accuracy.csv')

# Folder containing results from rotational run
ROTAT_SHAP_CSV = os.path.join('results_rotational', 'shap_ranking.csv')
ROTAT_ACC_CSV  = os.path.join('results_rotational', 'cumulative_accuracy.csv')

# Output folder for harmonic fusion results
OUT_DIR = 'results_harmonic_fusion'
os.makedirs(OUT_DIR, exist_ok=True)

# Number of features
N = 15

# =============================================================================
#  STEP 1 — LOAD SHAP RANKINGS FROM BOTH CONDITIONS
# =============================================================================

print("=" * 70)
print("  Harmonic Rank Fusion — Translational + Rotational")
print("=" * 70)

for path in [TRANS_SHAP_CSV, TRANS_ACC_CSV,
             ROTAT_SHAP_CSV, ROTAT_ACC_CSV]:
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"File not found: {path}\n"
            f"Run shap_rf_analysis.py for both conditions first."
        )

df_trans_shap = pd.read_csv(TRANS_SHAP_CSV)
df_rotat_shap = pd.read_csv(ROTAT_SHAP_CSV)
df_trans_acc  = pd.read_csv(TRANS_ACC_CSV)
df_rotat_acc  = pd.read_csv(ROTAT_ACC_CSV)

# Build rank dictionaries: {feature_name: rank_position}
# rank column is 1-indexed (1 = most important)
trans_rank = dict(zip(df_trans_shap['feature'], df_trans_shap['rank']))
rotat_rank = dict(zip(df_rotat_shap['feature'], df_rotat_shap['rank']))

# All 15 Pool B features
all_features = list(df_trans_shap['feature'])

print(f"\nStep 1 — SHAP rank positions from both conditions:")
print(f"\n  {'Feature':<28} {'Trans rank':>11} {'Rotat rank':>11}")
print(f"  {'─'*27} {'─'*11} {'─'*11}")
for f in all_features:
    print(f"  {f:<28} {trans_rank[f]:>11} {rotat_rank[f]:>11}")

# =============================================================================
#  STEP 2 — CONVERT RANK → SCORE
#   rank 1 (most important) → score 1.0
#   rank 15 (least important) → score 1/15 ≈ 0.067
# =============================================================================

def rank_to_score(rank, n=15):
    """
    Converts rank position to a normalised score.
    rank 1 → 1.0 (best), rank 15 → 0.067 (worst).
    This ensures features ranked #1 receive full weight.
    """
    return (n - rank + 1) / n

trans_score = {f: rank_to_score(trans_rank[f]) for f in all_features}
rotat_score = {f: rank_to_score(rotat_rank[f]) for f in all_features}

print(f"\nStep 2 — Rank converted to score (rank 1 = 1.0, rank 15 = 0.067):")
print(f"\n  {'Feature':<28} {'Trans score':>12} {'Rotat score':>12}")
print(f"  {'─'*27} {'─'*12} {'─'*12}")
for f in sorted(all_features, key=lambda x: -trans_score[x]):
    print(f"  {f:<28} {trans_score[f]:>12.5f} {rotat_score[f]:>12.5f}")

# =============================================================================
#  STEP 3 — HARMONIC MEAN SCORE
#   S_harm = 2ab / (a + b)
#   A feature ranked highly in only ONE condition is penalised:
#     e.g., trans_score=1.0, rotat_score=0.067 → S_harm ≈ 0.12
#   A feature ranked highly in BOTH conditions is rewarded:
#     e.g., trans_score=1.0, rotat_score=1.0   → S_harm = 1.0
# =============================================================================

S_HARM = {
    f: 2 * trans_score[f] * rotat_score[f]
         / (trans_score[f] + rotat_score[f] + 1e-12)
    for f in all_features
}

print(f"\nStep 3 — Harmonic mean score S_harm = 2ab/(a+b):")
print(f"\n  {'Feature':<28} {'Trans sc':>9} {'Rotat sc':>9} {'S_harm':>9}")
print(f"  {'─'*27} {'─'*9} {'─'*9} {'─'*9}")
for f in sorted(all_features, key=lambda x: -S_HARM[x]):
    print(f"  {f:<28} {trans_score[f]:>9.5f} "
          f"{rotat_score[f]:>9.5f} {S_HARM[f]:>9.5f}")

# =============================================================================
#  STEP 4 — FINAL HARMONIC RANKING (sorted by S_harm descending)
# =============================================================================

HARM_ORDER = sorted(all_features, key=lambda x: -S_HARM[x])

print(f"\nStep 4 — Final harmonic ranking:")
print(f"\n  {'Rank':<5} {'Feature':<28} {'S_harm':>9} "
      f"{'Trans R':>8} {'Rotat R':>8}")
print(f"  {'─'*4} {'─'*27} {'─'*9} {'─'*8} {'─'*8}")
for i, f in enumerate(HARM_ORDER):
    marker = '  ← TOP 5'  if i < 5  else \
             '  ← TOP 10' if i < 10 else ''
    print(f"  {i+1:<5} {f:<28} {S_HARM[f]:>9.5f} "
          f"{trans_rank[f]:>8} {rotat_rank[f]:>8}{marker}")

# =============================================================================
#  STEP 5 — CUMULATIVE ACCURACY IN HARMONIC ORDER
#   We derive the marginal accuracy gain of each feature from each condition
#   then average both to get the harmonic ordering's expected accuracy.
# =============================================================================

# Build per-feature marginal accuracy gain from each condition
# cumulative_accuracy.csv has columns: step, feature, accuracy, macro_f1, condition
# The feature column tells us which feature was added at that step

def build_gain_dict(acc_df):
    """
    Converts a cumulative accuracy DataFrame into a per-feature gain dict.
    gain[feature] = accuracy[step_k] - accuracy[step_k-1]
    """
    acc_vals = acc_df['accuracy'].values
    features = acc_df['feature'].values
    gains    = [acc_vals[0]] + [acc_vals[i] - acc_vals[i-1]
                                 for i in range(1, len(acc_vals))]
    return {f: g for f, g in zip(features, gains)}

trans_gain = build_gain_dict(df_trans_acc)
rotat_gain = build_gain_dict(df_rotat_acc)

# Average gain per feature across both conditions
avg_gain = {
    f: (trans_gain.get(f, 0) + rotat_gain.get(f, 0)) / 2.0
    for f in all_features
}

# Cumulative accuracy in harmonic order
harm_cum_acc  = np.cumsum([avg_gain[f] for f in HARM_ORDER])

# Also get cumulative accuracies in each condition's own order
# (for the comparison plot — each curve uses its condition's order)
trans_order = list(df_trans_acc['feature'])
rotat_order = list(df_rotat_acc['feature'])

trans_cum_acc = df_trans_acc['accuracy'].values   # already cumulative
rotat_cum_acc = df_rotat_acc['accuracy'].values

print(f"\nStep 5 — Cumulative accuracy in harmonic order:")
print(f"\n  {'Rank':<5} {'Feature':<28} {'Harm acc':>10} "
      f"{'Trans acc':>10} {'Rotat acc':>10}")
print(f"  {'─'*4} {'─'*27} {'─'*10} {'─'*10} {'─'*10}")
for i, f in enumerate(HARM_ORDER):
    # Find trans and rotat accuracy at the step where this feature was added
    t_step = trans_order.index(f) if f in trans_order else -1
    r_step = rotat_order.index(f) if f in rotat_order else -1
    t_acc  = trans_cum_acc[t_step] if t_step >= 0 else float('nan')
    r_acc  = rotat_cum_acc[r_step] if r_step >= 0 else float('nan')
    print(f"  {i+1:<5} {f:<28} {harm_cum_acc[i]:>10.4f} "
          f"{t_acc:>10.4f} {r_acc:>10.4f}")

# =============================================================================
#  STEP 6 — SAVE RESULTS
# =============================================================================

# Save final harmonic ranking CSV
harm_df = pd.DataFrame([{
    'final_rank'       : i + 1,
    'feature'          : f,
    'S_harm'           : round(S_HARM[f],       5),
    'trans_rank'       : trans_rank[f],
    'rotat_rank'       : rotat_rank[f],
    'trans_rank_score' : round(trans_score[f],  5),
    'rotat_rank_score' : round(rotat_score[f],  5),
    'trans_gain'       : round(trans_gain.get(f, 0), 5),
    'rotat_gain'       : round(rotat_gain.get(f, 0), 5),
    'avg_gain'         : round(avg_gain[f],      5),
    'cum_acc_harmonic' : round(harm_cum_acc[i],  4),
} for i, f in enumerate(HARM_ORDER)])

harm_df.to_csv(os.path.join(OUT_DIR, 'harmonic_ranking.csv'), index=False)
print(f"\n  Saved: harmonic_ranking.csv")

# =============================================================================
#  STEP 7 — PLOTS
# =============================================================================

plt.rcParams.update({
    'figure.dpi'       : 150,
    'font.family'      : 'DejaVu Sans',
    'font.size'        : 11,
    'axes.titlesize'   : 12,
    'axes.labelsize'   : 11,
    'xtick.labelsize'  : 8.5,
    'ytick.labelsize'  : 10,
    'figure.facecolor' : 'white',
    'axes.facecolor'   : 'white',
    'axes.grid'        : True,
    'grid.alpha'       : 0.3,
    'grid.linestyle'   : '--',
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

steps       = np.arange(1, N + 1)
harm_labels = [f.replace('SP_', '') for f in HARM_ORDER]

# ── Plot 1: Single harmonic curve ────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(steps, harm_cum_acc,
        color='#4C72B0', lw=2.8, marker='o', ms=9, zorder=4,
        label=f'Harmonic ranking  [final = {harm_cum_acc[-1]:.4f}]')
ax.fill_between(steps, 0, harm_cum_acc, alpha=0.08, color='#4C72B0')
for s, v in zip(steps, harm_cum_acc):
    ax.text(s, v + 0.012, f'{v:.4f}',
            ha='center', va='bottom', fontsize=7.5,
            color='#1a3a6e', fontweight='bold')
ax.set_xticks(steps)
ax.set_xticklabels(
    [f'{s}\n{lbl}' for s, lbl in zip(steps, harm_labels)],
    fontsize=8, rotation=30, ha='right')
ax.set_xlabel('Pool B features — added in final harmonic rank order',
              fontsize=11, labelpad=8)
ax.set_ylabel('Cumulative Accuracy\n(avg across translational + rotational)',
              fontsize=10)
ax.set_xlim(0.5, N + 0.5)
ax.set_ylim(max(0.3, harm_cum_acc.min() - 0.06),
            min(1.02, harm_cum_acc.max() + 0.04))
ax.set_title(
    'Pool B — Final Harmonic Combined Ranking\n'
    'S_harm = 2 × trans_score × rotat_score / (trans + rotat)\n'
    'Single curve: avg accuracy gain across translational + rotational conditions',
    fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=10, loc='lower right', framealpha=0.93)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'harmonic_accuracy_curve.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: harmonic_accuracy_curve.png")

# ── Plot 2: 3-curve comparison on harmonic x-axis ────────────
# Re-accumulate each condition in harmonic order for fair comparison
trans_gain_harm = np.cumsum([trans_gain.get(f, 0) for f in HARM_ORDER])
rotat_gain_harm = np.cumsum([rotat_gain.get(f, 0) for f in HARM_ORDER])

fig, ax = plt.subplots(figsize=(15, 7))

# Translational (orange, lower)
ax.plot(steps, trans_gain_harm,
        color='#DD8452', lw=2.5, marker='o', ms=8,
        ls='--', zorder=4,
        label=f'Translational  [final = {trans_gain_harm[-1]:.4f}]')

# Harmonic (blue, middle)
ax.plot(steps, harm_cum_acc,
        color='#4C72B0', lw=2.5, marker='D', ms=7,
        ls='-.', zorder=4,
        label=f'Harmonic combined  [final = {harm_cum_acc[-1]:.4f}]')

# Rotational (green, upper or variable)
ax.plot(steps, rotat_gain_harm,
        color='#55A868', lw=2.5, marker='s', ms=8,
        zorder=4,
        label=f'Rotational  [final = {rotat_gain_harm[-1]:.4f}]')

# Gap shading between translational and rotational
ax.fill_between(steps,
                np.minimum(trans_gain_harm, rotat_gain_harm),
                np.maximum(trans_gain_harm, rotat_gain_harm),
                alpha=0.07, color='gray',
                label='Gap (translational ↔ rotational)')

ax.set_xticks(steps)
ax.set_xticklabels(
    [f'{s}\n{lbl}' for s, lbl in zip(steps, harm_labels)],
    fontsize=8, rotation=30, ha='right')
ax.set_xlabel('Pool B features — harmonic rank order (common x-axis)',
              fontsize=11, labelpad=8)
ax.set_ylabel('Cumulative Accuracy', fontsize=11)
all_vals = np.concatenate([trans_gain_harm, rotat_gain_harm, harm_cum_acc])
ax.set_xlim(0.5, N + 0.5)
ax.set_ylim(max(0.3, all_vals.min() - 0.06),
            min(1.02, all_vals.max() + 0.04))
ax.set_title(
    'Three-Curve Comparison — Common Harmonic Feature Order\n'
    'Orange = Translational  |  Blue = Harmonic  |  Green = Rotational\n'
    'Harmonic curve lies between both condition-specific curves',
    fontsize=12, fontweight='bold', pad=12)
ax.legend(fontsize=9.5, loc='lower right', framealpha=0.93)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'comparison_curves.png'),
            bbox_inches='tight', dpi=150)
plt.close()
print(f"  Saved: comparison_curves.png")

# =============================================================================
#  FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("  HARMONIC RANK FUSION — FINAL SUMMARY")
print("=" * 70)
print(f"\n  Conditions fused  : translational + rotational")
print(f"\n  {'Rank':<5} {'Feature':<28} {'S_harm':>9} "
      f"{'Trans R':>8} {'Rotat R':>8}")
print(f"  {'─'*4} {'─'*27} {'─'*9} {'─'*8} {'─'*8}")
for i, f in enumerate(HARM_ORDER):
    marker = '  ← TOP 5'  if i < 5  else \
             '  ← TOP 10' if i < 10 else ''
    print(f"  {i+1:<5} {f:<28} {S_HARM[f]:>9.5f} "
          f"{trans_rank[f]:>8} {rotat_rank[f]:>8}{marker}")

print(f"\n  Harmonic rank 1  : {HARM_ORDER[0]}")
print(f"  Harmonic rank 2  : {HARM_ORDER[1]}")
print(f"  Harmonic rank 3  : {HARM_ORDER[2]}")
print(f"\n  Output folder    : {OUT_DIR}")
print("  DONE.")
