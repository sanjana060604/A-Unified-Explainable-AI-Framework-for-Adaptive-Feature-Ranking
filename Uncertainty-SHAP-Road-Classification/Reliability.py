"""
=============================================================
  Model 3 v2 — Balanced Reliability-Weighted SHAP (RW-SHAP)
  Dataset : AWR2243 mmWave radar — Road Surface Classification
  Author  : Sanjana Arigela, University of Agder, 2026

  PURPOSE
  -------
  Produce a single RELIABILITY-WEIGHTED feature ranking that
  combines the strengths of both SHAP importance (how much
  does a feature influence predictions?) and epistemic UQ
  change (how much does a feature stabilise predictions?).

  The core insight: a feature that is both discriminative
  (high SHAP) AND epistemically stable (high UQ_change) is
  more trustworthy than a feature that excels on only one axis.

  SCORE FORMULA
  -------------
  For fold k and feature i:

      S_i^(k)(α) = φ̃_i^(k) ^ α  ·  ũ_i^(k) ^ (1 - α)

  where:
    φ̃_i^(k)  = min-max normalised |SHAP_i| for fold k  ∈ [ε, 1]
    ũ_i^(k)   = min-max normalised UQ_change_i for fold k ∈ [ε, 1]
    α ∈ (0,1) = balance parameter:
                  α = 1 → pure SHAP  (UQ ignored)
                  α = 0 → pure UQ    (SHAP ignored)
                  α = 0.5 → equal geometric weight

  WHY GEOMETRIC MEAN?
  -------------------
  Unlike a linear combination (α·SHAP + (1-α)·UQ), the
  geometric mean PENALISES features that score poorly on
  either signal:
    If UQ ≈ 0 → S ≈ 0 regardless of SHAP
    If SHAP ≈ 0 → S ≈ 0 regardless of UQ
  A feature must be strong on BOTH axes to rank highly.

  HOW ALPHA IS CHOSEN — Leave-One-Out Spearman Calibration
  ---------------------------------------------------------
  Rather than fixing α manually, this script finds the best
  α for each fold independently, using a Leave-One-Out (LOO)
  criterion:

  For fold k, the best α satisfies:
      α*(k) = argmax_α  ρ_s( S^(k)(α),  S̄^(-k)(α) )

  where:
    S^(k)(α)   = the 15-dim score vector for fold k at this α
    S̄^(-k)(α) = the mean score of the OTHER 9 folds at same α
    ρ_s        = Spearman rank correlation

  This forces fold k's ranking to agree with the consensus of
  the other 9 folds — creating genuine calibration pressure
  rather than simply tuning α to maximise fold k's own score
  (which would be circular).

  ANTI-COLLAPSE GUARD
  -------------------
  If ≥80% of folds select the boundary value of α or γ, the
  search space is too narrow. The script automatically expands
  the grid and re-tunes.

  WHAT THIS SCRIPT READS
  ----------------------
  contribution_10fold/run_XX/run_XX_contribution_scores.csv
    ↑ produced by run_contribution_ranking.py
    ↑ contains: shap_value, uq_change, contribution per feature per fold

  WHAT THIS SCRIPT WRITES
  -----------------------
  contribution_reliability_weighted_v2/
    run_01/ ... run_10/     per-fold results
      run_XX_reliability_scores_v2.csv  S_i scores + ranks
      run_XX_reliability_bar_v2.png     feature ranking bar chart
      run_XX_alpha_balance_v2.png       α position on SHAP–UQ spectrum
      run_XX_four_method_bar_v2.png     comparison: v2 vs SHAP vs UQ vs old
    summary/
      reliability_summary_v2.csv        10-fold averaged rankings
      reliability_report_v2.txt         full text report
      four_method_ranking_bar_v2.png
      reliability_rank_stability_heatmap_v2.png
      alpha_sensitivity_v2.png          rank sensitivity to α
      reliability_rank_delta_v2.png
      reliability_summary_table_v2.png
      four_method_score_heatmap_v2.png
      alpha_gamma_rho_per_fold.png

  HOW TO RUN
  ----------
    python run_reliability_weighted_v2.py
    (Requires contribution_10fold/ from run_contribution_ranking.py)

  DEPENDENCIES
  ------------
    pip install scipy pandas numpy matplotlib seaborn
=============================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
from scipy.special import expit          # sigmoid function
from scipy.stats import spearmanr        # Spearman rank correlation
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════

# Sigmoid gate: set True to multiply score by sigmoid(γ·ũ)
# Default: False — the pure geometric mean without extra sharpening
USE_SIGMOID = False

# Alpha search grid: 19 values from 0.05 to 0.95
# Excludes 0 and 1 to avoid pure UQ or pure SHAP degeneracy
ALPHA_GRID = np.linspace(0.05, 0.95, 19)

# Gamma search grid (only matters if USE_SIGMOID=True)
# Log-spaced from 0.01 to 1000 to cover both flat and sharp sigmoid gates
GAMMA_GRID = np.logspace(-2, 3, 50)

# Anti-collapse detection thresholds
BOUNDARY_TOL  = 1e-6   # how close to grid edge counts as "hit boundary"
COLLAPSE_FRAC = 0.8    # if >=80% of folds hit boundary, expand grid

BASE_DIR  = os.getcwd()
FULL_CSV  = os.path.join(BASE_DIR, 'obj1_poolB_features_shuffled.csv')
SRC_DIR   = os.path.join(BASE_DIR, 'contribution_10fold')
ROOT_OUT  = os.path.join(BASE_DIR, 'contribution_reliability_weighted_v2')
SUM_DIR   = os.path.join(ROOT_OUT, 'summary')
os.makedirs(SUM_DIR, exist_ok=True)

K_FOLDS     = 10
POOL_PREFIX = 'SP_'

COL_SHAP = '#3498DB'
COL_UQ   = '#E67E22'
COL_REL  = '#8E44AD'
COL_OLD  = '#27AE60'

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
    """Save and close the current matplotlib figure."""
    plt.savefig(os.path.join(directory, name), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"    Saved: {name}")


def feat_color(f):
    """Return the assigned colour for a given feature name."""
    return FEAT_COLORS.get(f, '#95A5A6')


def make_rank(values):
    """Rank features by descending value: rank 1 = highest score."""
    return pd.Series(values).rank(ascending=False).astype(int).values


def zscore_safe(v):
    """Z-score normalise; return zeros if std is near zero."""
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def minmax_safe(v):
    """
    Min-max scale v to [0, 1].
    If all values are equal (zero variance), return 0.5 for all.
    """
    lo, hi = v.min(), v.max()
    if hi - lo < 1e-12:
        return np.full_like(v, 0.5)
    return (v - lo) / (hi - lo)


def compute_score(shap_vals, uq_vals, alpha, gamma,
                  use_sigmoid=USE_SIGMOID):
    """
    Compute the balanced reliability score for one fold.

    Steps:
      1. Take abs(SHAP) for importance magnitude
      2. Clip UQ to >= 0 (negative = destabilising, gets no credit)
      3. Min-max normalise both to [0, 1]
      4. Clip to [ε, 1] to avoid 0^0 = 1 artefact
      5. Compute geometric mean: S = φ̃^α · ũ^(1-α)
      6. Optionally multiply by sigmoid gate (if USE_SIGMOID=True)

    Parameters
    ----------
    shap_vals   : array [F]  raw SHAP values for this fold
    uq_vals     : array [F]  UQ_change values for this fold
    alpha       : float      SHAP weight in [0, 1]
    gamma       : float      sigmoid sharpness (only if use_sigmoid=True)
    use_sigmoid : bool       whether to apply sigmoid gate

    Returns
    -------
    score : array [F]  reliability score per feature
    """
    phi_pos  = np.abs(shap_vals)          # SHAP magnitude (always positive)
    uq_pos   = np.clip(uq_vals, 0, None)  # drop negative UQ contributions

    phi_norm = minmax_safe(phi_pos)
    u_norm   = minmax_safe(uq_pos)

    # Prevent 0^0 = 1 numerical artefact
    phi_norm = np.clip(phi_norm, 1e-8, 1.0)
    u_norm   = np.clip(u_norm,   1e-8, 1.0)

    # Geometric weighted mean (the core RW-SHAP formula)
    score = (phi_norm ** alpha) * (u_norm ** (1.0 - alpha))

    # Optional sigmoid sharpening gate
    if use_sigmoid:
        u_z   = zscore_safe(uq_pos)
        gate  = expit(gamma * u_z)   # sigmoid(γ · zscore(UQ))
        score = score * gate

    return score


def loo_tune(k, shap_mat, uq_mat, alpha_grid, gamma_grid):
    """
    Find the best (alpha, gamma) for fold k using leave-one-out
    Spearman rank correlation.

    Objective: find the α that makes fold k's feature ranking
    agree most with the consensus of the OTHER 9 folds —
    using the SAME α for both fold k and the other folds.

    This avoids circular tuning: fold k must match an independent
    consensus, not just its own ranking.

    Parameters
    ----------
    k          : int  fold index (0-based)
    shap_mat   : array [K, F]  SHAP scores for all folds
    uq_mat     : array [K, F]  UQ scores for all folds
    alpha_grid : array         candidate α values
    gamma_grid : array         candidate γ values

    Returns
    -------
    best_alpha : float  selected α for fold k
    best_gamma : float  selected γ for fold k
    best_rho   : float  Spearman ρ achieved
    """
    other_folds = [j for j in range(K_FOLDS) if j != k]

    best_rho   = -2.0
    best_alpha = alpha_grid[len(alpha_grid) // 2]  # default: middle of grid
    best_gamma = gamma_grid[0]

    for alpha in alpha_grid:
        for gamma in gamma_grid:

            # Fold k's score at this (α, γ)
            sk = compute_score(shap_mat[k], uq_mat[k], alpha, gamma)

            # LOO consensus: mean score of other 9 folds at same (α, γ)
            other_scores = np.stack([
                compute_score(shap_mat[j], uq_mat[j], alpha, gamma)
                for j in other_folds
            ], axis=0).mean(axis=0)

            # Spearman rank correlation between fold k and consensus
            rho, _ = spearmanr(sk, other_scores)
            if np.isnan(rho):
                continue

            if rho > best_rho:
                best_rho   = rho
                best_alpha = alpha
                best_gamma = gamma

    return best_alpha, best_gamma, best_rho


def check_collapse(best_alphas, best_gammas, alpha_grid, gamma_grid):
    """
    Detect if tuned parameters collapsed to the grid boundary.

    Collapse means the optimal value lies outside the search grid —
    the grid needs to be expanded.

    Returns a dict with:
      alpha_collapse, gamma_collapse : bool
      alpha_boundary_frac, gamma_boundary_frac : fraction of folds at boundary
    """
    a_lo, a_hi = alpha_grid.min(), alpha_grid.max()
    g_lo, g_hi = gamma_grid.min(), gamma_grid.max()

    a_boundary = np.mean(
        (np.abs(best_alphas - a_lo) < BOUNDARY_TOL) |
        (np.abs(best_alphas - a_hi) < BOUNDARY_TOL)
    )
    g_boundary = np.mean(
        (np.abs(best_gammas - g_lo) < BOUNDARY_TOL) |
        (np.abs(best_gammas - g_hi) < BOUNDARY_TOL)
    )

    return {
        'alpha_collapse'      : a_boundary >= COLLAPSE_FRAC,
        'gamma_collapse'      : g_boundary >= COLLAPSE_FRAC,
        'alpha_boundary_frac' : a_boundary,
        'gamma_boundary_frac' : g_boundary,
    }


def rank_consistency_W(scores_matrix):
    """
    Compute Kendall's coefficient of concordance W.

    W measures how consistent the feature rankings are across
    all K folds:
      W = 1.0 → all folds agree perfectly on the ranking
      W = 0.0 → rankings are random (no consistency)

    Formula:
        W = 12 * S_W / (K^2 * (F^3 - F))
    where S_W = sum_i (R_i - R̄)^2
          R_i = sum of ranks for feature i across all folds

    Parameters
    ----------
    scores_matrix : array [K, F]

    Returns
    -------
    W : float in [0, 1]
    """
    n, k  = scores_matrix.shape    # n=K_FOLDS, k=n_feats
    ranks = np.apply_along_axis(
        lambda x: pd.Series(x).rank(ascending=False).values,
        axis=1, arr=scores_matrix)
    rank_sums = ranks.sum(axis=0)
    mean_rs   = rank_sums.mean()
    S         = np.sum((rank_sums - mean_rs) ** 2)
    W         = 12 * S / (n ** 2 * (k ** 3 - k))
    return float(np.clip(W, 0, 1))


# ══════════════════════════════════════════════════════════════
#  STEP 1: LOAD FEATURE METADATA
# ══════════════════════════════════════════════════════════════

print("=" * 65)
print("  RW-SHAP v2 — Balanced Reliability-Weighted SHAP")
print(f"  Formula: S_i = φ̃^α · ũ^(1-α)"
      + ("  · sigmoid(γ·ũ)" if USE_SIGMOID else ""))
print("  α calibrated per fold via leave-one-out Spearman ρ")
print("=" * 65)

# Load feature matrix to identify Pool B column names
df_full   = pd.read_csv(FULL_CSV)
label_col = 'label' if 'label' in df_full.columns else \
            df_full.select_dtypes(include='object').columns[0]
skip      = {label_col, 'file_id', 'frame_id', 'index', 'Unnamed: 0'}
feat_cols = [c for c in df_full.columns
             if c not in skip
             and pd.api.types.is_numeric_dtype(df_full[c])
             and c.startswith(POOL_PREFIX)]
n_feats   = len(feat_cols)
print(f"\nPool B features ({n_feats}): {feat_cols}")


# ══════════════════════════════════════════════════════════════
#  STEP 2: LOAD PER-FOLD SHAP + UQ SCORES
# ══════════════════════════════════════════════════════════════

print(f"\nLoading per-fold SHAP and UQ scores from {SRC_DIR} ...")

# These matrices hold the raw scores computed by earlier pipeline steps
shap_mat  = np.zeros((K_FOLDS, n_feats))   # SHAP importance
uq_mat    = np.zeros((K_FOLDS, n_feats))   # UQ_change
old_contr = np.zeros((K_FOLDS, n_feats))   # linear combination (old method)
train_accs = np.zeros(K_FOLDS)
test_accs  = np.zeros(K_FOLDS)

for k in range(K_FOLDS):
    run_num  = k + 1

    # Load contribution scores CSV (created by run_contribution_ranking.py)
    # This file contains shap_value, uq_change, contribution per feature
    csv_path = os.path.join(SRC_DIR, f'run_{run_num:02d}',
                             f'run_{run_num:02d}_contribution_scores.csv')
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Missing: {csv_path}\n"
            "Run run_contribution_ranking.py first.")

    df_k = pd.read_csv(csv_path).set_index('feature').reindex(feat_cols)
    shap_mat[k]  = df_k['shap_value'].values
    uq_mat[k]    = df_k['uq_change'].values
    old_contr[k] = df_k['contribution'].values

    # Load accuracy from the UQ classification report
    # (format: "Train Accuracy : 0.9871  (98.71%)")
    train_accs[k] = test_accs[k] = 0.0
    for search_dir in [
        os.path.join(BASE_DIR, 'unc_poolB_10fold',  f'run_{run_num:02d}'),
        os.path.join(BASE_DIR, 'poolB_SHAP_10fold', f'run_{run_num:02d}'),
    ]:
        rep_path = os.path.join(search_dir,
                                f'run_{run_num:02d}_classification_report.txt')
        if os.path.isfile(rep_path):
            with open(rep_path) as fh:
                for line in fh:
                    if 'Train Accuracy' in line and ':' in line:
                        try:
                            val = line.split(':')[1].strip()
                            train_accs[k] = float(
                                val.split('%')[0].strip().split('(')[-1]) / 100
                        except Exception:
                            pass
                    elif ('Test  Accuracy' in line or
                          'Test Accuracy' in line) and ':' in line:
                        try:
                            val = line.split(':')[1].strip()
                            test_accs[k] = float(
                                val.split('%')[0].strip().split('(')[-1]) / 100
                        except Exception:
                            pass
            if test_accs[k] > 0:
                break

    print(f"  run_{run_num:02d}: train={train_accs[k]*100:.2f}%  "
          f"test={test_accs[k]*100:.2f}%  loaded ✓")

mean_train = train_accs.mean(); std_train = train_accs.std()
mean_test  = test_accs.mean();  std_test  = test_accs.std()


# ══════════════════════════════════════════════════════════════
#  STEP 3: AUTO-TUNE α PER FOLD VIA LOO SPEARMAN
# ══════════════════════════════════════════════════════════════

print(f"\nTuning α per fold via LOO Spearman ρ ...")
print(f"Alpha grid: {ALPHA_GRID.min():.2f} → {ALPHA_GRID.max():.2f} "
      f"({len(ALPHA_GRID)} values)")
print(f"Gamma grid: {GAMMA_GRID.min():.4f} → {GAMMA_GRID.max():.2f} "
      f"({len(GAMMA_GRID)} values)")
print(f"Sigmoid gate: {'ENABLED' if USE_SIGMOID else 'DISABLED'}\n")

# Arrays to store per-fold tuning results
rel_mat     = np.zeros((K_FOLDS, n_feats))
rel_ranks   = np.zeros((K_FOLDS, n_feats), dtype=int)
best_alphas = np.zeros(K_FOLDS)
best_gammas = np.zeros(K_FOLDS)
best_rhos   = np.zeros(K_FOLDS)

for k in range(K_FOLDS):
    run_num = k + 1

    # Find the best (α, γ) for this fold via LOO calibration
    alpha_k, gamma_k, rho_k = loo_tune(
        k, shap_mat, uq_mat, ALPHA_GRID, GAMMA_GRID)

    # Store tuned parameters
    best_alphas[k] = alpha_k
    best_gammas[k] = gamma_k
    best_rhos[k]   = rho_k

    # Compute the reliability score for this fold using the tuned α
    rel_mat[k]   = compute_score(shap_mat[k], uq_mat[k], alpha_k, gamma_k)
    rel_ranks[k] = make_rank(rel_mat[k])
    top_feat     = feat_cols[np.argmax(rel_mat[k])]

    print(f"  Fold {run_num:02d}: α={alpha_k:.3f}  γ={gamma_k:.4f}  "
          f"ρ={rho_k:.4f}  top feature: {top_feat}")


# ── Anti-collapse check ────────────────────────────────────────
# If most folds selected the grid boundary, the search range is too narrow
collapse = check_collapse(best_alphas, best_gammas, ALPHA_GRID, GAMMA_GRID)

print(f"\n  Anti-collapse check:")
print(f"    Alpha boundary hits: {collapse['alpha_boundary_frac']*100:.0f}%  "
      f"({'COLLAPSE!' if collapse['alpha_collapse'] else 'ok'})")
print(f"    Gamma boundary hits: {collapse['gamma_boundary_frac']*100:.0f}%  "
      f"({'COLLAPSE!' if collapse['gamma_collapse'] else 'ok'})")

if collapse['alpha_collapse'] or collapse['gamma_collapse']:
    print("\n  *** COLLAPSE — expanding grid and re-tuning ***")

    ALPHA_GRID_EX = np.linspace(0.01, 0.99, 49) \
        if collapse['alpha_collapse'] else ALPHA_GRID
    GAMMA_GRID_EX = np.logspace(-3, 4, 70) \
        if collapse['gamma_collapse'] else GAMMA_GRID

    for k in range(K_FOLDS):
        run_num = k + 1
        alpha_k, gamma_k, rho_k = loo_tune(
            k, shap_mat, uq_mat, ALPHA_GRID_EX, GAMMA_GRID_EX)
        best_alphas[k] = alpha_k
        best_gammas[k] = gamma_k
        best_rhos[k]   = rho_k
        rel_mat[k]     = compute_score(
            shap_mat[k], uq_mat[k], alpha_k, gamma_k)
        rel_ranks[k]   = make_rank(rel_mat[k])
        print(f"    Fold {run_num:02d}: α={alpha_k:.3f}  γ={gamma_k:.4f}  "
              f"ρ={rho_k:.4f}")

    ALPHA_GRID = ALPHA_GRID_EX
    GAMMA_GRID = GAMMA_GRID_EX
    collapse   = check_collapse(best_alphas, best_gammas,
                                 ALPHA_GRID, GAMMA_GRID)
    if collapse['alpha_collapse'] or collapse['gamma_collapse']:
        print("\n  WARNING: collapse persists — data may be "
              "SHAP-dominated by a single dominant feature. "
              "The score still produces meaningful rankings.")

# ── Ranking consistency (Kendall's W) ────────────────────────
# W close to 1 means the ranking is stable across all 10 folds
W_rel = rank_consistency_W(rel_mat)

print(f"\n  α summary  : {best_alphas.mean():.3f} ± {best_alphas.std():.3f}")
print(f"  γ summary  : {best_gammas.mean():.4f} ± {best_gammas.std():.4f}")
print(f"  LOO ρ mean : {best_rhos.mean():.4f} ± {best_rhos.std():.4f}")
print(f"  Kendall W  : {W_rel:.4f}  (1.0 = perfect agreement across folds)")


# ══════════════════════════════════════════════════════════════
#  STEP 4: AGGREGATE — 10-FOLD MEAN RANKING
# ══════════════════════════════════════════════════════════════

# Average scores across all 10 folds
mean_rel  = rel_mat.mean(axis=0)
std_rel   = rel_mat.std(axis=0)
mean_shap = np.abs(shap_mat).mean(axis=0)          # abs for magnitude
mean_uq   = np.clip(uq_mat, 0, None).mean(axis=0)  # clip negatives
mean_old  = old_contr.mean(axis=0)

# Final ranks from averaged scores
final_rel_rank  = make_rank(mean_rel)
final_shap_rank = make_rank(mean_shap)
final_uq_rank   = make_rank(mean_uq)
final_old_rank  = make_rank(mean_old)

mean_rel_rank = rel_ranks.mean(axis=0)
std_rel_rank  = rel_ranks.std(axis=0)

# Build summary DataFrame sorted by reliability rank
sum_df = pd.DataFrame({
    'feature'      : feat_cols,
    'rel_rank'     : final_rel_rank,
    'mean_rel'     : mean_rel,
    'std_rel'      : std_rel,
    'shap_rank'    : final_shap_rank,
    'mean_shap'    : mean_shap,
    'uq_rank'      : final_uq_rank,
    'mean_uq'      : mean_uq,
    'old_rank'     : final_old_rank,
    'mean_old'     : mean_old,
    'mean_rel_rank': mean_rel_rank,
    'std_rel_rank' : std_rel_rank,
}).sort_values('rel_rank').reset_index(drop=True)

feats_by_rel = sum_df['feature'].tolist()

# Print final ranking with rank movements vs SHAP and UQ
print(f"\n  RW-SHAP v2 Ranking (10-fold avg, α={best_alphas.mean():.3f}):")
print(f"  {'Rank':<5}{'Feature':<25}{'Score':>9}  "
      f"{'SHAP_r':>7}  {'UQ_r':>6}  {'Δ_vs_SHAP':>10}")
print(f"  {'-'*65}")
for _, row in sum_df.iterrows():
    d_shap = int(row['shap_rank']) - int(row['rel_rank'])
    arrow  = '↑' if d_shap > 0 else ('↓' if d_shap < 0 else '=')
    print(f"  #{int(row['rel_rank']):<4}{row['feature']:<25}"
          f"  {row['mean_rel']:>7.5f}"
          f"  #{int(row['shap_rank']):<6}"
          f"  #{int(row['uq_rank']):<5}"
          f"  {arrow}{abs(d_shap)}")

# Save summary CSV and text report
sum_df.to_csv(os.path.join(SUM_DIR, 'reliability_summary_v2.csv'), index=False)

with open(os.path.join(SUM_DIR, 'reliability_report_v2.txt'), 'w') as f:
    f.write("=" * 65 + "\n")
    f.write("  RW-SHAP v2 — Balanced Reliability-Weighted SHAP\n")
    f.write("=" * 65 + "\n\n")
    f.write(f"  Sigmoid gate : {'ENABLED' if USE_SIGMOID else 'DISABLED'}\n")
    f.write(f"  α summary    : {best_alphas.mean():.3f} ± "
            f"{best_alphas.std():.3f}\n")
    f.write(f"  LOO ρ        : {best_rhos.mean():.4f} ± "
            f"{best_rhos.std():.4f}\n")
    f.write(f"  Kendall W    : {W_rel:.4f}\n\n")
    f.write(f"  Train Acc    : {mean_train*100:.2f}% ± "
            f"{std_train*100:.2f}%\n")
    f.write(f"  Test  Acc    : {mean_test*100:.2f}% ± "
            f"{std_test*100:.2f}%\n\n")
    f.write("  Per-Fold Parameters:\n")
    f.write(f"  {'Fold':<10}{'Alpha':>8}{'Gamma':>10}{'Rho':>10}\n")
    for k in range(K_FOLDS):
        f.write(f"  run_{k+1:02d}    "
                f"{best_alphas[k]:>8.3f}"
                f"{best_gammas[k]:>10.4f}"
                f"{best_rhos[k]:>10.4f}\n")

print("  Saved: reliability_report_v2.txt")
print("  Saved: reliability_summary_v2.csv")


# ══════════════════════════════════════════════════════════════
#  STEP 5: PER-FOLD PLOTS
# ══════════════════════════════════════════════════════════════

print(f"\nGenerating per-fold plots ...")

for k in range(K_FOLDS):
    run_num = k + 1
    run_dir = os.path.join(ROOT_OUT, f'run_{run_num:02d}')
    os.makedirs(run_dir, exist_ok=True)
    prefix  = f'run_{run_num:02d}'

    # Build per-fold DataFrame
    fold_df = pd.DataFrame({
        'feature'         : feat_cols,
        'rel_score'       : rel_mat[k],
        'rel_rank'        : rel_ranks[k],
        'shap_value'      : np.abs(shap_mat[k]),
        'shap_rank'       : make_rank(np.abs(shap_mat[k])),
        'uq_change'       : np.clip(uq_mat[k], 0, None),
        'uq_rank'         : make_rank(np.clip(uq_mat[k], 0, None)),
        'old_contribution': old_contr[k],
        'old_rank'        : make_rank(old_contr[k]),
        'alpha_used'      : best_alphas[k],
        'gamma_used'      : best_gammas[k],
        'loo_rho'         : best_rhos[k],
    }).sort_values('rel_rank').reset_index(drop=True)

    fold_df.to_csv(
        os.path.join(run_dir, f'{prefix}_reliability_scores_v2.csv'),
        index=False)

    # ── Plot 1: Reliability bar chart ────────────────────────
    df_p = fold_df.sort_values('rel_score', ascending=True)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(df_p['feature'], df_p['rel_score'],
            color=[feat_color(f) for f in df_p['feature']],
            edgecolor='white', lw=0.5)
    for _, row in df_p.iterrows():
        ax.text(row['rel_score'] + df_p['rel_score'].max() * 0.012,
                list(df_p['feature']).index(row['feature']),
                f"{row['rel_score']:.5f}  "
                f"[S#{int(row['shap_rank'])} U#{int(row['uq_rank'])}]",
                va='center', fontsize=7.5)
    ax.set_xlabel(
        f'Reliability Score S = φ̃^{best_alphas[k]:.3f} · ũ^{1-best_alphas[k]:.3f}'
        f'  (ρ={best_rhos[k]:.4f})',
        fontsize=10)
    ax.set_title(
        f'RW-SHAP v2 Feature Ranking — Fold {run_num:02d}\n'
        f'Train={train_accs[k]*100:.1f}%  Test={test_accs[k]*100:.1f}%  '
        f'α={best_alphas[k]:.3f}',
        fontsize=11, fontweight='bold')
    ax.set_xlim(0, df_p['rel_score'].max() * 1.35)
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_reliability_bar_v2.png')

    # ── Plot 2: Alpha balance spectrum ────────────────────────
    # Shows where this fold's auto-tuned α sits on the UQ–SHAP axis
    # and how the mean score across features responds to α changes
    fig, ax = plt.subplots(figsize=(10, 4))
    spectrum    = np.linspace(0, 1, 200)
    mean_scores = [compute_score(shap_mat[k], uq_mat[k],
                                 a, best_gammas[k]).mean()
                   for a in spectrum]
    ax.plot(spectrum, mean_scores, color=COL_REL, lw=2)
    ax.axvline(best_alphas[k], color='black', lw=2, ls='--',
               label=f'Selected α={best_alphas[k]:.3f}  (ρ={best_rhos[k]:.4f})')
    ax.axvline(0.5, color='grey', lw=1, ls=':', alpha=0.6,
               label='α=0.5 (equal weight)')
    ax.fill_betweenx([0, max(mean_scores) * 1.1], 0, 0.5,
                     alpha=0.07, color=COL_UQ, label='UQ-dominant')
    ax.fill_betweenx([0, max(mean_scores) * 1.1], 0.5, 1,
                     alpha=0.07, color=COL_SHAP, label='SHAP-dominant')
    ax.set_xlabel('α  (0=pure UQ  →  1=pure SHAP)', fontsize=11)
    ax.set_ylabel('Mean score across features', fontsize=10)
    ax.set_title(f'Alpha Balance Spectrum — Fold {run_num:02d}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_alpha_balance_v2.png')

    # ── Plot 3: Four-method comparison bar ────────────────────
    # Side-by-side: RW-SHAP v2, SHAP, UQ, old linear method
    x  = np.arange(n_feats)
    w  = 0.20
    r_inv  = (n_feats + 1) - fold_df['rel_rank'].values
    sh_inv = (n_feats + 1) - fold_df['shap_rank'].values
    uq_inv = (n_feats + 1) - fold_df['uq_rank'].values
    ol_inv = (n_feats + 1) - fold_df['old_rank'].values
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.bar(x - 1.5*w, r_inv,  w, color=COL_REL,  label='RW-SHAP v2', alpha=0.9)
    ax.bar(x - 0.5*w, sh_inv, w, color=COL_SHAP, label='SHAP',       alpha=0.8)
    ax.bar(x + 0.5*w, uq_inv, w, color=COL_UQ,   label='UQ',         alpha=0.8)
    ax.bar(x + 1.5*w, ol_inv, w, color=COL_OLD,  label='Old linear', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(fold_df['feature'], rotation=35, ha='right', fontsize=8)
    ax.set_ylabel('Inverted rank  (higher = more important)', fontsize=11)
    ax.set_title(f'Four-Method Rank Comparison — Fold {run_num:02d}  '
                 f'α={best_alphas[k]:.3f}  ρ={best_rhos[k]:.4f}',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    plt.tight_layout()
    savefig(run_dir, f'{prefix}_four_method_bar_v2.png')

    n_saved = len([f for f in os.listdir(run_dir)
                   if os.path.isfile(os.path.join(run_dir, f))])
    print(f"  run_{run_num:02d}: {n_saved} files  "
          f"α={best_alphas[k]:.3f}  ρ={best_rhos[k]:.4f}")


# ══════════════════════════════════════════════════════════════
#  STEP 6: SUMMARY PLOTS
# ══════════════════════════════════════════════════════════════

print(f"\nGenerating summary plots ...")

# ── S1: α, γ, ρ per fold (3-panel bar chart) ─────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
fold_nums = range(1, K_FOLDS + 1)

axes[0].bar(fold_nums, best_alphas, color=COL_REL,
            edgecolor='white', lw=0.5, alpha=0.85)
axes[0].axhline(best_alphas.mean(), color='black', ls='--', lw=1.3,
                label=f'Mean α={best_alphas.mean():.3f}')
axes[0].axhline(0.5, color='grey', ls=':', lw=1, alpha=0.7,
                label='α=0.5 (equal weight)')
for i, a in enumerate(best_alphas):
    axes[0].text(i + 1, a + 0.01, f'{a:.3f}',
                 ha='center', fontsize=8, fontweight='bold')
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('Alpha (SHAP weight)', fontsize=11)
axes[0].set_title('Auto-tuned α per Fold  (1=pure SHAP, 0=pure UQ)',
                  fontsize=11, fontweight='bold')
axes[0].legend(fontsize=9)

axes[1].bar(fold_nums, best_gammas, color=COL_UQ,
            edgecolor='white', lw=0.5, alpha=0.85)
axes[1].axhline(best_gammas.mean(), color='black', ls='--', lw=1.3,
                label=f'Mean γ={best_gammas.mean():.4f}')
axes[1].set_ylabel('Gamma (sigmoid sharpness)', fontsize=11)
axes[1].set_title('Auto-tuned γ per Fold  (only relevant if USE_SIGMOID=True)',
                  fontsize=11, fontweight='bold')
axes[1].legend(fontsize=9)

axes[2].bar(fold_nums, best_rhos, color='#27AE60',
            edgecolor='white', lw=0.5, alpha=0.85)
axes[2].axhline(best_rhos.mean(), color='black', ls='--', lw=1.3,
                label=f'Mean ρ={best_rhos.mean():.4f}')
for i, r in enumerate(best_rhos):
    axes[2].text(i + 1, r + 0.005, f'{r:.3f}',
                 ha='center', fontsize=8, fontweight='bold')
axes[2].set_ylim(0, 1.1)
axes[2].set_ylabel('LOO Spearman ρ', fontsize=11)
axes[2].set_xlabel('Fold', fontsize=11)
axes[2].set_xticks(fold_nums)
axes[2].set_xticklabels([f'run_{k+1:02d}' for k in range(K_FOLDS)],
                         rotation=25, fontsize=9)
axes[2].set_title('LOO Spearman ρ per Fold  (higher = fold agrees with consensus)',
                  fontsize=11, fontweight='bold')
axes[2].legend(fontsize=9)
plt.tight_layout()
savefig(SUM_DIR, 'alpha_gamma_rho_per_fold.png')

# ── S2: Alpha sensitivity plot ────────────────────────────────
# Shows how each feature's rank changes as α sweeps 0 → 1
# Flat lines = robust features; crossing lines = α-sensitive features
alpha_test  = np.linspace(0, 1, 50)
rank_matrix = np.zeros((len(alpha_test), n_feats))
for ai, a in enumerate(alpha_test):
    scores_a = np.zeros(n_feats)
    for k in range(K_FOLDS):
        scores_a += compute_score(shap_mat[k], uq_mat[k],
                                  a, best_gammas.mean())
    rank_matrix[ai] = make_rank(scores_a / K_FOLDS)

fig, ax = plt.subplots(figsize=(14, 7))
for fi, feat in enumerate(feat_cols):
    ax.plot(alpha_test, rank_matrix[:, fi],
            color=feat_color(feat), lw=1.8, alpha=0.8,
            label=feat.replace(POOL_PREFIX, ''))
ax.axvline(best_alphas.mean(), color='black', lw=2, ls='--',
           label=f'Selected α={best_alphas.mean():.3f}')
ax.axvline(0.5, color='grey', lw=1, ls=':', alpha=0.6,
           label='α=0.5 (equal)')
ax.set_xlabel('Alpha  (0=pure UQ  →  1=pure SHAP)', fontsize=12)
ax.set_ylabel('Feature rank  (1=best)', fontsize=12)
ax.set_title('Rank Sensitivity to Alpha\n'
             'Flat lines = stable rank regardless of α  |  '
             'Crossing lines = rank depends on α',
             fontsize=12, fontweight='bold')
ax.invert_yaxis()
ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1),
          fontsize=7, ncol=1)
plt.tight_layout()
savefig(SUM_DIR, 'alpha_sensitivity_v2.png')

# ── S3: Rank stability heatmap across 10 folds ───────────────
fig, ax = plt.subplots(figsize=(16, 6))
sns.heatmap(rel_ranks, annot=True, fmt='d', cmap='RdYlGn_r',
            xticklabels=feat_cols,
            yticklabels=[f'run_{k+1:02d}' for k in range(K_FOLDS)],
            linewidths=0.3, linecolor='grey',
            cbar_kws={'label': 'Rank  (1=best)'}, ax=ax,
            vmin=1, vmax=n_feats)
ax.set_title(
    f'RW-SHAP v2 Rank Stability (10 folds)  |  Kendall W={W_rel:.4f}\n'
    f'Mean α={best_alphas.mean():.3f} ± {best_alphas.std():.3f}',
    fontsize=12, fontweight='bold', pad=10)
ax.set_xticklabels(feat_cols, rotation=40, ha='right', fontsize=8)
plt.tight_layout()
savefig(SUM_DIR, 'reliability_rank_stability_heatmap_v2.png')

# ── S4: Rank delta vs SHAP, UQ, old method ───────────────────
fig, axes = plt.subplots(1, 3, figsize=(21, 7))
for ax, (other_rank, title) in zip(axes, [
    (final_shap_rank, 'RW-SHAP v2 vs SHAP'),
    (final_uq_rank,   'RW-SHAP v2 vs UQ'),
    (final_old_rank,  'RW-SHAP v2 vs Old (linear)'),
]):
    delta   = other_rank - final_rel_rank
    order   = np.argsort(delta)[::-1]
    names_o = [feat_cols[i] for i in order]
    delta_o = delta[order]
    colors_d = ['#2ECC71' if v > 0 else
                ('#E74C3C' if v < 0 else '#95A5A6')
                for v in delta_o]
    ax.barh(range(n_feats), delta_o,
            color=colors_d, edgecolor='white', lw=0.5)
    ax.set_yticks(range(n_feats))
    ax.set_yticklabels(names_o, fontsize=8.5)
    ax.axvline(0, color='black', lw=0.9)
    ax.set_xlabel('Rank change  (+ = feature rose in RW-SHAP v2)')
    ax.set_title(title, fontsize=10, fontweight='bold')
    for i, v in enumerate(delta_o):
        ax.text(v + (0.1 if v >= 0 else -0.1), i, f'{v:+d}',
                va='center', fontsize=8,
                ha='left' if v >= 0 else 'right')
    ax.legend(handles=[
        mpatches.Patch(color='#2ECC71', label='Rose in v2'),
        mpatches.Patch(color='#E74C3C', label='Fell in v2'),
        mpatches.Patch(color='#95A5A6', label='No change'),
    ], fontsize=7, loc='lower right')
fig.suptitle(
    f'Rank Movement — RW-SHAP v2 vs Other Methods\n'
    f'Mean α={best_alphas.mean():.3f}  LOO ρ={best_rhos.mean():.4f}  '
    f'Kendall W={W_rel:.4f}',
    fontsize=13, fontweight='bold')
plt.tight_layout()
savefig(SUM_DIR, 'reliability_rank_delta_v2.png')

# ── S5: Summary table ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(22, max(8, n_feats * 0.65)))
ax.axis('off')
col_h = ['Rank', 'Feature', 'Score', 'SHAP\nRank', 'UQ\nRank',
         'Old\nRank', 'Δ vs\nSHAP', 'Δ vs\nUQ', 'Δ vs\nOld',
         'α', 'LOO ρ']
tdata = []
for _, row in sum_df.iterrows():
    d_s = int(row['shap_rank']) - int(row['rel_rank'])
    d_u = int(row['uq_rank'])   - int(row['rel_rank'])
    d_o = int(row['old_rank'])  - int(row['rel_rank'])
    tdata.append([
        int(row['rel_rank']), row['feature'],
        f"{row['mean_rel']:.5f}",
        f"#{int(row['shap_rank'])}",
        f"#{int(row['uq_rank'])}",
        f"#{int(row['old_rank'])}",
        f"{d_s:+d}", f"{d_u:+d}", f"{d_o:+d}",
        f"{best_alphas.mean():.3f}",
        f"{best_rhos.mean():.4f}",
    ])
tbl = ax.table(cellText=tdata, colLabels=col_h,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(8.5)
tbl.scale(1, 1.7)
for c in range(len(col_h)):
    tbl[0, c].set_facecolor('#2C3E50')
    tbl[0, c].set_text_props(color='white', fontweight='bold')
for ri, rd in enumerate(tdata):
    bg = '#F5F5F5' if ri % 2 == 0 else 'white'
    for c in range(len(col_h)):
        tbl[ri+1, c].set_facecolor(bg)
        tbl[ri+1, c].set_edgecolor('#DDD')
    # Green/red cells for rank movement columns
    for ci, dval in enumerate([rd[6], rd[7], rd[8]]):
        v = int(dval.replace('+', ''))
        if   v > 0: tbl[ri+1, ci+6].set_facecolor('#d5f5e3')
        elif v < 0: tbl[ri+1, ci+6].set_facecolor('#fadbd8')
ax.set_title(
    f'RW-SHAP v2 Summary — Pool B (10-fold avg)\n'
    f'Train={mean_train*100:.2f}%  Test={mean_test*100:.2f}% ± '
    f'{std_test*100:.2f}%  |  '
    f'Mean α={best_alphas.mean():.3f}  LOO ρ={best_rhos.mean():.4f}  '
    f'Kendall W={W_rel:.4f}',
    fontsize=11, fontweight='bold', pad=20)
plt.tight_layout()
savefig(SUM_DIR, 'reliability_summary_table_v2.png')

# ── FINAL PRINT ───────────────────────────────────────────────
print(f"\n{'='*65}")
print(f"  ALL DONE")
print(f"{'='*65}")
print(f"  Test  Acc  : {mean_test*100:.2f}% ± {std_test*100:.2f}%")
print(f"  Mean α     : {best_alphas.mean():.4f} ± {best_alphas.std():.4f}")
print(f"  LOO ρ mean : {best_rhos.mean():.4f} ± {best_rhos.std():.4f}")
print(f"  Kendall W  : {W_rel:.4f}")
print(f"\n  Output: {ROOT_OUT}/")
print(f"  summary/ ({len(os.listdir(SUM_DIR))} files)")
for k in range(K_FOLDS):
    rd = os.path.join(ROOT_OUT, f'run_{k+1:02d}')
    n  = len(os.listdir(rd)) if os.path.isdir(rd) else 0
    print(f"  run_{k+1:02d}/  ({n} files)")
print(f"{'='*65}")
