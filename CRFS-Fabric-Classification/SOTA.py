"""
SOTA Comparison - Fabric Material Classification (AWR2243 Radar)
================================================================

This script concatenates and shuffles rotational and translational 
datasets before running SOTA feature comparisons.

Rotational data  (paper_obj2.csv)    : data collected at random angles 
                                       between 10 and 40 degrees
Translational data (paper_obj2NEW.csv): data collected at distances 
                                        between 10 cm and 1 m

Both datasets are concatenated and shuffled to create a combined dataset 
on which all 9 SOTA feature sets are evaluated.

This single script was used to reproduce results from 9 different papers.
For each paper, only the features mentioned in that paper were selected 
in the classification script, and the output folder was changed accordingly.

The same LightGBM model and 10-fold stratified split was used for all 
papers to ensure a fair comparison on our dataset.

Papers compared:

Set 1 - MRF-Based Features [14]
        Features  : MRF, SignalAmplitude, TargetRange, CIR_Mean, 
                    CIR_Std, CIR_Max, SpatialVar, SpatialMean

Set 2 - Range FFT and Data Cube [5]
        Features  : (features from that paper)

Set 3 - Range-Angle Heatmap [2]
        Features  : (features from that paper)

Set 4 - Wavelet and Spectral [8]
        Features  : (features from that paper)

Set 5 - Transmittance [7]
        Features  : (features from that paper)

Set 6 - Time-Domain [10]
        Features  : (features from that paper)

Set 7 - Range Cross-Range [13]
        Features  : (features from that paper)

Set 8 - SAR Features [4]
        Features  : (features from that paper)

Set 9 - Capon Beamforming [12]
        Features  : (features from that paper)

To reproduce any comparison:
    1. Run this script first to generate paper_concat_shuffled.csv
    2. In the classification script, change FEATURE_COLUMNS to the 
       features of that paper
    3. Change OUTPUT_DIR to the corresponding folder name
    4. Run the classification script

Output files:
    paper_concat.csv          : concatenated rotational + translational data
    paper_concat_shuffled.csv : shuffled version used for all comparisons
"""

import os
import numpy as np
import pandas as pd

# ── PATHS — update if needed ──────────────────────────────
BASE_DIR = os.getcwd()

ROTATIONAL_CSV   = os.path.join(BASE_DIR, 'paper_obj2.csv')      # rotational data
TRANSLATIONAL_CSV = os.path.join(BASE_DIR, 'paper_obj2NEW.csv')  # translational data
CONCAT_CSV       = os.path.join(BASE_DIR, 'paper_concat.csv')
SHUFFLED_CSV     = os.path.join(BASE_DIR, 'paper_concat_shuffled.csv')

RANDOM_SEED = 42
# ─────────────────────────────────────────────────────────

print("=" * 55)
print("  Concatenate + Shuffle  |  Rotational + Translational")
print("=" * 55)

# ── Load ─────────────────────────────────────────────────
if not os.path.isfile(ROTATIONAL_CSV):
    raise FileNotFoundError(f"Not found: {ROTATIONAL_CSV}")
if not os.path.isfile(TRANSLATIONAL_CSV):
    raise FileNotFoundError(f"Not found: {TRANSLATIONAL_CSV}")

print(f"\nLoading rotational data : {ROTATIONAL_CSV}")
df_rotational = pd.read_csv(ROTATIONAL_CSV)
print(f"  Shape: {df_rotational.shape}")

print(f"\nLoading translational data : {TRANSLATIONAL_CSV}")
df_translational = pd.read_csv(TRANSLATIONAL_CSV)
print(f"  Shape: {df_translational.shape}")

# ── Column check ─────────────────────────────────────────
if list(df_rotational.columns) != list(df_translational.columns):
    raise ValueError(
        "Column mismatch between the two CSVs!\n"
        f"  Rotational    : {list(df_rotational.columns)}\n"
        f"  Translational : {list(df_translational.columns)}"
    )

print(f"\n  Columns match: {len(df_rotational.columns)} columns ✓")

# ── Concatenate ──────────────────────────────────────────
df_concat = pd.concat([df_rotational, df_translational], ignore_index=True)

print(f"\nConcatenated shape: {df_concat.shape}")
print(f"  ({len(df_rotational)} rows from rotational + "
      f"{len(df_translational)} rows from translational)")

print(f"\n  Class distribution:")
for cls, cnt in df_concat['ClassName'].value_counts().sort_index().items():
    print(f"    {cls:<15}  {cnt:>8} rows")

df_concat.to_csv(CONCAT_CSV, index=False)
print(f"\n  Saved: {CONCAT_CSV}")

# ── Shuffle ──────────────────────────────────────────────
rng         = np.random.default_rng(RANDOM_SEED)
shuffle_idx = rng.permutation(len(df_concat))
df_shuffled = df_concat.iloc[shuffle_idx].reset_index(drop=True)

print(f"\n  Class distribution after shuffle (must match):")
for cls, cnt in df_shuffled['ClassName'].value_counts().sort_index().items():
    print(f"    {cls:<15}  {cnt:>8} rows")

df_shuffled.to_csv(SHUFFLED_CSV, index=False)
print(f"\n  Saved: {SHUFFLED_CSV}")

# ── Summary ──────────────────────────────────────────────
print("\n" + "="*55)
print("  SUMMARY")
print("="*55)
print(f"  paper_concat.csv          : {df_concat.shape[0]:>7} rows x "
      f"{df_concat.shape[1]} cols")
print(f"  paper_concat_shuffled.csv : {df_shuffled.shape[0]:>7} rows x "
      f"{df_shuffled.shape[1]} cols")
print(f"  Classes : {sorted(df_concat['ClassName'].unique().tolist())}")
print(f"  Seed    : {RANDOM_SEED}")
print("\n  DONE.")
