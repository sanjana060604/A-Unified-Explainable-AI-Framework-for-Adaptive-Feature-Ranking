"""
SOTA Comparison - Wall Material Classification (AWR1843 Radar)
==============================================================

This single script was used to reproduce and compare results from 9 different 
papers on radar-based material classification.

For each paper, only the features mentioned in that paper were selected in 
FEATURE_COLUMNS, and OUTPUT_DIR was changed accordingly.

The same LightGBM model and 80-20 train-test split was used for all papers 
to ensure a fair comparison on our dataset.

Papers compared:

Set 1 - MRF-Based Features [14]
        Features  : MRF, SignalAmplitude, TargetRange, CIR_Mean, CIR_Std, 
                    CIR_Max, SpatialVar, SpatialMean
        OUTPUT_DIR: New_Feature_Set_1_Results

Set 2 - Range FFT and Data Cube [5]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_2_Results

Set 3 - Range-Angle Heatmap [2]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_3_Results

Set 4 - Wavelet and Spectral [8]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_4_Results

Set 5 - Transmittance [7]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_5_Results

Set 6 - Time-Domain [10]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_6_Results

Set 7 - Range Cross-Range [13]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_7_Results

Set 8 - SAR Features [4]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_8_Results

Set 9 - Capon Beamforming [12]
        Features  : (features from that paper)
        OUTPUT_DIR: New_Feature_Set_9_Results

To reproduce any comparison:
    1. Change FEATURE_COLUMNS to the features of that paper
    2. Change OUTPUT_DIR to the corresponding folder name
    3. Run the script

Results for each set are saved in their respective output folders.
"""



"""
Feature Set 1 Classification for AWR1843 Radar Material Classification - OPTIMIZED
Features: MRF, SignalAmplitude, TargetRange, CIR_Mean, CIR_Std, CIR_Max, SpatialVar, SpatialMean
Model: LightGBM
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix, 
                             precision_recall_fscore_support, cohen_kappa_score)
import lightgbm as lgb
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ==================== CONFIGURATION ====================
CSV_PATH = 'new_features_all_frames.csv'
OUTPUT_DIR = 'New_Feature_Set_1_Results'

FEATURE_COLUMNS = [
    'MRF', 'SignalAmplitude', 'TargetRange', 
    'CIR_Mean', 'CIR_Std', 'CIR_Max', 
    'SpatialVar', 'SpatialMean'
]

# ==================== LOAD DATA ====================
print("="*80)
print("FEATURE SET 1: MRF-BASED FEATURES - LightGBM")
print("="*80)
print(f"\nLoading data from: {CSV_PATH}")

required_cols = ['ClassName', 'FileName'] + FEATURE_COLUMNS
df = pd.read_csv(CSV_PATH, usecols=required_cols)

print(f"Total samples loaded: {len(df)}")
print(f"Columns selected: {len(FEATURE_COLUMNS)} features")

# ==================== DATA PREPROCESSING ====================
print("\n" + "="*80)
print("DATA PREPROCESSING")
print("="*80)

print(f"\nMissing values: {df[FEATURE_COLUMNS].isnull().sum().sum()}")

if df[FEATURE_COLUMNS].isnull().sum().sum() > 0:
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())

df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
if df[FEATURE_COLUMNS].isnull().sum().sum() > 0:
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())

print("\n" + "-"*80)
print("CLASS DISTRIBUTION")
print("-"*80)
print(df['ClassName'].value_counts())

# ==================== PREPARE FEATURES AND LABELS ====================
X = df[FEATURE_COLUMNS].values
y = df['ClassName'].values

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Class names: {label_encoder.classes_}")

# ==================== TRAIN-TEST SPLIT ====================
print("\n" + "="*80)
print("TRAIN-TEST SPLIT (80% - 20%)")
print("="*80)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.2, 
    random_state=RANDOM_STATE,
    stratify=y_encoded
)

print(f"Training: {len(X_train)} | Testing: {len(X_test)}")

# ==================== NORMALIZATION ====================
print("\n" + "="*80)
print("FEATURE NORMALIZATION")
print("="*80)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ==================== MODEL TRAINING ====================
print("\n" + "="*80)
print("TRAINING: LightGBM")
print("="*80)

model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    verbose=-1,
    force_col_wise=True
)

model.fit(X_train_scaled, y_train)

y_pred_train = model.predict(X_train_scaled)
y_pred_test  = model.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc  = accuracy_score(y_test,  y_pred_test)

print(f"✓ Training Accuracy: {train_acc*100:.2f}%")
print(f"✓ Testing Accuracy:  {test_acc*100:.2f}%")

# ==================== EVALUATION ====================
print("\n" + "="*80)
print("DETAILED EVALUATION - LightGBM")
print("="*80)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use('default')

print(f"\nTraining Accuracy: {train_acc*100:.2f}%")
print(f"Testing Accuracy:  {test_acc*100:.2f}%")
print(f"Overfitting Gap:   {(train_acc - test_acc)*100:.2f}%")

print(f"\n{'─'*80}")
print("CLASSIFICATION REPORT")
print(f"{'─'*80}")
print(classification_report(y_test, y_pred_test, target_names=label_encoder.classes_, digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_test)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_, ax=ax)
ax.set_title('Confusion Matrix - LightGBM', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/confusion_matrix_LightGBM.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature importance
importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': FEATURE_COLUMNS,
    'Importance': importances
}).sort_values('Importance', ascending=False)

print(f"\n{'─'*80}")
print("FEATURE IMPORTANCE")
print(f"{'─'*80}")
print(feature_importance_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance - LightGBM', fontsize=14, fontweight='bold')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/feature_importance_LightGBM.png', dpi=300)
plt.close()

feature_importance_df.to_csv(f'{OUTPUT_DIR}/feature_importance_LightGBM.csv', index=False)

# Additional metrics
precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_test, average='weighted')
kappa = cohen_kappa_score(y_test, y_pred_test)

print(f"\n{'─'*80}")
print(f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | Kappa: {kappa:.4f}")

# Per-class accuracy
class_accuracies = cm.diagonal() / cm.sum(axis=1)
per_class_df = pd.DataFrame({
    'Class':    label_encoder.classes_,
    'Accuracy': class_accuracies,
    'Samples':  cm.sum(axis=1)
}).sort_values('Accuracy', ascending=False)

per_class_df.to_csv(f'{OUTPUT_DIR}/per_class_accuracy.csv', index=False)

# ==================== FINAL SUMMARY ====================
print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE")
print("="*80)
print(f"📁 Results saved to: {OUTPUT_DIR}/")
print(f"🎯 LightGBM Test Accuracy: {test_acc*100:.2f}%")
print("="*80)
