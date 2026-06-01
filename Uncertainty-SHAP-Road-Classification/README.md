# Uncertainty SHAP - Road Classification

## What this is about

SHAP gives one fixed importance score per feature but says nothing 
about whether that score is stable or trustworthy. In outdoor radar 
environments there is always noise, surface variation, and random 
signal changes. So a feature with a high SHAP score on one day might 
not have the same score on another day.

To solve this, we proposed combining SHAP importance with epistemic 
uncertainty quantification to produce a reliability weighted feature 
ranking. A parameter called alpha controls the balance between SHAP 
and uncertainty, and it is automatically tuned using cross validation 
instead of being fixed manually.

## Dataset

We collected our own road surface dataset by mounting the TI AWR2243 
radar on a baby stroller and pushing it over different surfaces.

Classes : Artificial grass, Laboratory floor, Natural grass, 
          Parking surface, Rocky road, Sand, Snow road, Wooden floor

Stroller setup : radar at 20 degree downward angle, 28 cm height 
                 from ground, Euclidean distance to surface 44.17 cm

Data size : 8 classes, 20 experiments per class, 1024 frames 
            per experiment

Why stroller : a stroller can go over sand, grass, snow and rocky 
terrain which most vehicles cannot. It is also lightweight and easy 
to carry anywhere making it suitable for real world testing.

## How the algorithm works

1. Extract 15 spectral features from road surface data
2. Compute SHAP importance for each feature using Random Forest
3. Compute uncertainty importance using UbiQTrees - permute each 
   feature and measure how much model confidence drops
4. Normalize both SHAP and uncertainty scores
5. Combine them using geometric mean with parameter alpha
6. For each fold, find the alpha that maximizes Spearman rank 
   correlation with leave one out consensus of other folds
7. Aggregate rankings across all 10 folds to get final ranking

## About alpha tuning

Alpha equal to 1 means pure SHAP ranking.
Alpha equal to 0 means pure uncertainty ranking.
Alpha between 0 and 1 balances both.

Mean alpha across 10 folds is 0.435, slightly favoring uncertainty 
stability over pure SHAP. Alpha varies from 0.05 to 0.90 across 
folds, which shows that different data subsets need different balance 
points. This is why we cannot fix alpha manually.

## Results

Test accuracy       : 94.24% with standard deviation of 3.30%
F1 score            : 0.9413
Auto calibrated alpha : 0.435
Spearman correlation  : 0.9982 across folds

Best performing class  : Lab floor at 99.59%
Lowest performing class : Parking surface at 88.47% due to 
                          high surface variability

## Key Observation

SP_dc_component and SP_spec_entropy rank first and second in both 
SHAP and uncertainty rankings making them unconditionally stable 
features. SP_peak_freq ranks last in both and is always dropped. 
Features where SHAP and uncertainty disagree are the ones where 
the auto calibrated alpha makes the most difference.
