# CRFS - Fabric Classification
CRFS - Condition-Robust spectral Feature Selection
## What this is about

When a radar is deployed in real life, it cannot always be at the 
exact same angle and distance every time. This causes a problem 
because SHAP rankings change depending on how the data was collected. 
A feature that is important at one angle might not be important at 
a different angle.

To solve this, we proposed CRFS which stands for Condition Robust 
spectral Feature Selection. The idea is to collect data in two 
different sensing conditions, get SHAP rankings for both separately, 
and then combine them using harmonic mean to get one unified ranking 
that works well for both conditions.

## Dataset

We collected our own fabric material dataset using the TI AWR2243 radar.

Classes : Cotton, Polyester, Nylon taffeta, Oxford weave, Leather

Two sensing conditions:

Rotational  : radar tilted at random angle between 10 and 40 degrees
Translational : radar moved from 10 cm to 1 m distance at 20 cm height

Data size : 5 classes, 20 experiments per class, 1024 frames 
            per experiment, collected for both conditions

## Why harmonic mean

We use harmonic mean instead of arithmetic or geometric mean because 
harmonic mean is dominated by the smaller value. If a feature scores 
high in one condition but low in another, harmonic mean pulls it down 
significantly. This automatically penalizes features that only work 
in one condition and keeps features that are robust in both.

## How the algorithm works

1. Extract 15 spectral features from both datasets
2. Train Random Forest and apply SHAP on translational data 
   to get ranking 1
3. Train Random Forest and apply SHAP on rotational data 
   to get ranking 2
4. Convert both rankings to normalized scores
5. Compute harmonic mean of both scores for each feature
6. Sort features by harmonic score to get unified ranking
7. Train final model on unified ranking and deploy on edge

## Results

Test accuracy          : 95.81% with top 7 features
Cross condition gap    : reduced from 0.046 to 0.018
Kendall correlation    : 0.85 to 0.95 across 10 folds
SOTA comparison        : best accuracy among all compared methods

## Key Observation

SP_mean_abs_deriv and SP_dc_component score high in both conditions 
and rank at the top. SP_spec_kurt scores well rotationally but poorly 
translationally and gets penalized to rank 9. This shows the harmonic 
mean is working correctly in identifying truly robust features.
