# IDDFL - Wall Classification
INCREMENTAL DATA DECREMENTAL FEATURE LEARNING

## What this is about

In real radar deployments, data is not collected all at once. It keeps 
coming in over time. The problem is that SHAP does not have a way to 
update feature rankings as new data arrives. You would have to retrain 
and re-explain everything from scratch every time.

To solve this, we proposed IDDFL which stands for Incremental Data 
Decremental Feature Learning. The idea is simple - every day new data 
is added to all previous data, the model is retrained, SHAP importance 
is computed, and the weakest features are removed. This continues for 
20 days until a compact and accurate feature set is obtained.

## Dataset

We collected our own wall material dataset using the TI AWR1843 radar and DCA1000EVM capture card.

Classes : Transparent glass wall, Smooth glass board, Cement wall, 
          Metal door, Vinyl floor, Olefin carpet floor, Ceramic tile, 
          Concrete wall

Radar placement : 30 to 90 cm range, azimuth and elevation both 
                  at +/- 70 degrees

Data size : 8 classes, 20 experiments per class, 1024 frames 
            per experiment

## How the algorithm works

1. Start with 100 features on day 1
2. Every day, add new data to all previous data
3. Train LightGBM on the full cumulative dataset
4. Compute SHAP importance for all features
5. Remove 3 to 4 features with lowest importance
6. Repeat for 20 days
7. End with 30 features and deploy on edge

Feature dropping rule : any feature scoring less than or equal to 
10 percent of the maximum SHAP contribution score is removed

## Results

Starting features  : 100
Final features     : 30
Test accuracy      : 79.80%
F1 score           : 0.79
Inference time     : 3.47ms on Raspberry Pi 4B
SOTA comparison    : 13 percentage points better than next best method

## Key Observation

After around 11 to 12 days the learning saturates and accuracy stops 
increasing. This confirms that the model has seen enough data and 
further collection does not add much value. The top retained features 
are mostly from the frequency domain, which motivated us to use 
spectral features in the next two objectives.
