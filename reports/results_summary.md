# Results summary (generated)

## Crop recommendation (classification)

| Model | accuracy | macro_f1 | balanced_accuracy | cv_accuracy_mean | cv_accuracy_std | fit_seconds |
|---|---|---|---|---|---|---|
| Random Forest | 0.9705 | 0.9702 | 0.9705 | 0.9676 | 0.0069 | 2.5000 |
| Extra Trees | 0.9682 | 0.9681 | 0.9682 | 0.9659 | 0.0074 | 2.1000 |
| XGBoost | 0.9523 | 0.9522 | 0.9523 | 0.9568 | 0.0116 | 7.5000 |
| Decision Tree | 0.9409 | 0.9404 | 0.9409 | 0.9250 | 0.0199 | 0.1000 |
| Logistic Regression | 0.9318 | 0.9311 | 0.9318 | 0.9330 | 0.0161 | 0.2000 |

## Yield forecast (regression)

| Model | r2_log | mae_log | rmse_t_per_ha | mae_t_per_ha | r2_t_per_ha | cv_r2_mean | cv_r2_std | temporal_r2_log | temporal_mae_t_per_ha | fit_seconds | final_conformal_r2_log | final_conformal_mae_t_per_ha |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9559 | 0.1142 | 3.3150 | 0.7754 | 0.9273 | 0.9439 | 0.0030 | 0.9136 | 1.1271 | 52.9000 |  |  |
| XGBoost (tuned) | 0.9395 | 0.1428 | 3.9779 | 0.9981 | 0.8954 | 0.9239 | nan | 0.9496 | 0.8810 | 48.7000 | 0.9372 | 1.0541 |
| Decision Tree | 0.9352 | 0.1366 | 3.7376 | 0.9198 | 0.9076 | 0.9179 | 0.0050 | 0.8866 | 1.2628 | 7.4000 |  |  |
| Hist Gradient Boosting (tuned) | 0.9341 | 0.1535 | 4.0038 | 1.0693 | 0.8940 | 0.9230 | nan | 0.9379 | 1.0071 | 25.3000 |  |  |
| XGBoost | 0.9161 | 0.1713 | 4.6261 | 1.2043 | 0.8585 | 0.9100 | 0.0062 | 0.9032 | 1.3217 | 9.7000 |  |  |
| Hist Gradient Boosting | 0.9152 | 0.1781 | 4.4412 | 1.2265 | 0.8696 | 0.9110 | 0.0049 | 0.9023 | 1.3211 | 12.5000 |  |  |
| Ridge (linear baseline) | 0.8132 | 0.2656 | 6.7589 | 2.0919 | 0.6979 | 0.8180 | 0.0015 | 0.8148 | 2.1884 | 2.3000 |  |  |
