# Results summary (generated)

## Crop recommendation (classification)

| Model | accuracy | macro_f1 | balanced_accuracy | cv_accuracy_mean | cv_accuracy_std | fit_seconds |
|---|---|---|---|---|---|---|
| Random Forest | 0.9705 | 0.9702 | 0.9705 | 0.9676 | 0.0069 | 2.4000 |
| XGBoost | 0.9523 | 0.9522 | 0.9523 | 0.9568 | 0.0116 | 6.6000 |
| Decision Tree | 0.9409 | 0.9404 | 0.9409 | 0.9250 | 0.0199 | 0.1000 |
| Logistic Regression | 0.9318 | 0.9311 | 0.9318 | 0.9330 | 0.0161 | 0.2000 |

## Yield forecast (regression)

| Model | r2_log | mae_log | rmse_t_per_ha | mae_t_per_ha | r2_t_per_ha | cv_r2_mean | cv_r2_std | fit_seconds |
|---|---|---|---|---|---|---|---|---|
| Random Forest | 0.9485 | 0.1266 | 3.3758 | 0.8794 | 0.9232 | 0.9329 | 0.0119 | 18.7000 |
| Decision Tree | 0.9224 | 0.1517 | 4.0078 | 1.0479 | 0.8917 | 0.9016 | 0.0136 | 2.6000 |
| Hist Gradient Boosting | 0.9187 | 0.1744 | 4.2493 | 1.2095 | 0.8783 | 0.8972 | 0.0127 | 5.9000 |
| XGBoost | 0.9141 | 0.1740 | 4.6454 | 1.2575 | 0.8545 | 0.9021 | 0.0146 | 3.9000 |
| Ridge (linear baseline) | 0.8324 | 0.2538 | 6.5884 | 1.9961 | 0.7074 | 0.8272 | 0.0027 | 0.9000 |
