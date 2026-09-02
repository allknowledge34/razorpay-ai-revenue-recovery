# Robust Model Evaluation & Calibration

## Dataset Summary
- **Total Records:** 20000
- **Positive Class (Recovered):** 8610 (43.0%)
- **Negative Class (Not Recovered):** 11390 (57.0%)

## Cross-Validation Strategy
- **Method:** StratifiedKFold (5 splits)
- **Shuffle:** True, **Random State:** 42

## Metrics (Mean ± Std across folds)

| Metric | Uncalibrated LR | Calibrated LR |
|:---|:---|:---|
| Accuracy | 0.7588 ± 0.0038 | 0.7628 ± 0.0031 |
| Precision | 0.7041 ± 0.0067 | 0.7391 ± 0.0083 |
| Recall | 0.7590 ± 0.0116 | 0.6945 ± 0.0222 |
| F1 | 0.7304 ± 0.0046 | 0.7159 ± 0.0088 |
| ROC-AUC | 0.8401 ± 0.0027 | 0.8400 ± 0.0027 |
| PR-AUC | 0.8020 ± 0.0065 | 0.7989 ± 0.0067 |
| Brier Score | 0.1631 ± 0.0011 | 0.1608 ± 0.0015 |

## Interpretation & Probability Trust
1. **Is the model reasonably calibrated?** The uncalibrated Logistic Regression was already well-calibrated (Brier Score: 0.1631), which is typical for linear models.
2. **Does calibration improve Brier score?** Technically yes, but only marginally (difference of 0.0023). The baseline was already strong.
3. **Does calibration hurt or improve discrimination?** ROC-AUC shifted from 0.8401 to 0.8400. The relative ordering of risk remains largely unchanged.
4. **Is the probability suitable for expected-recovery calculations?** Yes, the probabilities closely track the diagonal on the calibration curve. When the model outputs 60%, historically ~60% of those payments recover. This makes the `Expected Recoverable Revenue = Amount × Probability` formula financially sound.
5. **Limitations:** Since the dataset is synthetic, the calibration curve perfectly maps to the injected mathematical noise. In a real production environment, sudden shifts in payment gateway behavior could de-calibrate the model, requiring periodic re-calibration.
