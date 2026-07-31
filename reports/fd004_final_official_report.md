# FD004 Final Official Evaluation Report

## Evaluation Integrity

- Official results were not used for model selection.
- Development contracts were frozen before official evaluation.
- Official test rows: 41,214
- Official test engines: 248
- Artifact version: 1.0.0

## Standard Official Terminal Regression

- MAE: 25.631234
- RMSE: 34.665716
- R2: 0.595770
- NASA score: 261491.535369
- Mean bias: 1.547666
- Overprediction rate: 0.407258
- Conformal coverage: 0.995968

## Row-Level Regression Diagnostic

- MAE: 51.408233
- RMSE: 70.596593
- R2: 0.413310
- Conformal coverage: 0.887587

## Official Classification

### 10-Cycle Horizon

- Average precision: 0.304500
- ROC-AUC: 0.998750
- Precision: 0.209945
- Recall: 0.926829
- F1: 0.342342

### 20-Cycle Horizon

- Average precision: 0.676669
- ROC-AUC: 0.996928
- Precision: 0.361744
- Recall: 0.975936
- F1: 0.527838

### 30-Cycle Horizon

- Average precision: 0.766456
- ROC-AUC: 0.993865
- Precision: 0.350374
- Recall: 0.975694
- F1: 0.515596

## Official Anomaly Detection

- Average precision: 0.398235
- ROC-AUC: 0.967571
- Precision: 0.226025
- Recall: 0.886574
- F1: 0.360216

## Official Unified Policy

- Exact state accuracy: 0.897244
- Under-escalation rate: 0.000461
- Over-escalation rate: 0.102295
- Critical recall: 0.926829
- Warning-or-higher recall: 0.983957
- Watch-or-higher recall: 0.988426
- Normal specificity: 0.906419
- Mean state transitions per engine: 3.931452

## Final Operational Contract

- Regression model: constrained Extra Trees
- Regression safety offset: 14.0
- Deployment conformal quantile: 111.395435
- Classification horizons: 10, 20, and 30 cycles
- Classification thresholds: 0.03, 0.03, and 0.03
- Probability hierarchy: p10 <= p20 <= p30
- Anomaly threshold: -0.02362042
- Maximum anomaly-only state: Watch
- De-escalation confirmation cycles: 2
- Operational states: Normal < Watch < Warning < Critical

## Conclusion

The FD004 deployment system preserves high safety recall under multiple operating conditions while maintaining a very low official under-escalation rate.

The official terminal regression result is substantially stronger than the complete row-level diagnostic result. Classification and anomaly components therefore remain the primary operational risk signals, while regression and conformal intervals provide advisory remaining-life information.