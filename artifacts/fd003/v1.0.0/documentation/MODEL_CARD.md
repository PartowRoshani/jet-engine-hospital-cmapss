# FD003 Jet Engine Hospital Model Card

## Artifact identity

- Dataset: NASA C-MAPSS FD003
- Artifact version: 1.0.0
- Created: 2026-07-24T20:04:03.725637+00:00
- Training engines: 100
- Operating conditions: 1
- Fault modes: 2

## Intended use

This artifact provides maintenance decision support for FD003-style engine
trajectories. It produces:

- remaining useful life estimates;
- failure probabilities for 10, 20, and 30 cycles;
- anomaly evidence;
- conformal uncertainty intervals;
- CONTINUE, INSPECT, or STOP maintenance actions.

It is an educational predictive-maintenance artifact and must not be used as
the sole authority for real aviation maintenance decisions.

## Regression

- Model: Ridge — Temporal | alpha=1000
- Features: 147 causal temporal features
- Prediction ceiling: 300.0
- Safety offset: 0.0
- Deployment conformal quantile: 122.663678
- Conformal confidence: 0.95

## Classification

- Horizon 10: Weighted Logistic + Sigmoid, threshold 0.10
- Horizon 20: Weighted Logistic + Isotonic, threshold 0.08
- Horizon 30: Weighted Logistic + Isotonic, threshold 0.08
- Probability monotonicity: enabled
- Persistence rule: 2 of the latest 3 cycles

## Anomaly detection

- Model: Isolation Forest
- Sensors: 16
- Healthy reference: first 30 cycles of all 100 training engines
- Reference rows: 3000
- Threshold quantile: 0.99
- Normalized threshold: 3.715377
- Operational role: advisory only

## Maintenance policy

- Persistent 10-cycle risk: STOP
- Persistent 20-cycle or 30-cycle risk: INSPECT
- No persistent supervised risk: CONTINUE
- Anomaly evidence: advisory only
- Conformal uncertainty: advisory only
- Hysteresis: actions cannot downgrade within an engine trajectory

## Held-out FD003 internal-test performance

- Regression MAE: 42.522372 cycles
- Regression RMSE: 60.061392 cycles
- Regression R2: 0.531300
- Classification H10 recall: 0.987879
- Classification H20 recall: 1.000000
- Classification H30 recall: 0.997849
- Policy exact-action accuracy: 0.869829
- Policy under-alert rate: 0.001379
- Policy STOP recall: 0.981818

## Limitations

- Exact RUL regression is substantially less reliable on FD003 than FD001.
- Two degradation modes create heterogeneous trajectories.
- Conformal intervals are wide, especially outside the near-failure region.
- Cost-sensitive classification thresholds intentionally increase false alerts.
- The anomaly detector may signal degradation well before imminent failure.
- Input histories must be complete, chronologically ordered, and causally
  transformed with the same feature-engineering function.

## Reproducibility

The artifact includes:

- serialized estimators;
- complete feature schema;
- policy configuration;
- frozen evaluation summaries;
- final freeze audit;
- cryptographic SHA-256 hashes.
