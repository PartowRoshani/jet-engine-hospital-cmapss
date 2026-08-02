---
title: Jet Engine Hospital
emoji: ✈️
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: Leakage-safe predictive maintenance for NASA C-MAPSS FD001, FD003, and FD004.
---

# Jet Engine Hospital

A unified Streamlit application for the NASA C-MAPSS turbofan prognostics project.

## Project stages

- **FD001 — Foundation:** one operating condition and one fault mode
- **FD003 — Stage 2:** one operating condition and two fault modes
- **FD004 — Bonus:** six operating conditions and two fault modes

Users can select a dataset and upload a complete chronological engine trajectory to obtain:

- Remaining Useful Life (RUL) prediction
- Prediction interval and uncertainty evidence
- Calibrated 10-, 20-, and 30-cycle failure probabilities
- Unsupervised anomaly evidence and persistence
- A final `CONTINUE`, `INSPECT`, or `STOP` maintenance recommendation
- An auditable explanation of the rule that triggered the recommendation

The application loads the frozen preprocessing pipelines, fitted models, calibrators, thresholds, conformal parameters, anomaly detectors, and policy metadata exported by the final notebook.

> Educational demonstration only. This application is not an aviation-certified maintenance system.
