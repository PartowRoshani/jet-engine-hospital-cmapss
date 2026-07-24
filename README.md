# Jet Engine Hospital

## Predictive Maintenance with NASA C-MAPSS

Jet Engine Hospital is an end-to-end machine-learning system for turbofan
predictive maintenance.

The system combines:

- Remaining Useful Life regression;
- failure-risk classification for 10, 20, and 30 cycles;
- unsupervised anomaly detection;
- conformal prediction intervals;
- persistent early-warning rules;
- auditable maintenance actions: `CONTINUE`, `INSPECT`, and `STOP`.

## Project stages

| Stage | Dataset | Operating conditions | Fault modes |
|---|---|---:|---:|
| Foundation | FD001 | 1 | 1 |
| Multi-fault challenge | FD003 | 1 | 2 |

## Leakage-safe evaluation

The 100 run-to-failure training engines in each stage are split by engine ID:

- 70 training engines;
- 15 validation engines;
- 15 internal-test engines.

No engine appears in more than one split. All temporal features are causal,
and future rows cannot change features computed at an earlier cycle.

The internal-test split is evaluated once after model and threshold locking.

## FD001 foundation stage

FD001 implements the complete predictive-maintenance workflow for one
operating condition and one fault mode.

Final locked FD001 internal-test regression results:

| Metric | Value |
|---|---:|
| MAE | 24.092 |
| RMSE | 31.029 |
| R² | 0.771 |
| NASA Score | 136526.909 |

FD001 deployment artifact version: `1.0.2`.

## FD003 deployment configuration

### Regression

- Model: `Ridge — Temporal | alpha=1000`
- Feature configuration: `Temporal`
- Temporal feature count: `147`
- Prediction ceiling: `300.0`
- Safety offset: `0.0`
- Deployment conformal quantile: `122.663678`

### Failure-risk classification

| Horizon | Model | Calibration | Threshold |
|---:|---|---|---:|
| 10 | Weighted Logistic | Sigmoid | 0.10 |
| 20 | Weighted Logistic | Isotonic | 0.08 |
| 30 | Weighted Logistic | Isotonic | 0.08 |

Probabilities satisfy:

`P(H10) <= P(H20) <= P(H30)`

A persistent warning requires at least 2 alerts in the latest 3 cycles.

### Anomaly detection

- Model: Isolation Forest
- Healthy reference: first 30 cycles of all 100 FD003 training engines
- Healthy-reference rows: 3000
- Sensors: 16
- Threshold quantile: 0.99
- Deployment normalized threshold: 3.715377
- Role: advisory evidence only

### Maintenance policy

| Evidence | Action |
|---|---|
| Persistent 10-cycle failure risk | `STOP` |
| Persistent 20-cycle or 30-cycle risk | `INSPECT` |
| No persistent supervised risk | `CONTINUE` |
| Anomaly or conformal uncertainty | Advisory only |

Actions use no-downgrade hysteresis:

`CONTINUE < INSPECT < STOP`

## FD003 held-out results

### Regression

| Metric | Internal test |
|---|---:|
| MAE | 42.522 |
| RMSE | 60.061 |
| R² | 0.531 |
| Near-failure MAE | 11.871 |
| Conformal coverage | 0.950 |

### Classification

| Horizon | Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|---:|
| 10 | 0.649 | 0.988 | 0.784 | 0.962 |
| 20 | 0.643 | 1.000 | 0.783 | 0.957 |
| 30 | 0.711 | 0.998 | 0.830 | 0.956 |

### Integrated policy

| Metric | Value |
|---|---:|
| Exact action accuracy | 0.870 |
| Under-alert rate | 0.001 |
| Over-alert rate | 0.129 |
| STOP recall for RUL <= 10 | 0.982 |
| At-least-INSPECT recall for RUL <= 30 | 0.996 |
| Downgrade violations | 0 |

## Repository structure

```text
FinalProject/
├── app/
├── artifacts/
│   ├── fd001/
│   │   └── v1.0.2/
│   └── fd003/
│       └── v1.0.0/
├── data/
│   ├── raw/
│   └── splits/
├── huggingface_space_fd003/
├── notebooks/
│   ├── jet_engine_hospital_fd001.ipynb
│   └── jet_engine_hospital_fd003.ipynb
├── reports/
│   ├── figures/
│   ├── final/
│   └── tables/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

## Installation

On Windows Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the notebooks

```bash
jupyter lab
```

The two main notebooks are:

- `notebooks/jet_engine_hospital_fd001.ipynb`
- `notebooks/jet_engine_hospital_fd003.ipynb`

For the final reproducibility check, restart each notebook kernel and run
every cell from top to bottom.

## Run the FD003 Gradio application

```bash
cd huggingface_space_fd003
pip install -r requirements.txt
python app.py
```

## Versioned artifacts

FD001:

```text
artifacts/fd001/v1.0.2/
```

FD003:

```text
artifacts/fd003/v1.0.0/
```

FD003 compressed release:

```text
artifacts/fd003/jet_engine_hospital_fd003_v1.0.0.zip
```

## Limitations

FD003 contains two fault modes. Exact RUL regression is substantially harder
than in FD001, particularly during early life.

The classification and policy subsystems intentionally favour failure recall
over low inspection workload. Therefore, FD003 produces more early alerts
and false positives than FD001.

Anomaly and conformal uncertainty outputs are advisory and cannot independently
escalate the operational action.

This project is an educational predictive-maintenance system and must not be
used as the sole authority for real aviation maintenance.

## Author

Partow Roshani  
Computer Science — Shahid Beheshti University
