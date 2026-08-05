# Jet Engine Hospital

## Predictive Maintenance with NASA C-MAPSS

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-dashboard-red.svg)](https://streamlit.io/)
[![Datasets](https://img.shields.io/badge/C--MAPSS-FD001%20%7C%20FD003%20%7C%20FD004-green.svg)](#supported-c-mapss-datasets)

**Jet Engine Hospital** is an end-to-end machine-learning system for turbofan-engine predictive maintenance using the NASA C-MAPSS datasets.

The project estimates Remaining Useful Life (RUL), predicts near-term failure risk, detects anomalous engine behaviour, quantifies prediction uncertainty, and converts model outputs into auditable maintenance recommendations.

## Live dashboard

The unified Streamlit dashboard is available at:

**https://jet-engine-hospital-cmapss-egywtpbmnfnczliqitpnc4.streamlit.app/**

The dashboard supports the FD001, FD003, and FD004 pipelines from one interface.

## Main capabilities

- Remaining Useful Life regression
- Failure-risk classification for 10, 20, and 30-cycle horizons
- Unsupervised anomaly detection
- Conformal prediction intervals
- Causal temporal feature engineering
- Leakage-safe engine-level train, validation, and test splits
- Condition-aware processing for the multi-condition FD004 dataset
- Persistent early-warning rules and policy hysteresis
- Auditable maintenance decisions
- Standalone inference modules and serialized deployment artifacts
- Unified Streamlit deployment for FD001, FD003, and FD004
- Reproducibility, parity, and end-to-end inference checks

## Supported C-MAPSS datasets

| Stage | Dataset | Operating conditions | Fault modes | Main purpose |
|---|---|---:|---:|---|
| Foundation | FD001 | 1 | 1 | Complete baseline predictive-maintenance workflow |
| Multi-fault challenge | FD003 | 1 | 2 | Robust modelling under multiple fault modes |
| Multi-condition bonus | FD004 | Multiple | 2 | Condition-aware modelling under multiple operating regimes and fault modes |

## End-to-end workflow

1. Load and validate the NASA C-MAPSS raw trajectories.
2. Calculate engine-level RUL targets.
3. Split engines rather than individual rows to prevent leakage.
4. Analyse operating settings, sensors, sequence lengths, and degradation behaviour.
5. Build causal temporal and condition-aware features.
6. Train and validate RUL regression models.
7. Train calibrated classifiers for failure within 10, 20, and 30 cycles.
8. Train an Isolation Forest anomaly detector using healthy-reference behaviour.
9. Construct conformal prediction intervals for RUL uncertainty.
10. Lock thresholds, policies, feature schemas, and deployment contracts.
11. Evaluate the frozen system on held-out and official test data.
12. Serialize artifacts and verify notebook-to-inference parity.
13. Deploy all supported datasets through the unified Streamlit dashboard.

## Leakage-safe evaluation

Model development is performed with engine-level splitting. Rows from the same engine are never distributed across different development subsets.

For FD001 and FD003, the 100 run-to-failure training engines are divided into:

- 70 model-training engines
- 15 validation engines
- 15 internal-test engines

FD004 uses a stratified engine-level split based on sequence-length quartiles:

- 174 model-training engines
- 37 validation engines
- 38 internal-test engines

All temporal features are causal. Future cycles cannot change features calculated at an earlier cycle. The internal-test and official-test evaluations are performed only after the corresponding model and threshold contracts have been locked.

# Dataset results

## FD001 foundation stage

FD001 implements the complete predictive-maintenance workflow for one operating condition and one fault mode.

### Locked internal-test regression results

| Metric | Value |
|---|---:|
| MAE | 24.092 |
| RMSE | 31.029 |
| R² | 0.771 |
| NASA Score | 136526.909 |

FD001 deployment artifact version: `1.0.2`.

## FD003 multi-fault stage

FD003 extends the system to two fault modes while retaining one operating condition.

### Locked regression configuration

- Model: `Ridge — Temporal | alpha=1000`
- Feature configuration: temporal
- Temporal feature count: 147
- Prediction ceiling: 300.0
- Safety offset: 0.0
- Deployment conformal quantile: 122.663678

### Failure-risk classification

| Horizon | Model | Calibration | Threshold |
|---:|---|---|---:|
| 10 cycles | Weighted Logistic | Sigmoid | 0.10 |
| 20 cycles | Weighted Logistic | Isotonic | 0.08 |
| 30 cycles | Weighted Logistic | Isotonic | 0.08 |

The calibrated probabilities satisfy:

```text
P(H10) <= P(H20) <= P(H30)
```

A persistent warning requires at least two alerts within the latest three cycles.

### Anomaly detection

- Model: Isolation Forest
- Healthy reference: first 30 cycles of all 100 FD003 training engines
- Healthy-reference rows: 3000
- Sensors: 16
- Threshold quantile: 0.99
- Normalized deployment threshold: 3.715377
- Anomaly evidence is advisory and cannot independently escalate the maintenance action

### Maintenance policy

| Evidence | Action |
|---|---|
| Persistent 10-cycle failure risk | `STOP` |
| Persistent 20 or 30-cycle failure risk | `INSPECT` |
| No persistent supervised risk | `CONTINUE` |
| Anomaly or conformal uncertainty only | Advisory evidence |

Actions use no-downgrade hysteresis:

```text
CONTINUE < INSPECT < STOP
```

### FD003 held-out results

#### Regression

| Metric | Internal test |
|---|---:|
| MAE | 42.522 |
| RMSE | 60.061 |
| R² | 0.531 |
| Near-failure MAE | 11.871 |
| Conformal coverage | 0.950 |

#### Classification

| Horizon | Precision | Recall | F1 | PR-AUC |
|---:|---:|---:|---:|---:|
| 10 | 0.649 | 0.988 | 0.784 | 0.962 |
| 20 | 0.643 | 1.000 | 0.783 | 0.957 |
| 30 | 0.711 | 0.998 | 0.830 | 0.956 |

#### Integrated policy

| Metric | Value |
|---|---:|
| Exact action accuracy | 0.870 |
| Under-alert rate | 0.001 |
| Over-alert rate | 0.129 |
| STOP recall for RUL <= 10 | 0.982 |
| At-least-INSPECT recall for RUL <= 30 | 0.996 |
| Downgrade violations | 0 |

FD003 deployment artifact version: `1.0.0`.

## FD004 multi-condition and multi-fault stage

FD004 is the bonus extension of the project. It adds multiple operating conditions and two fault modes, condition-aware feature construction, a frozen standalone inference contract, and integration into the unified Streamlit application.

### FD004 deployment configuration

- Training rows: 61,249
- Training engines: 249
- Regression model: constrained `ExtraTreesRegressor`
- Regression estimators: 120
- Regression and classification features: 164
- Anomaly features: 153
- Classification horizons: 10, 20, and 30 cycles
- Classification thresholds: 0.03, 0.03, and 0.03
- Regression safety offset: 14.0
- Deployment conformal quantile: 111.395435
- Anomaly model: Isolation Forest with 500 estimators
- Deployment anomaly threshold: -0.02362042
- De-escalation confirmation: two cycles
- Artifact version: `1.0.0`

The classification probabilities satisfy:

```text
p10 <= p20 <= p30
```

### FD004 operational states

```text
Normal < Watch < Warning < Critical
```

The dashboard maps these states to operational actions:

| FD004 state | Dashboard action |
|---|---|
| `Normal` | `CONTINUE` |
| `Watch` | `INSPECT` |
| `Warning` | `INSPECT` |
| `Critical` | `STOP` |

Anomaly evidence alone is limited to the `Watch` state. Higher escalation requires supervised failure-risk evidence.

### FD004 official terminal regression

| Metric | Official test |
|---|---:|
| MAE | 25.631234 |
| RMSE | 34.665716 |
| R² | 0.595770 |
| NASA Score | 261491.535369 |
| Mean bias | 1.547666 |
| Conformal coverage | 0.995968 |

### FD004 official classification

| Horizon | Precision | Recall | F1 | ROC-AUC |
|---:|---:|---:|---:|---:|
| 10 cycles | 0.209945 | 0.926829 | 0.342342 | 0.998750 |
| 20 cycles | 0.361744 | 0.975936 | 0.527838 | 0.996928 |
| 30 cycles | 0.350374 | 0.975694 | 0.515596 | 0.993865 |

### FD004 official anomaly detection

| Metric | Value |
|---|---:|
| Average precision | 0.398235 |
| ROC-AUC | 0.967571 |
| Precision | 0.226025 |
| Recall | 0.886574 |
| F1 | 0.360216 |

### FD004 official unified policy

| Metric | Value |
|---|---:|
| Exact state accuracy | 0.897244 |
| Under-escalation rate | 0.000461 |
| Over-escalation rate | 0.102295 |
| Critical recall | 0.926829 |
| Warning-or-higher recall | 0.983957 |
| Watch-or-higher recall | 0.988426 |
| Normal specificity | 0.906419 |

### FD004 deployment parity

The serialized FD004 artifact passed the notebook-to-inference and raw-to-output parity checks:

- Model-layer parity passed
- Raw-to-feature parity passed
- Raw-to-output parity passed
- Maximum end-to-end numeric parity error: approximately `1.71e-13`
- Total end-to-end mismatches: `0`

The detailed official FD004 report is available at:

```text
reports/fd004_final_official_report.md
```

# Repository structure

```text
jet-engine-hospital-cmapss/
├── app/
├── artifacts/
│   ├── fd001/
│   │   └── v1.0.2/
│   ├── fd003/
│   │   └── v1.0.0/
│   ├── fd004_artifact.joblib
│   └── fd004_artifact_manifest.json
├── data/
│   ├── raw/
│   └── splits/
│       └── fd004_engine_splits.json
├── huggingface_space/
│   ├── artifacts/
│   │   ├── fd004_artifact.joblib
│   │   └── fd004_artifact_manifest.json
│   ├── app.py
│   ├── fd004_inference.py
│   ├── streamlit_app.py
│   └── requirements.txt
├── notebooks/
│   ├── jet_engine_hospital_fd001.ipynb
│   ├── jet_engine_hospital_fd003.ipynb
│   └── jet_engine_hospital_fd004_bonus.ipynb
├── reports/
│   ├── figures/
│   ├── final/
│   ├── tables/
│   └── fd004_final_official_report.md
├── src/
│   └── fd004_inference.py
├── tests/
├── README.md
└── requirements.txt
```

# Installation

## 1. Clone the repository

The complete project can be cloned from GitHub:

```bash
git clone https://github.com/PartowRoshani/jet-engine-hospital-cmapss.git
cd jet-engine-hospital-cmapss
```

The repository can also be downloaded as a ZIP file from the GitHub repository page.

## 2. Create a virtual environment

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Windows Git Bash

```bash
python -m venv .venv
source .venv/Scripts/activate
```

### macOS or Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## 3. Install project dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

# Run the notebooks

Start JupyterLab from the repository root:

```bash
jupyter lab
```

The main notebooks are:

- `notebooks/jet_engine_hospital_fd001.ipynb`
- `notebooks/jet_engine_hospital_fd003.ipynb`
- `notebooks/jet_engine_hospital_fd004_bonus.ipynb`

For the final reproducibility check:

1. Restart the notebook kernel.
2. Clear previous outputs when necessary.
3. Run every cell from top to bottom.
4. Confirm that the expected artifacts, reports, figures, and tables are regenerated without errors.

# Run the unified Streamlit dashboard locally

From the repository root:

```bash
cd huggingface_space
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The application provides dataset selection for:

- `FD001 — Foundation`
- `FD003 — Multi-fault`
- `FD004 — Multi-condition and multi-fault`

For each supported dataset, the dashboard displays the current engine condition, RUL prediction, uncertainty interval, near-term failure probabilities, anomaly evidence, maintenance state, and recommended operational action.

# Run tests

From the repository root:

```bash
pytest -q
```

# Versioned deployment artifacts

## FD001

```text
artifacts/fd001/v1.0.2/
```

## FD003

```text
artifacts/fd003/v1.0.0/
```

Compressed FD003 release:

```text
artifacts/fd003/jet_engine_hospital_fd003_v1.0.0.zip
```

## FD004

```text
artifacts/fd004_artifact.joblib
artifacts/fd004_artifact_manifest.json
src/fd004_inference.py
```

The Streamlit deployment contains its own FD004 copies under:

```text
huggingface_space/artifacts/
huggingface_space/fd004_inference.py
```

# Reproducibility and deployment safeguards

- Engine-level data splitting prevents trajectory leakage.
- Temporal features are causal.
- Validation decisions are frozen before internal-test and official-test evaluation.
- Deployment artifacts contain model, preprocessing, feature, threshold, uncertainty, anomaly, and policy contracts.
- The FD004 standalone inference module does not depend on notebook-defined functions.
- Artifact manifests record software versions, feature counts, model settings, checksums, and parity results.
- Hysteresis reduces unstable maintenance-state changes across adjacent cycles.

# Limitations

- FD003 and FD004 contain multiple fault modes, making exact RUL regression more difficult than FD001.
- FD004 additionally contains multiple operating conditions and requires condition-aware processing.
- The safety policy intentionally prioritizes failure recall and low under-escalation over a minimal inspection workload.
- This design may produce early alerts and false positives.
- Anomaly and uncertainty outputs are primarily advisory and cannot independently justify the highest operational state.
- The official terminal regression metric evaluates the final available cycle for each test engine and is not equivalent to the complete row-level diagnostic.
- This project is an educational predictive-maintenance system and must not be used as the sole authority for real aviation maintenance.

# Dataset acknowledgement

This project uses the NASA C-MAPSS turbofan-engine degradation datasets for educational machine-learning research and predictive-maintenance analysis.

## Course Instructor:

Dr. Hadi Farahani

## Author

 [Partow Roshani](https://github.com/PartowRoshani/)


## Date / Version

Summer 2026
Version 1.0

