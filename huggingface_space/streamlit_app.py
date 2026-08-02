from __future__ import annotations

import html
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# -----------------------------------------------------------------------------
# Paths and local imports
# -----------------------------------------------------------------------------

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent

for import_path in (APP_DIR, PROJECT_ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

from app import (  # noqa: E402
    FD001_BUNDLE,
    FD003_FEATURE_SCHEMA,
    FD003_POLICY_CONFIG,
    read_engine_file,
    run_fd001_pipeline,
    run_fd003_pipeline,
)

try:  # noqa: E402
    from fd004_inference import (
        create_fd004_terminal_summary,
        load_fd004_artifact,
        predict_fd004_trajectory,
    )
except ImportError:  # pragma: no cover - alternate package layout
    from src.fd004_inference import (
        create_fd004_terminal_summary,
        load_fd004_artifact,
        predict_fd004_trajectory,
    )


# -----------------------------------------------------------------------------
# Page configuration and visual constants
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
)

DATASET_OPTIONS = [
    "FD001 — Foundation",
    "FD003 — Multi-fault",
    "FD004 — Multi-condition and multi-fault",
]

STATUS_COLORS = {
    "CONTINUE": "#16a34a",
    "INSPECT": "#f59e0b",
    "STOP": "#dc2626",
}

PROBABILITY_COLORS = {
    10: "#dc2626",
    20: "#f59e0b",
    30: "#2563eb",
}

FD004_STATE_TO_ACTION = {
    "Normal": "CONTINUE",
    "Watch": "INSPECT",
    "Warning": "INSPECT",
    "Critical": "STOP",
}

FD004_STATE_TO_NEXT_REVIEW = {
    "Normal": 10,
    "Watch": 5,
    "Warning": 3,
    "Critical": 0,
}

FD004_ARTIFACT_CANDIDATES = [
    APP_DIR / "artifacts" / "fd004_artifact.joblib",
    PROJECT_ROOT / "artifacts" / "fd004_artifact.joblib",
    APP_DIR / "fd004_artifact.joblib",
]


st.markdown(
    """
    <style>
    .status-card {
        border-radius: 14px;
        padding: 1.15rem 1.35rem;
        color: white;
        margin: 0.35rem 0 1rem 0;
        box-shadow: 0 5px 16px rgba(15, 23, 42, 0.14);
    }
    .status-card h2 {
        color: white;
        margin: 0 0 0.35rem 0;
    }
    .status-card p {
        margin: 0.2rem 0;
    }
    .small-muted {
        color: #64748b;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# General helpers
# -----------------------------------------------------------------------------


def _mapping_value(mapping: dict, key: int | str, default: Any = None) -> Any:
    """Read integer-or-string keys from serialized configuration dictionaries."""
    if key in mapping:
        return mapping[key]
    text_key = str(key)
    if text_key in mapping:
        return mapping[text_key]
    return default


def _as_float(value: Any, default: float = float("nan")) -> float:
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return default
    return numeric_value if np.isfinite(numeric_value) else default


def _format_number(value: Any, digits: int = 3) -> str:
    numeric_value = _as_float(value)
    if np.isnan(numeric_value):
        return "N/A"
    return f"{numeric_value:.{digits}f}"


def _format_confidence(value: Any) -> str:
    if isinstance(value, str):
        return value
    numeric_value = _as_float(value)
    if np.isnan(numeric_value):
        return "N/A"
    return f"{numeric_value:.1%}"


def _format_boolean(value: Any) -> str:
    if pd.isna(value):
        return "N/A"
    return "YES" if bool(value) else "NO"


def _extract_rolling_windows(schema: dict[str, Any]) -> list[int]:
    """Infer rolling-window sizes from a saved feature schema."""
    schema_text = json.dumps(schema, default=str)
    matches = re.findall(r"_(?:mean|std)_(\d+)\b", schema_text)
    return sorted({int(match) for match in matches})


def _style_action(value: Any) -> str:
    color = STATUS_COLORS.get(str(value).upper())
    if color is None:
        return ""
    return f"background-color: {color}; color: white; font-weight: 700;"


def _rounded_frame(frame: pd.DataFrame, digits: int = 4) -> pd.DataFrame:
    result = frame.copy()
    numeric_columns = result.select_dtypes(include=[np.number]).columns
    result[numeric_columns] = result[numeric_columns].round(digits)
    return result


def _latest_status_from_timeline(timeline_df: pd.DataFrame) -> pd.DataFrame:
    """Create a current-status table without changing any model decision."""
    latest = (
        timeline_df.sort_values(["engine_id", "cycle"])
        .groupby("engine_id", sort=True)
        .tail(1)
        .reset_index(drop=True)
    )

    preferred_columns = [
        "engine_id",
        "cycle",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_percentile",
        "anomaly_score",
        "anomaly_severity",
        "Policy state",
        "Action",
        "Confidence",
        "Trigger",
        "Next review cycles",
        "Signal disagreement",
    ]
    available_columns = [
        column for column in preferred_columns if column in latest.columns
    ]
    return latest[available_columns].copy()


# -----------------------------------------------------------------------------
# FD004 artifact and pipeline
# -----------------------------------------------------------------------------


@st.cache_resource
def load_streamlit_fd004_artifact() -> tuple[dict[str, Any], Path]:
    """Locate and load the frozen FD004 deployment artifact."""
    for artifact_candidate in FD004_ARTIFACT_CANDIDATES:
        if artifact_candidate.exists():
            return load_fd004_artifact(artifact_candidate), artifact_candidate

    checked_paths = "\n".join(str(path) for path in FD004_ARTIFACT_CANDIDATES)
    raise FileNotFoundError(
        "The FD004 artifact could not be found. Checked paths:\n"
        f"{checked_paths}"
    )


def run_fd004_pipeline(
    raw_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete raw-to-policy FD004 deployment pipeline."""
    fd004_artifact, _ = load_streamlit_fd004_artifact()

    native_timeline = predict_fd004_trajectory(
        raw_trajectory=raw_frame,
        artifact_or_path=fd004_artifact,
        require_contiguous_cycles=True,
        return_feature_frame=False,
    )

    results = (
        native_timeline.sort_values(["engine_id", "cycle"])
        .reset_index(drop=True)
        .copy()
    )

    results["RUL prediction"] = results["locked_RUL_prediction"]
    results["RUL lower"] = results["RUL_lower"]
    results["RUL upper"] = results["RUL_upper"]
    results["Policy state"] = results["policy_state"]
    results["Action"] = results["policy_state"].map(FD004_STATE_TO_ACTION)

    if results["Action"].isna().any():
        unknown_states = sorted(
            results.loc[results["Action"].isna(), "policy_state"]
            .drop_duplicates()
            .astype(str)
            .tolist()
        )
        raise ValueError(f"Unknown FD004 policy states: {unknown_states}")

    results["Trigger"] = results["policy_reason"]
    results["Next review cycles"] = (
        results["policy_state"].map(FD004_STATE_TO_NEXT_REVIEW).astype(int)
    )

    supervised_alert_active = (
        results[["alert_10", "alert_20", "alert_30"]]
        .max(axis=1)
        .astype(bool)
    )
    results["Signal disagreement"] = (
        supervised_alert_active != results["anomaly_alert"].astype(bool)
    )

    results["Confidence"] = np.select(
        condlist=[
            results["policy_state"].eq("Critical"),
            results["policy_state"].eq("Warning"),
            results["policy_state"].eq("Watch")
            & results["alert_30"].eq(1),
            results["policy_state"].eq("Watch")
            & results["anomaly_alert"].eq(1),
        ],
        choicelist=[
            results["probability_10"],
            results["probability_20"],
            results["probability_30"],
            np.nan,
        ],
        default=1.0 - results["probability_30"],
    )

    # Validate that one terminal row per engine can be produced.
    create_fd004_terminal_summary(results)
    dashboard_df = _latest_status_from_timeline(results)
    return results, dashboard_df, results


# -----------------------------------------------------------------------------
# Artifact-driven dashboard contracts
# -----------------------------------------------------------------------------


def get_dashboard_contract(dataset: str) -> dict[str, Any]:
    """Read thresholds, persistence, labels, and metadata from artifacts."""
    if dataset == "FD001":
        classification = FD001_BUNDLE.classification_config
        anomaly = FD001_BUNDLE.anomaly_config
        manifest = FD001_BUNDLE.manifest
        schema = FD001_BUNDLE.feature_schema

        probability_thresholds = {
            horizon: float(
                _mapping_value(classification["horizons"], horizon)[
                    "threshold"
                ]
            )
            for horizon in (10, 20, 30)
        }
        supervised_persistence = {
            "required": int(classification["persistence"]["alerts_required"]),
            "window": int(classification["persistence"]["window_size"]),
        }
        anomaly_persistence = {
            "required": int(anomaly["persistence"]["alerts_required"]),
            "window": int(anomaly["persistence"]["window_size"]),
        }
        models = manifest.get("locked_models", {})
        training_scope = manifest.get("training_scope", {})

        metadata = {
            "Dataset": manifest.get("dataset", "NASA C-MAPSS FD001"),
            "Artifact/model version": manifest.get("artifact_version", "N/A"),
            "Last training / artifact creation (UTC)": manifest.get(
                "created_at_utc", "N/A"
            ),
            "Training engines": training_scope.get(
                "evaluation_models", "N/A"
            ),
            "Regression model": models.get("rul", "N/A"),
            "Classification model": models.get("classification", "N/A"),
            "Anomaly model": models.get("anomaly", "N/A"),
            "Feature windows": ", ".join(
                str(window) for window in _extract_rolling_windows(schema)
            )
            or "N/A",
        }

        return {
            "probability_thresholds": probability_thresholds,
            "supervised_persistence": supervised_persistence,
            "anomaly": {
                "column": "anomaly_percentile",
                "label": "Anomaly reference percentile",
                "threshold": float(anomaly["percentile_threshold"]),
            },
            "anomaly_persistence": anomaly_persistence,
            "metadata": metadata,
        }

    if dataset == "FD003":
        classification = FD003_POLICY_CONFIG["classification"]
        anomaly = FD003_POLICY_CONFIG["anomaly"]
        manifest_path = (
            APP_DIR / "artifacts" / "fd003" / "v1.0.0" / "manifest.json"
        )
        if not manifest_path.exists():
            manifest_path = (
                PROJECT_ROOT
                / "artifacts"
                / "fd003"
                / "v1.0.0"
                / "manifest.json"
            )
        with manifest_path.open(encoding="utf-8") as file_handle:
            manifest = json.load(file_handle)

        probability_thresholds = {
            horizon: float(
                _mapping_value(classification["horizons"], horizon)[
                    "threshold"
                ]
            )
            for horizon in (10, 20, 30)
        }
        supervised_persistence = {
            "required": int(classification["persistence_required_alerts"]),
            "window": int(classification["persistence_window"]),
        }
        anomaly_persistence = {
            "required": int(anomaly["persistence_required_alerts"]),
            "window": int(anomaly["persistence_window"]),
        }

        classification_models = sorted(
            {
                str(_mapping_value(manifest.get("classification", {}), horizon, {}).get("model", "N/A"))
                for horizon in (10, 20, 30)
            }
        )
        metadata = {
            "Dataset": manifest.get("dataset", "FD003"),
            "Artifact/model version": manifest.get("artifact_version", "N/A"),
            "Last training / artifact creation (UTC)": manifest.get(
                "created_utc", "N/A"
            ),
            "Training engines": manifest.get("training_engines", "N/A"),
            "Regression model": manifest.get("regression", {}).get(
                "configuration", "N/A"
            ),
            "Classification model": ", ".join(classification_models),
            "Anomaly model": manifest.get("anomaly", {}).get("model", "N/A"),
            "Feature windows": ", ".join(
                str(window)
                for window in _extract_rolling_windows(FD003_FEATURE_SCHEMA)
            )
            or "N/A",
        }

        return {
            "probability_thresholds": probability_thresholds,
            "supervised_persistence": supervised_persistence,
            "anomaly": {
                "column": "anomaly_score",
                "label": "Normalized anomaly score",
                "threshold": float(anomaly["normalized_threshold"]),
            },
            "anomaly_persistence": anomaly_persistence,
            "metadata": metadata,
        }

    if dataset == "FD004":
        artifact, _ = load_streamlit_fd004_artifact()
        artifact_metadata = artifact["artifact_metadata"]
        classification = artifact["classification_contract"]
        anomaly = artifact["anomaly_contract"]
        policy = artifact["policy_contract"]
        feature_engineering = artifact["feature_engineering"]
        regression = artifact["regression_contract"]

        probability_thresholds = {
            horizon: float(
                _mapping_value(classification["thresholds"], horizon)
            )
            for horizon in (10, 20, 30)
        }
        rolling_windows = feature_engineering.get("rolling_windows", [])
        metadata = {
            "Dataset": artifact_metadata.get("dataset", "NASA C-MAPSS FD004"),
            "Artifact/model version": artifact_metadata.get(
                "artifact_version", "N/A"
            ),
            "Last training / artifact creation (UTC)": artifact_metadata.get(
                "created_at_utc", "N/A"
            ),
            "Training engines": artifact_metadata.get("training_engines", "N/A"),
            "Regression model": regression.get(
                "model_name", regression.get("model_family", "N/A")
            ),
            "Classification model": "Calibrated logistic classifiers (10/20/30 cycles)",
            "Anomaly model": anomaly.get("model_family", "N/A"),
            "Feature windows": ", ".join(str(value) for value in rolling_windows)
            or "N/A",
            "De-escalation confirmation": (
                f"{int(policy['deescalation_confirmation_cycles'])} cycles"
            ),
        }

        return {
            "probability_thresholds": probability_thresholds,
            "supervised_persistence": None,
            "anomaly": {
                "column": "anomaly_severity",
                "label": "Anomaly severity",
                "threshold": float(anomaly["deployment_threshold"]),
            },
            "anomaly_persistence": {
                "required": int(anomaly["persistence_required"]),
                "window": int(anomaly["persistence_window"]),
            },
            "metadata": metadata,
            "deescalation_confirmation_cycles": int(
                policy["deescalation_confirmation_cycles"]
            ),
        }

    raise ValueError(f"Unsupported dataset: {dataset}")


# -----------------------------------------------------------------------------
# Plotting helpers
# -----------------------------------------------------------------------------


def create_rul_figure(
    frame: pd.DataFrame,
    selected_cycle: int,
    dataset: str,
    engine_id: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    cycles = pd.to_numeric(frame["cycle"]).to_numpy(dtype=float)
    prediction = pd.to_numeric(frame["RUL prediction"]).to_numpy(dtype=float)
    lower = pd.to_numeric(frame["RUL lower"]).to_numpy(dtype=float)
    upper = pd.to_numeric(frame["RUL upper"]).to_numpy(dtype=float)

    axis.plot(cycles, prediction, color="#2563eb", linewidth=2, label="Predicted RUL")
    axis.fill_between(
        cycles,
        lower,
        upper,
        color="#60a5fa",
        alpha=0.24,
        label="95% conformal interval",
    )
    axis.axvline(
        selected_cycle,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected cycle: {selected_cycle}",
    )
    selected = frame.loc[frame["cycle"].eq(selected_cycle)].iloc[0]
    axis.scatter(
        [selected_cycle],
        [_as_float(selected["RUL prediction"])],
        color="#111827",
        s=55,
        zorder=5,
    )
    axis.set_title(f"{dataset} — Engine {engine_id} RUL trajectory")
    axis.set_xlabel("Operating cycle")
    axis.set_ylabel("Remaining useful life (cycles)")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    return figure


def create_sensor_figure(
    frame: pd.DataFrame,
    sensor_name: str,
    selected_cycle: int,
    engine_id: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    cycles = pd.to_numeric(frame["cycle"]).to_numpy(dtype=float)
    sensor_values = pd.to_numeric(frame[sensor_name]).to_numpy(dtype=float)
    selected_value = _as_float(
        frame.loc[frame["cycle"].eq(selected_cycle), sensor_name].iloc[0]
    )

    axis.plot(cycles, sensor_values, color="#0f766e", linewidth=2, label=sensor_name)
    axis.axvline(
        selected_cycle,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected cycle: {selected_cycle}",
    )
    axis.scatter([selected_cycle], [selected_value], color="#111827", s=55, zorder=5)
    axis.set_title(f"Engine {engine_id} — {sensor_name} trajectory")
    axis.set_xlabel("Operating cycle")
    axis.set_ylabel(sensor_name)
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    return figure


def create_probability_figure(
    frame: pd.DataFrame,
    thresholds: dict[int, float],
    selected_cycle: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.7), constrained_layout=True)
    cycles = pd.to_numeric(frame["cycle"]).to_numpy(dtype=float)

    for horizon in (10, 20, 30):
        probability_column = f"probability_{horizon}"
        if probability_column not in frame.columns:
            continue
        color = PROBABILITY_COLORS[horizon]
        probabilities = pd.to_numeric(frame[probability_column]).to_numpy(dtype=float)
        axis.plot(
            cycles,
            probabilities,
            color=color,
            linewidth=2,
            label=f"P(failure within {horizon} cycles)",
        )
        axis.axhline(
            thresholds[horizon],
            color=color,
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label=f"P{horizon} threshold = {thresholds[horizon]:.3f}",
        )

    axis.axvline(
        selected_cycle,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected cycle: {selected_cycle}",
    )
    axis.set_ylim(-0.02, 1.02)
    axis.set_xlabel("Operating cycle")
    axis.set_ylabel("Calibrated probability")
    axis.set_title("Failure-probability trajectories and locked thresholds")
    axis.grid(alpha=0.25)
    axis.legend(loc="best", ncol=2, fontsize=8)
    return figure


def create_anomaly_figure(
    frame: pd.DataFrame,
    anomaly_column: str,
    anomaly_label: str,
    threshold: float,
    selected_cycle: int,
) -> plt.Figure:
    figure, axis = plt.subplots(figsize=(10, 4.2), constrained_layout=True)
    cycles = pd.to_numeric(frame["cycle"]).to_numpy(dtype=float)
    anomaly_values = pd.to_numeric(frame[anomaly_column]).to_numpy(dtype=float)
    selected_value = _as_float(
        frame.loc[frame["cycle"].eq(selected_cycle), anomaly_column].iloc[0]
    )

    axis.plot(cycles, anomaly_values, color="#7c3aed", linewidth=2, label=anomaly_label)
    axis.axhline(
        threshold,
        color="#dc2626",
        linestyle=":",
        linewidth=1.8,
        label=f"Threshold = {threshold:.4f}",
    )
    axis.axvline(
        selected_cycle,
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label=f"Selected cycle: {selected_cycle}",
    )
    axis.scatter([selected_cycle], [selected_value], color="#111827", s=55, zorder=5)
    axis.set_xlabel("Operating cycle")
    axis.set_ylabel(anomaly_label)
    axis.set_title("Anomaly evidence and locked threshold")
    axis.grid(alpha=0.25)
    axis.legend(loc="best")
    return figure


# -----------------------------------------------------------------------------
# Evidence and persistence tables
# -----------------------------------------------------------------------------


def _recent_alert_count(
    frame: pd.DataFrame,
    alert_column: str,
    selected_cycle: int,
    window: int,
) -> int | None:
    if alert_column not in frame.columns:
        return None
    history = frame.loc[frame["cycle"].le(selected_cycle)].tail(window)
    return int(pd.to_numeric(history[alert_column], errors="coerce").fillna(0).sum())


def create_probability_evidence_table(
    selected_row: pd.Series,
    engine_frame: pd.DataFrame,
    selected_cycle: int,
    dataset: str,
    contract: dict[str, Any],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    persistence = contract.get("supervised_persistence")

    for horizon in (10, 20, 30):
        probability = _as_float(selected_row.get(f"probability_{horizon}"))
        threshold = float(contract["probability_thresholds"][horizon])
        alert_column = (
            f"alert_{horizon}" if dataset == "FD004" else f"risk_alert_{horizon}"
        )
        persistent_column = f"persistent_risk_{horizon}"
        direct_alert = bool(selected_row.get(alert_column, probability >= threshold))

        if persistence is not None:
            recent_count = _recent_alert_count(
                engine_frame,
                alert_column,
                selected_cycle,
                int(persistence["window"]),
            )
            persistent_active = bool(
                selected_row.get(
                    persistent_column,
                    recent_count is not None
                    and recent_count >= int(persistence["required"]),
                )
            )
            persistence_text = (
                f"{recent_count} of last {int(persistence['window'])}; "
                f"requires {int(persistence['required'])}"
                if recent_count is not None
                else "Unavailable"
            )
            active_text = "ACTIVE" if persistent_active else "INACTIVE"
        else:
            persistence_text = "Direct locked alert (no persistence gate)"
            active_text = "ACTIVE" if direct_alert else "INACTIVE"

        rows.append(
            {
                "Signal": f"P{horizon}",
                "Probability": probability,
                "Threshold": threshold,
                "Margin": probability - threshold,
                "Direct alert": "ACTIVE" if direct_alert else "INACTIVE",
                "Persistence evidence": persistence_text,
                "Policy signal": active_text,
            }
        )

    return pd.DataFrame(rows)


def create_anomaly_evidence_table(
    selected_row: pd.Series,
    engine_frame: pd.DataFrame,
    selected_cycle: int,
    dataset: str,
    contract: dict[str, Any],
) -> pd.DataFrame:
    anomaly_contract = contract["anomaly"]
    persistence = contract["anomaly_persistence"]
    score = _as_float(selected_row.get(anomaly_contract["column"]))
    threshold = float(anomaly_contract["threshold"])
    direct_alert = bool(selected_row.get("anomaly_alert", score >= threshold))
    recent_count = _recent_alert_count(
        engine_frame,
        "anomaly_alert",
        selected_cycle,
        int(persistence["window"]),
    )
    persistent_active = bool(
        selected_row.get(
            "persistent_anomaly",
            recent_count is not None
            and recent_count >= int(persistence["required"]),
        )
    )

    if recent_count is None:
        recent_text = "Unavailable"
    else:
        recent_text = (
            f"{recent_count} of last {int(persistence['window'])}; "
            f"requires {int(persistence['required'])}"
        )

    policy_role = (
        "Diagnostic only; FD004 policy uses its frozen direct anomaly alert"
        if dataset == "FD004"
        else "Advisory evidence"
    )

    return pd.DataFrame(
        [
            {
                "Signal": anomaly_contract["label"],
                "Score": score,
                "Threshold": threshold,
                "Margin": score - threshold,
                "Direct alert": "ACTIVE" if direct_alert else "INACTIVE",
                "Recent persistence": recent_text,
                "Persistent evidence": "ACTIVE" if persistent_active else "INACTIVE",
                "Policy role": policy_role,
            }
        ]
    )


# -----------------------------------------------------------------------------
# Inference execution
# -----------------------------------------------------------------------------


def execute_inference(uploaded_file: Any, dataset_label: str) -> dict[str, Any]:
    suffix = Path(uploaded_file.name).suffix or ".csv"
    temporary_path: str | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = temporary_file.name

        raw_frame = read_engine_file(temporary_path)

        if dataset_label.startswith("FD001"):
            timeline_df, dashboard_df, plot_frame = run_fd001_pipeline(raw_frame)
            dataset = "FD001"
        elif dataset_label.startswith("FD003"):
            timeline_df, dashboard_df, plot_frame = run_fd003_pipeline(raw_frame)
            dataset = "FD003"
        elif dataset_label.startswith("FD004"):
            timeline_df, dashboard_df, plot_frame = run_fd004_pipeline(raw_frame)
            dataset = "FD004"
        else:  # pragma: no cover - guarded by radio options
            raise ValueError("Unsupported dataset selection.")

        return {
            "dataset": dataset,
            "dataset_label": dataset_label,
            "source_filename": uploaded_file.name,
            "raw_frame": raw_frame,
            "timeline_df": timeline_df,
            "dashboard_df": dashboard_df,
            "plot_frame": plot_frame,
        }
    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)


# -----------------------------------------------------------------------------
# Header and controls
# -----------------------------------------------------------------------------

st.title("✈️ Jet Engine Hospital")
st.caption(
    "Unified predictive-maintenance dashboard for NASA C-MAPSS "
    "FD001, FD003, and FD004."
)

with st.expander("Input requirements and dashboard scope", expanded=False):
    st.markdown(
        """
        Upload a complete chronological engine trajectory containing the 26
        standard NASA C-MAPSS columns. Each engine history must begin at cycle 1.

        The dashboard reports RUL and its conformal interval, calibrated
        10/20/30-cycle failure probabilities, anomaly evidence, persistence,
        and the frozen `CONTINUE` / `INSPECT` / `STOP` maintenance decision.

        > Educational demonstration only. Not an aviation-certified maintenance system.
        """
    )

control_col_1, control_col_2 = st.columns([1.35, 1.65])

with control_col_1:
    selected_dataset_label = st.radio(
        "Dataset / project stage",
        options=DATASET_OPTIONS,
        key="dataset_selector",
    )

with control_col_2:
    uploaded_file = st.file_uploader(
        "Engine trajectory file",
        type=["csv", "txt"],
        key="trajectory_uploader",
    )
    run_button = st.button(
        "Run inference",
        type="primary",
        width="stretch",
    )

if run_button:
    st.session_state.pop("inference_result", None)
    if uploaded_file is None:
        st.error("Upload a CSV or C-MAPSS text file first.")
    else:
        try:
            with st.spinner("Running the frozen inference pipeline..."):
                st.session_state["inference_result"] = execute_inference(
                    uploaded_file,
                    selected_dataset_label,
                )
            st.success("Inference completed successfully.")
        except Exception as error:
            st.exception(error)


# -----------------------------------------------------------------------------
# Persistent results display
# -----------------------------------------------------------------------------

result = st.session_state.get("inference_result")

if result is None:
    st.info("Choose a dataset, upload a trajectory file, and click **Run inference**.")
    st.stop()

dataset = result["dataset"]
timeline_df = result["timeline_df"]
dashboard_df = result["dashboard_df"]
plot_frame = result["plot_frame"]
raw_frame = result["raw_frame"]
contract = get_dashboard_contract(dataset)

if selected_dataset_label != result["dataset_label"]:
    st.warning(
        f"The displayed results belong to {result['dataset_label']}. "
        "Click Run inference to analyze the newly selected dataset."
    )

action_counts = dashboard_df["Action"].value_counts().to_dict()
summary_columns = st.columns(5)
summary_columns[0].metric("Dataset", dataset)
summary_columns[1].metric("Engines", int(dashboard_df["engine_id"].nunique()))
summary_columns[2].metric("CONTINUE", int(action_counts.get("CONTINUE", 0)))
summary_columns[3].metric("INSPECT", int(action_counts.get("INSPECT", 0)))
summary_columns[4].metric("STOP", int(action_counts.get("STOP", 0)))

st.subheader("Latest maintenance status per engine")
rounded_dashboard = _rounded_frame(dashboard_df)
if "Action" in rounded_dashboard.columns:
    dashboard_styler = rounded_dashboard.style.map(_style_action, subset=["Action"])
    st.dataframe(dashboard_styler, width="stretch", hide_index=True)
else:
    st.dataframe(rounded_dashboard, width="stretch", hide_index=True)


# -----------------------------------------------------------------------------
# Engine and cycle selection
# -----------------------------------------------------------------------------

st.subheader("Cycle-level engine evidence")
engine_options = sorted(
    pd.to_numeric(plot_frame["engine_id"], errors="raise").astype(int).unique().tolist()
)

selector_col_1, selector_col_2 = st.columns([1, 2])
with selector_col_1:
    selected_engine = st.selectbox(
        "Select engine",
        options=engine_options,
        key=f"engine_selector_{dataset}",
    )

selected_plot_frame = (
    plot_frame.loc[plot_frame["engine_id"].eq(selected_engine)]
    .sort_values("cycle")
    .reset_index(drop=True)
)
selected_raw_frame = (
    raw_frame.loc[raw_frame["engine_id"].eq(selected_engine)]
    .sort_values("cycle")
    .reset_index(drop=True)
)

cycle_options = sorted(
    pd.to_numeric(selected_plot_frame["cycle"], errors="raise")
    .astype(int)
    .unique()
    .tolist()
)

with selector_col_2:
    selected_cycle = st.select_slider(
        "Select cycle",
        options=cycle_options,
        value=cycle_options[-1],
        key=f"cycle_selector_{dataset}_{selected_engine}",
    )

selected_row = selected_plot_frame.loc[
    selected_plot_frame["cycle"].eq(selected_cycle)
].iloc[0]


# -----------------------------------------------------------------------------
# Decision card and selected-cycle metrics
# -----------------------------------------------------------------------------

action = str(selected_row.get("Action", "UNKNOWN")).upper()
status_color = STATUS_COLORS.get(action, "#475569")
trigger = html.escape(str(selected_row.get("Trigger", "N/A")))
confidence = html.escape(_format_confidence(selected_row.get("Confidence")))
disagreement = html.escape(
    _format_boolean(selected_row.get("Signal disagreement", np.nan))
)
next_review = selected_row.get("Next review cycles", "N/A")
policy_state = selected_row.get("Policy state", selected_row.get("policy_state", None))
policy_state_text = (
    f"<p><strong>Policy state:</strong> {html.escape(str(policy_state))}</p>"
    if policy_state is not None and not pd.isna(policy_state)
    else ""
)

st.markdown(
    f"""
    <div class="status-card" style="background:{status_color};">
        <h2>{html.escape(action)}</h2>
        {policy_state_text}
        <p><strong>Trigger:</strong> {trigger}</p>
        <p><strong>Confidence:</strong> {confidence}</p>
        <p><strong>Signal disagreement:</strong> {disagreement}</p>
        <p><strong>Next review:</strong> {html.escape(str(next_review))} cycles</p>
    </div>
    """,
    unsafe_allow_html=True,
)

rul_prediction = _as_float(selected_row.get("RUL prediction"))
rul_lower = _as_float(selected_row.get("RUL lower"))
rul_upper = _as_float(selected_row.get("RUL upper"))

rul_columns = st.columns(3)
rul_columns[0].metric("Predicted RUL", f"{_format_number(rul_prediction, 1)} cycles")
rul_columns[1].metric("95% lower bound", f"{_format_number(rul_lower, 1)} cycles")
rul_columns[2].metric("95% upper bound", f"{_format_number(rul_upper, 1)} cycles")

probability_columns = st.columns(3)
for index, horizon in enumerate((10, 20, 30)):
    probability = _as_float(selected_row.get(f"probability_{horizon}"))
    threshold = contract["probability_thresholds"][horizon]
    margin = probability - threshold
    probability_columns[index].metric(
        f"Failure within {horizon} cycles",
        f"{probability:.1%}" if not np.isnan(probability) else "N/A",
        delta=f"{margin:+.3f} vs threshold {threshold:.3f}",
        delta_color="inverse",
    )

anomaly_contract = contract["anomaly"]
anomaly_value = _as_float(selected_row.get(anomaly_contract["column"]))
anomaly_margin = anomaly_value - float(anomaly_contract["threshold"])
anomaly_alert = bool(
    selected_row.get(
        "anomaly_alert",
        anomaly_value >= float(anomaly_contract["threshold"]),
    )
)

anomaly_columns = st.columns(3)
anomaly_columns[0].metric(anomaly_contract["label"], _format_number(anomaly_value, 4))
anomaly_columns[1].metric(
    "Anomaly threshold", _format_number(anomaly_contract["threshold"], 4)
)
anomaly_columns[2].metric(
    "Anomaly alert",
    "ACTIVE" if anomaly_alert else "INACTIVE",
    delta=f"Margin {anomaly_margin:+.4f}",
    delta_color="inverse",
)


# -----------------------------------------------------------------------------
# Visual evidence
# -----------------------------------------------------------------------------

rul_figure = create_rul_figure(
    selected_plot_frame,
    selected_cycle,
    dataset,
    selected_engine,
)
st.pyplot(rul_figure, width="stretch")
plt.close(rul_figure)

sensor_options = sorted(
    [column for column in selected_raw_frame.columns if column.startswith("sensor_")],
    key=lambda name: int(name.split("_")[-1]),
)
default_sensor_index = sensor_options.index("sensor_2") if "sensor_2" in sensor_options else 0
selected_sensor = st.selectbox(
    "Sensor / health feature",
    options=sensor_options,
    index=default_sensor_index,
    key=f"sensor_selector_{dataset}_{selected_engine}",
)

sensor_figure = create_sensor_figure(
    selected_raw_frame,
    selected_sensor,
    selected_cycle,
    selected_engine,
)
st.pyplot(sensor_figure, width="stretch")
plt.close(sensor_figure)

chart_col_1, chart_col_2 = st.columns(2)
with chart_col_1:
    probability_figure = create_probability_figure(
        selected_plot_frame,
        contract["probability_thresholds"],
        selected_cycle,
    )
    st.pyplot(probability_figure, width="stretch")
    plt.close(probability_figure)

with chart_col_2:
    anomaly_figure = create_anomaly_figure(
        selected_plot_frame,
        anomaly_contract["column"],
        anomaly_contract["label"],
        float(anomaly_contract["threshold"]),
        selected_cycle,
    )
    st.pyplot(anomaly_figure, width="stretch")
    plt.close(anomaly_figure)


# -----------------------------------------------------------------------------
# Thresholds, persistence, and decision history
# -----------------------------------------------------------------------------

st.subheader("Threshold and persistence evidence")
probability_evidence = create_probability_evidence_table(
    selected_row,
    selected_plot_frame,
    selected_cycle,
    dataset,
    contract,
)
st.dataframe(
    _rounded_frame(probability_evidence),
    width="stretch",
    hide_index=True,
)

anomaly_evidence = create_anomaly_evidence_table(
    selected_row,
    selected_plot_frame,
    selected_cycle,
    dataset,
    contract,
)
st.dataframe(
    _rounded_frame(anomaly_evidence),
    width="stretch",
    hide_index=True,
)

if dataset == "FD004":
    deescalation_run_length = selected_row.get("deescalation_run_length", "N/A")
    confirmation_cycles = contract["deescalation_confirmation_cycles"]
    st.info(
        "FD004 de-escalation confirmation: "
        f"{confirmation_cycles} consecutive lower-state cycles. "
        f"Current run length at cycle {selected_cycle}: {deescalation_run_length}. "
        "The recent anomaly-persistence row is diagnostic; the displayed policy "
        "decision is read directly from the frozen FD004 pipeline."
    )

with st.expander("Decision history through the selected cycle", expanded=False):
    decision_columns = [
        "cycle",
        "Action",
        "Policy state",
        "policy_state",
        "Trigger",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_alert",
        "persistent_anomaly",
        "persistent_risk_10",
        "persistent_risk_20",
        "persistent_risk_30",
        "deescalation_run_length",
        "Next review cycles",
    ]
    available_decision_columns = [
        column for column in decision_columns if column in selected_plot_frame.columns
    ]
    decision_history = selected_plot_frame.loc[
        selected_plot_frame["cycle"].le(selected_cycle),
        available_decision_columns,
    ].tail(50)
    st.dataframe(
        _rounded_frame(decision_history),
        width="stretch",
        hide_index=True,
    )


# -----------------------------------------------------------------------------
# Metadata and download
# -----------------------------------------------------------------------------

with st.expander("Model and dataset metadata", expanded=True):
    metadata_frame = pd.DataFrame(
        {
            "Field": list(contract["metadata"].keys()),
            "Value": [str(value) for value in contract["metadata"].values()],
        }
    )
    st.dataframe(metadata_frame, width="stretch", hide_index=True)

csv_bytes = timeline_df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download complete inference timeline",
    data=csv_bytes,
    file_name=f"{dataset.lower()}_predictions.csv",
    mime="text/csv",
    width="stretch",
)