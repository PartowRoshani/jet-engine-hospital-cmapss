from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path

import gradio as gr
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.errors import PerformanceWarning

from src.artifact_loader import load_fd001_artifacts
from src.feature_engineering import build_time_series_features
from src.inference import create_fd001_dashboard_view, run_fd001_inference


warnings.filterwarnings("ignore", category=PerformanceWarning)

APP_ROOT = Path(__file__).resolve().parent

FD001_BUNDLE = load_fd001_artifacts(
    project_root=APP_ROOT,
    verify_checksums=True,
)

FD003_ROOT = APP_ROOT / "artifacts" / "fd003" / "v1.0.0"
FD003_MODELS_DIR = FD003_ROOT / "models"
FD003_CONFIG_DIR = FD003_ROOT / "config"

with open(
    FD003_CONFIG_DIR / "feature_schema.json",
    encoding="utf-8",
) as file_handle:
    FD003_FEATURE_SCHEMA = json.load(file_handle)

with open(
    FD003_CONFIG_DIR / "policy_config.json",
    encoding="utf-8",
) as file_handle:
    FD003_POLICY_CONFIG = json.load(file_handle)

FD003_REGRESSION_MODEL = joblib.load(
    FD003_MODELS_DIR / "regression_model.joblib"
)

FD003_CLASSIFICATION_MODELS = {
    horizon: joblib.load(
        FD003_MODELS_DIR
        / f"classification_h{horizon}_base_model.joblib"
    )
    for horizon in (10, 20, 30)
}

FD003_CLASSIFICATION_CALIBRATORS = {}

for horizon in (10, 20, 30):
    calibrator_path = (
        FD003_MODELS_DIR
        / f"classification_h{horizon}_calibrator.joblib"
    )

    FD003_CLASSIFICATION_CALIBRATORS[horizon] = (
        joblib.load(calibrator_path)
        if calibrator_path.exists()
        else None
    )

FD003_ANOMALY_SCALER = joblib.load(
    FD003_MODELS_DIR / "anomaly_scaler.joblib"
)

FD003_ANOMALY_MODEL = joblib.load(
    FD003_MODELS_DIR / "anomaly_isolation_forest.joblib"
)

RAW_COLUMNS = [
    "engine_id",
    "cycle",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    *[
        f"sensor_{sensor_number}"
        for sensor_number in range(1, 22)
    ],
]


def safe_logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1.0 - 1e-6,
    )

    return np.log(clipped / (1.0 - clipped))


def apply_calibrator(
    calibrator,
    calibration_method: str,
    raw_probabilities: np.ndarray,
) -> np.ndarray:
    raw_probabilities = np.asarray(
        raw_probabilities,
        dtype=float,
    )

    if calibration_method == "Raw":
        calibrated = raw_probabilities

    elif calibration_method == "Sigmoid":
        if calibrator is None:
            raise RuntimeError(
                "The saved sigmoid calibrator is missing."
            )

        calibrated = calibrator.predict_proba(
            safe_logit(raw_probabilities).reshape(-1, 1)
        )[:, 1]

    elif calibration_method == "Isotonic":
        if calibrator is None:
            raise RuntimeError(
                "The saved isotonic calibrator is missing."
            )

        calibrated = calibrator.predict(
            raw_probabilities
        )

    else:
        raise ValueError(
            "Unsupported calibration method: "
            f"{calibration_method}"
        )

    return np.clip(calibrated, 0.0, 1.0)


def read_engine_file(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            "The uploaded file could not be found."
        )

    frame = None

    try:
        named_frame = pd.read_csv(path)
    except Exception:
        named_frame = pd.DataFrame()

    if set(RAW_COLUMNS).issubset(named_frame.columns):
        frame = named_frame[RAW_COLUMNS].copy()

    if frame is None:
        attempts = [
            {
                "sep": r"\s+",
                "header": None,
                "engine": "python",
            },
            {
                "sep": ",",
                "header": None,
                "engine": "python",
            },
        ]

        for read_arguments in attempts:
            try:
                candidate = pd.read_csv(
                    path,
                    **read_arguments,
                )
            except Exception:
                continue

            candidate = candidate.dropna(
                axis=1,
                how="all",
            )

            if candidate.shape[1] == 26:
                candidate.columns = RAW_COLUMNS
                frame = candidate
                break

    if frame is None:
        raise ValueError(
            "The file must contain the 26 standard "
            "NASA C-MAPSS columns."
        )

    for column_name in RAW_COLUMNS:
        frame[column_name] = pd.to_numeric(
            frame[column_name],
            errors="coerce",
        )

    if frame[RAW_COLUMNS].isna().any().any():
        raise ValueError(
            "The input contains missing or non-numeric values."
        )

    if not np.isfinite(
        frame[RAW_COLUMNS].to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "The input contains infinite values."
        )

    frame["engine_id"] = frame["engine_id"].astype(int)
    frame["cycle"] = frame["cycle"].astype(int)

    if frame[
        ["engine_id", "cycle"]
    ].duplicated().any():
        raise ValueError(
            "Duplicate engine-cycle rows were detected."
        )

    frame = (
        frame.sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    for engine_id, engine_frame in frame.groupby(
        "engine_id",
        sort=True,
    ):
        observed_cycles = engine_frame[
            "cycle"
        ].to_numpy(dtype=int)

        expected_cycles = np.arange(
            1,
            int(observed_cycles.max()) + 1,
        )

        if not np.array_equal(
            observed_cycles,
            expected_cycles,
        ):
            raise ValueError(
                f"Engine {engine_id} must contain a complete "
                "history beginning at cycle 1."
            )

    return frame


def apply_persistence(
    frame: pd.DataFrame,
    direct_alert_column: str,
    *,
    required_alerts: int,
    window: int,
) -> pd.Series:
    return (
        frame.groupby(
            "engine_id",
            sort=False,
        )[direct_alert_column]
        .transform(
            lambda alert_series: (
                alert_series.astype(int)
                .rolling(
                    window=window,
                    min_periods=required_alerts,
                )
                .sum()
                .ge(required_alerts)
            )
        )
        .fillna(False)
        .astype(bool)
    )


def create_trajectory_figure(
    results: pd.DataFrame,
    *,
    dataset_name: str,
):
    first_engine_id = int(
        results["engine_id"].iloc[0]
    )

    engine_frame = results.loc[
        results["engine_id"].eq(first_engine_id)
    ]

    figure, axis = plt.subplots(
        figsize=(10, 5),
        constrained_layout=True,
    )

    axis.plot(
        engine_frame["cycle"],
        engine_frame["RUL prediction"],
        label="Predicted RUL",
    )

    axis.fill_between(
        engine_frame["cycle"],
        engine_frame["RUL lower"],
        engine_frame["RUL upper"],
        alpha=0.2,
        label="95% conformal interval",
    )

    axis.set_title(
        f"{dataset_name} — Engine {first_engine_id}"
    )
    axis.set_xlabel("Operating cycle")
    axis.set_ylabel("Remaining useful life")
    axis.grid(alpha=0.25)
    axis.legend()

    return figure


def save_timeline(
    results: pd.DataFrame,
    dataset_name: str,
) -> str:
    file_descriptor, output_path = tempfile.mkstemp(
        prefix=f"{dataset_name.lower()}_predictions_",
        suffix=".csv",
    )

    os.close(file_descriptor)

    results.to_csv(
        output_path,
        index=False,
    )

    return output_path


def run_fd001_pipeline(
    raw_frame: pd.DataFrame,
):
    sensor_columns = FD001_BUNDLE.feature_schema[
        "anomaly_detection"
    ]["columns"]

    feature_frame = build_time_series_features(
        raw_frame,
        sensor_columns=sensor_columns,
    )

    inference_df = run_fd001_inference(
        bundle=FD001_BUNDLE,
        feature_df=feature_frame,
        model_scope="evaluation",
    )

    dashboard_df = create_fd001_dashboard_view(
        inference_df
    )

    plot_frame = inference_df.rename(
        columns={
            "RUL prediction": "RUL prediction",
        }
    )

    return (
        inference_df,
        dashboard_df,
        plot_frame,
    )


def run_fd003_pipeline(
    raw_frame: pd.DataFrame,
):
    regression_features = FD003_FEATURE_SCHEMA[
        "regression_feature_columns"
    ]

    classification_features = FD003_FEATURE_SCHEMA[
        "classification_feature_columns"
    ]

    anomaly_sensors = FD003_FEATURE_SCHEMA[
        "anomaly_sensor_columns"
    ]

    feature_frame = build_time_series_features(
        raw_frame,
        sensor_columns=anomaly_sensors,
    )

    required_features = set(
        regression_features
    ) | set(
        classification_features
    )

    missing_features = sorted(
        required_features
        - set(feature_frame.columns)
    )

    if missing_features:
        raise ValueError(
            "Feature engineering did not create all required "
            f"columns: {missing_features[:10]}"
        )

    results = feature_frame[
        ["engine_id", "cycle"]
    ].copy()

    regression_config = FD003_POLICY_CONFIG[
        "regression"
    ]

    raw_predictions = FD003_REGRESSION_MODEL.predict(
        feature_frame[regression_features]
    )

    prediction_ceiling = regression_config[
        "prediction_ceiling"
    ]

    if prediction_ceiling is None:
        ceiling_predictions = np.asarray(
            raw_predictions,
            dtype=float,
        )
    else:
        ceiling_predictions = np.minimum(
            np.asarray(raw_predictions, dtype=float),
            float(prediction_ceiling),
        )

    predicted_rul = np.clip(
        ceiling_predictions
        - float(regression_config["safety_offset"]),
        a_min=0.0,
        a_max=None,
    )

    conformal_quantile = float(
        regression_config[
            "deployment_conformal_quantile"
        ]
    )

    results["RUL prediction"] = predicted_rul
    results["RUL lower"] = np.maximum(
        0.0,
        predicted_rul - conformal_quantile,
    )
    results["RUL upper"] = (
        predicted_rul + conformal_quantile
    )

    classification_config = FD003_POLICY_CONFIG[
        "classification"
    ]

    probability_columns = []

    for horizon in (10, 20, 30):
        horizon_config = classification_config[
            "horizons"
        ][str(horizon)]

        raw_probabilities = (
            FD003_CLASSIFICATION_MODELS[horizon]
            .predict_proba(
                feature_frame[classification_features]
            )[:, 1]
        )

        calibrated_probabilities = apply_calibrator(
            FD003_CLASSIFICATION_CALIBRATORS[horizon],
            horizon_config["calibration_method"],
            raw_probabilities,
        )

        probability_columns.append(
            calibrated_probabilities
        )

    probability_matrix = np.maximum.accumulate(
        np.column_stack(probability_columns),
        axis=1,
    )

    for column_index, horizon in enumerate(
        (10, 20, 30)
    ):
        probability_column = f"probability_{horizon}"
        direct_column = f"risk_alert_{horizon}"
        persistent_column = (
            f"persistent_risk_{horizon}"
        )

        results[probability_column] = (
            probability_matrix[:, column_index]
        )

        threshold = float(
            classification_config[
                "horizons"
            ][str(horizon)]["threshold"]
        )

        results[direct_column] = (
            results[probability_column] >= threshold
        ).astype(int)

        results[persistent_column] = (
            apply_persistence(
                results,
                direct_column,
                required_alerts=int(
                    classification_config[
                        "persistence_required_alerts"
                    ]
                ),
                window=int(
                    classification_config[
                        "persistence_window"
                    ]
                ),
            )
            .astype(int)
        )

    anomaly_config = FD003_POLICY_CONFIG[
        "anomaly"
    ]

    anomaly_matrix = FD003_ANOMALY_SCALER.transform(
        feature_frame[anomaly_sensors]
    )

    raw_anomaly_scores = (
        -FD003_ANOMALY_MODEL.decision_function(
            anomaly_matrix
        )
    )

    normalized_anomaly_scores = (
        (
            raw_anomaly_scores
            - float(anomaly_config["reference_median"])
        )
        / float(anomaly_config["reference_scale"])
    )

    results["anomaly_score"] = (
        normalized_anomaly_scores
    )

    results["anomaly_alert"] = (
        results["anomaly_score"]
        >= float(
            anomaly_config["normalized_threshold"]
        )
    ).astype(int)

    results["persistent_anomaly"] = (
        apply_persistence(
            results,
            "anomaly_alert",
            required_alerts=int(
                anomaly_config[
                    "persistence_required_alerts"
                ]
            ),
            window=int(
                anomaly_config[
                    "persistence_window"
                ]
            ),
        )
        .astype(int)
    )

    results["Raw severity"] = np.select(
        [
            results["persistent_risk_10"].eq(1),
            (
                results["persistent_risk_20"].eq(1)
                | results["persistent_risk_30"].eq(1)
            ),
        ],
        [2, 1],
        default=0,
    ).astype(int)

    results["Policy severity"] = (
        results.groupby(
            "engine_id",
            sort=False,
        )["Raw severity"]
        .cummax()
        .astype(int)
    )

    severity_to_action = {
        0: "CONTINUE",
        1: "INSPECT",
        2: "STOP",
    }

    results["Action"] = results[
        "Policy severity"
    ].map(severity_to_action)

    results["Confidence"] = np.select(
        [
            results["Action"].eq("STOP"),
            results["Action"].eq("INSPECT"),
        ],
        [
            results["probability_10"],
            np.maximum(
                results["probability_20"],
                results["probability_30"],
            ),
        ],
        default=(
            1.0
            - results["probability_30"]
        ),
    )

    results["Trigger"] = np.select(
        [
            results["persistent_risk_10"].eq(1),
            (
                results["persistent_risk_20"].eq(1)
                | results["persistent_risk_30"].eq(1)
            ),
            (
                results["Policy severity"]
                > results["Raw severity"]
            ),
        ],
        [
            "Persistent 10-cycle risk",
            "Persistent 20/30-cycle risk",
            "Higher previous action retained",
        ],
        default="No persistent supervised risk",
    )

    results["Next review cycles"] = np.select(
        [
            results["Action"].eq("STOP"),
            results["Action"].eq("INSPECT"),
        ],
        [0, 5],
        default=10,
    ).astype(int)

    results["Signal disagreement"] = (
        (
            results[
                [
                    "persistent_risk_10",
                    "persistent_risk_20",
                    "persistent_risk_30",
                ]
            ].sum(axis=1)
            > 0
        )
        != (
            (
                results["persistent_anomaly"].eq(1)
            )
            | (
                results["RUL lower"] <= 30.0
            )
        )
    )

    latest_rows = (
        results.groupby(
            "engine_id",
            sort=True,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    dashboard_columns = [
        "engine_id",
        "cycle",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_score",
        "Action",
        "Confidence",
        "Trigger",
        "Next review cycles",
        "Signal disagreement",
    ]

    dashboard_df = latest_rows[
        dashboard_columns
    ].copy()

    return (
        results,
        dashboard_df,
        results,
    )


def run_combined_inference(
    dataset_name: str,
    uploaded_file,
):
    if uploaded_file is None:
        raise gr.Error(
            "Upload a CSV or C-MAPSS text file first."
        )

    try:
        raw_frame = read_engine_file(
            uploaded_file
        )

        if dataset_name.startswith("FD001"):
            (
                timeline_df,
                dashboard_df,
                plot_frame,
            ) = run_fd001_pipeline(raw_frame)

            short_dataset_name = "FD001"

        elif dataset_name.startswith("FD003"):
            (
                timeline_df,
                dashboard_df,
                plot_frame,
            ) = run_fd003_pipeline(raw_frame)

            short_dataset_name = "FD003"

        else:
            raise ValueError(
                "Unsupported dataset selection."
            )

        numeric_columns = dashboard_df.select_dtypes(
            include=[np.number]
        ).columns

        dashboard_df = dashboard_df.copy()

        dashboard_df[numeric_columns] = (
            dashboard_df[numeric_columns].round(4)
        )

        action_counts = (
            dashboard_df["Action"]
            .value_counts()
            .to_dict()
        )

        status_message = (
            "### Inference completed\n\n"
            f"- Dataset: **{short_dataset_name}**\n"
            f"- Engines: **{dashboard_df['engine_id'].nunique()}**\n"
            f"- Input rows: **{len(raw_frame)}**\n"
            f"- CONTINUE: **{action_counts.get('CONTINUE', 0)}**\n"
            f"- INSPECT: **{action_counts.get('INSPECT', 0)}**\n"
            f"- STOP: **{action_counts.get('STOP', 0)}**"
        )

        trajectory_figure = create_trajectory_figure(
            plot_frame,
            dataset_name=short_dataset_name,
        )

        output_path = save_timeline(
            timeline_df,
            short_dataset_name,
        )

        return (
            status_message,
            dashboard_df,
            trajectory_figure,
            output_path,
        )

    except Exception as error:
        raise gr.Error(str(error)) from error


with gr.Blocks(
    title="Jet Engine Hospital",
) as demo:
    gr.Markdown(
        """
# ✈️ Jet Engine Hospital

Unified predictive-maintenance application for NASA C-MAPSS.

Choose the project stage, upload a complete chronological
engine trajectory, and receive:

- Remaining Useful Life prediction;
- conformal uncertainty interval;
- 10-, 20-, and 30-cycle failure probabilities;
- anomaly evidence;
- `CONTINUE`, `INSPECT`, or `STOP` recommendation.

**Datasets**

- **FD001** — one operating condition and one fault mode;
- **FD003** — one operating condition and two fault modes.

The uploaded history for every engine must begin at cycle 1.

> Educational demonstration only.  
> Not an aviation-certified maintenance system.
"""
    )

    dataset_selector = gr.Radio(
        choices=[
            "FD001 — Foundation",
            "FD003 — Multi-fault",
        ],
        value="FD001 — Foundation",
        label="Dataset / project stage",
    )

    uploaded_file = gr.File(
        label="Engine trajectory file",
        file_types=[".csv", ".txt"],
        type="filepath",
    )

    run_button = gr.Button(
        "Run inference",
        variant="primary",
    )

    status_output = gr.Markdown()

    dashboard_output = gr.Dataframe(
        label="Latest maintenance status per engine",
        interactive=False,
    )

    trajectory_output = gr.Plot(
        label="RUL trajectory for the first uploaded engine",
    )

    download_output = gr.File(
        label="Download complete inference timeline",
    )

    run_button.click(
        fn=run_combined_inference,
        inputs=[
            dataset_selector,
            uploaded_file,
        ],
        outputs=[
            status_output,
            dashboard_output,
            trajectory_output,
            download_output,
        ],
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(
            os.environ.get("PORT", "7860")
        ),
        show_error=True,
    )
