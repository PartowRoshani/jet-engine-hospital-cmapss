from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.artifact_loader import (
    FD001ArtifactBundle,
    create_fd001_rul_intervals,
    predict_fd001_anomaly,
    predict_fd001_failure_probabilities,
    predict_fd001_rul,
)
from src.decision_policy import (
    add_persistent_signal,
    enforce_action_hysteresis,
)


ACTION_LEVELS = {
    "CONTINUE": 0,
    "INSPECT": 1,
    "STOP": 2,
}


def validate_fd001_inference_frame(
    data: pd.DataFrame,
) -> None:
    """
    Validate engine identity and ordering columns before
    running the FD001 inference pipeline.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError(
            "Inference DataFrame is empty."
        )

    required_columns = {
        "engine_id",
        "cycle",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing identity columns: "
            f"{sorted(missing_columns)}"
        )

    if data[
        [
            "engine_id",
            "cycle",
        ]
    ].isna().any().any():
        raise ValueError(
            "engine_id and cycle cannot contain "
            "missing values."
        )

    if data.duplicated(
        subset=[
            "engine_id",
            "cycle",
        ]
    ).any():
        duplicate_rows = data.loc[
            data.duplicated(
                subset=[
                    "engine_id",
                    "cycle",
                ],
                keep=False,
            ),
            [
                "engine_id",
                "cycle",
            ],
        ]

        raise ValueError(
            "Duplicate engine-cycle keys detected: "
            f"{duplicate_rows.head().to_dict('records')}"
        )

    cycle_values = pd.to_numeric(
        data["cycle"],
        errors="coerce",
    )

    if cycle_values.isna().any():
        raise ValueError(
            "cycle must contain numeric values."
        )

    if (cycle_values <= 0).any():
        raise ValueError(
            "cycle values must be positive."
        )


def _read_classification_threshold(
    bundle: FD001ArtifactBundle,
    horizon: int,
) -> float:
    """
    Read one locked classification threshold.
    """
    horizons = bundle.classification_config[
        "horizons"
    ]

    horizon_key = str(
        horizon
    )

    if horizon_key not in horizons:
        raise KeyError(
            "Classification horizon missing from "
            f"configuration: {horizon}"
        )

    return float(
        horizons[
            horizon_key
        ][
            "threshold"
        ]
    )


def _add_persistent_inference_signals(
    data: pd.DataFrame,
    alerts_required: int,
    window_size: int,
) -> pd.DataFrame:
    """
    Add persistent supervised and anomaly alerts.
    """
    result = data.copy()

    signal_pairs = [
        (
            "risk_alert_10",
            "persistent_risk_10",
        ),
        (
            "risk_alert_20",
            "persistent_risk_20",
        ),
        (
            "risk_alert_30",
            "persistent_risk_30",
        ),
        (
            "anomaly_alert",
            "persistent_anomaly",
        ),
    ]

    for signal_column, output_column in (
        signal_pairs
    ):
        result = add_persistent_signal(
            data=result,
            signal_column=signal_column,
            output_column=output_column,
            alerts_required=alerts_required,
            window_size=window_size,
        )

    return result


def _assign_locked_supervised_action(
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply the locked supervised-only maintenance rule.

    Persistent 10-cycle risk:
        STOP

    Persistent 20-cycle or 30-cycle risk:
        INSPECT

    Anomaly and conformal interval:
        advisory evidence only
    """
    result = evidence_df.copy()

    stop_mask = (
        result[
            "persistent_risk_10"
        ].astype(bool)
    )

    inspect_mask = (
        result[
            "persistent_risk_20"
        ].astype(bool)
        | result[
            "persistent_risk_30"
        ].astype(bool)
    )

    result["Action"] = "CONTINUE"

    result.loc[
        inspect_mask,
        "Action",
    ] = "INSPECT"

    result.loc[
        stop_mask,
        "Action",
    ] = "STOP"

    result["Action level"] = (
        result["Action"]
        .map(ACTION_LEVELS)
        .astype(int)
    )

    result["Trigger"] = (
        "No persistent supervised risk"
    )

    result.loc[
        (
            result[
                "persistent_risk_30"
            ].astype(bool)
        ),
        "Trigger",
    ] = (
        "Persistent calibrated 30-cycle "
        "failure risk"
    )

    result.loc[
        (
            result[
                "persistent_risk_20"
            ].astype(bool)
        ),
        "Trigger",
    ] = (
        "Persistent calibrated 20-cycle "
        "failure risk"
    )

    result.loc[
        stop_mask,
        "Trigger",
    ] = (
        "Persistent calibrated 10-cycle "
        "failure risk"
    )

    result["Confidence"] = "MEDIUM"

    result.loc[
        result["Action"] == "INSPECT",
        "Confidence",
    ] = "HIGH"

    result.loc[
        result["Action"] == "STOP",
        "Confidence",
    ] = "HIGH"

    result["Next review cycles"] = (
        result["Action"]
        .map(
            {
                "CONTINUE": 10,
                "INSPECT": 1,
                "STOP": 0,
            }
        )
        .astype(int)
    )

    result["Policy"] = (
        "supervised_only"
    )

    return result


def run_fd001_inference(
    bundle: FD001ArtifactBundle,
    feature_df: pd.DataFrame,
    model_scope: Literal[
        "evaluation"
    ] = "evaluation",
) -> pd.DataFrame:
    """
    Run the complete locked FD001 inference pipeline.

    The input must contain:
    - engine_id
    - cycle
    - all saved regression features
    - all saved classification features
    - all saved anomaly features

    The evaluation RUL model is intentionally used because
    the conformal interval and maintenance policy were
    calibrated and validated with this model.
    """
    validate_fd001_inference_frame(
        feature_df
    )

    if model_scope != "evaluation":
        raise ValueError(
            "The locked unified decision pipeline "
            "currently supports only model_scope="
            "'evaluation'. The full-train RUL model "
            "does not yet have a separately calibrated "
            "conformal interval."
        )

    ordered_data = (
        feature_df
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    rul_predictions = predict_fd001_rul(
        bundle=bundle,
        data=ordered_data,
        model_scope="evaluation",
    )

    rul_interval_df = (
        create_fd001_rul_intervals(
            bundle=bundle,
            rul_predictions=rul_predictions,
        )
        .reset_index(drop=True)
    )

    probability_df = (
        predict_fd001_failure_probabilities(
            bundle=bundle,
            data=ordered_data,
        )
        .reset_index(drop=True)
    )

    anomaly_df = (
        predict_fd001_anomaly(
            bundle=bundle,
            data=ordered_data,
        )
        .reset_index(drop=True)
    )

    result = ordered_data[
        [
            "engine_id",
            "cycle",
        ]
    ].copy()

    if "RUL" in ordered_data.columns:
        result["RUL"] = (
            ordered_data["RUL"]
            .to_numpy()
        )

    result = pd.concat(
        [
            result.reset_index(
                drop=True
            ),
            rul_interval_df,
            probability_df,
            anomaly_df,
        ],
        axis=1,
    )

    threshold_10 = (
        _read_classification_threshold(
            bundle=bundle,
            horizon=10,
        )
    )

    threshold_20 = (
        _read_classification_threshold(
            bundle=bundle,
            horizon=20,
        )
    )

    threshold_30 = (
        _read_classification_threshold(
            bundle=bundle,
            horizon=30,
        )
    )

    anomaly_threshold = float(
        bundle.anomaly_config[
            "percentile_threshold"
        ]
    )

    result["risk_alert_10"] = (
        result["probability_10"]
        >= threshold_10
    ).astype(int)

    result["risk_alert_20"] = (
        result["probability_20"]
        >= threshold_20
    ).astype(int)

    result["risk_alert_30"] = (
        result["probability_30"]
        >= threshold_30
    ).astype(int)

    result["anomaly_alert"] = (
        result["anomaly_percentile"]
        >= anomaly_threshold
    ).astype(int)

    persistence_config = (
        bundle.classification_config[
            "persistence"
        ]
    )

    alerts_required = int(
        persistence_config[
            "alerts_required"
        ]
    )

    window_size = int(
        persistence_config[
            "window_size"
        ]
    )

    result = (
        _add_persistent_inference_signals(
            data=result,
            alerts_required=alerts_required,
            window_size=window_size,
        )
    )

    result["interval_crosses_10"] = (
        result["RUL lower"] <= 10.0
    ).astype(int)

    result["interval_crosses_30"] = (
        result["RUL lower"] <= 30.0
    ).astype(int)

    result[
        "supervised_signal_count"
    ] = result[
        [
            "persistent_risk_10",
            "persistent_risk_20",
            "persistent_risk_30",
        ]
    ].sum(
        axis=1
    )

    result[
        "advisory_signal_count"
    ] = result[
        [
            "persistent_anomaly",
            "interval_crosses_10",
            "interval_crosses_30",
        ]
    ].sum(
        axis=1
    )

    result[
        "total_signal_count"
    ] = (
        result[
            "supervised_signal_count"
        ]
        + result[
            "advisory_signal_count"
        ]
    )

    supervised_risk = (
        result[
            "supervised_signal_count"
        ] > 0
    )

    advisory_risk = (
        result[
            "advisory_signal_count"
        ] > 0
    )

    result[
        "Signal disagreement"
    ] = (
        supervised_risk
        != advisory_risk
    )

    result = (
        _assign_locked_supervised_action(
            evidence_df=result
        )
    )

    result = enforce_action_hysteresis(
        policy_df=result
    )

    return result


def get_latest_fd001_status(
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return the latest available maintenance status
    for every engine.
    """
    required_columns = {
        "engine_id",
        "cycle",
        "Action",
    }

    missing_columns = required_columns.difference(
        inference_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing inference result columns: "
            f"{sorted(missing_columns)}"
        )

    latest_rows = (
        inference_df
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .groupby(
            "engine_id",
            as_index=False,
        )
        .tail(1)
        .reset_index(drop=True)
    )

    return latest_rows


def create_fd001_dashboard_view(
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact engine-level table for the dashboard.
    """
    latest_rows = get_latest_fd001_status(
        inference_df
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
        "anomaly_percentile",
        "Action",
        "Confidence",
        "Trigger",
        "Next review cycles",
        "Signal disagreement",
    ]

    return latest_rows[
        dashboard_columns
    ].copy()