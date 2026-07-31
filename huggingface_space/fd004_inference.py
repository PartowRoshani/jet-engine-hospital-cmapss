"""
Standalone inference utilities for the NASA C-MAPSS FD004 system.

The module reconstructs the frozen condition-aware feature pipeline and
applies the serialized regression, classification, anomaly, uncertainty,
policy, and hysteresis contracts.

The module does not depend on notebook-defined functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


FD004_POLICY_STATE_LEVELS = {
    "Normal": 0,
    "Watch": 1,
    "Warning": 2,
    "Critical": 3,
}


FD004_POLICY_LEVEL_STATES = {
    level: state
    for state, level
    in FD004_POLICY_STATE_LEVELS.items()
}


class FD004InferenceError(ValueError):
    """Raised when the FD004 inference contract is violated."""


def load_fd004_artifact(
    artifact_path: str | Path,
) -> dict[str, Any]:
    """
    Load and structurally validate a serialized FD004 artifact.
    """
    resolved_path = Path(
        artifact_path
    ).expanduser().resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"FD004 artifact was not found: {resolved_path}"
        )

    artifact = joblib.load(
        resolved_path
    )

    required_sections = {
        "artifact_metadata",
        "raw_schema",
        "operating_regime",
        "condition_normalization",
        "feature_engineering",
        "models",
        "regression_contract",
        "classification_contract",
        "anomaly_contract",
        "policy_contract",
    }

    missing_sections = (
        required_sections
        - set(
            artifact
        )
    )

    if missing_sections:
        raise FD004InferenceError(
            "FD004 artifact is missing required sections: "
            f"{sorted(missing_sections)}"
        )

    return artifact


def _resolve_artifact(
    artifact_or_path: dict[str, Any] | str | Path,
) -> dict[str, Any]:
    """
    Accept either an already loaded artifact or an artifact path.
    """
    if isinstance(
        artifact_or_path,
        dict,
    ):
        return artifact_or_path

    return load_fd004_artifact(
        artifact_or_path
    )


def validate_fd004_raw_trajectory(
    raw_trajectory: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
    require_contiguous_cycles: bool = True,
) -> pd.DataFrame:
    """
    Validate, numerically clean, and sort raw FD004 trajectories.

    Required input columns are:

    - engine_id
    - cycle
    - three operating settings
    - twenty-one sensors
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    if not isinstance(
        raw_trajectory,
        pd.DataFrame,
    ):
        raise TypeError(
            "raw_trajectory must be a pandas DataFrame."
        )

    raw_schema = artifact[
        "raw_schema"
    ]

    required_columns = list(
        raw_schema[
            "trajectory_columns"
        ]
    )

    missing_columns = [
        column_name
        for column_name in required_columns
        if column_name
        not in raw_trajectory.columns
    ]

    if missing_columns:
        raise FD004InferenceError(
            "Raw FD004 trajectory is missing columns: "
            f"{missing_columns}"
        )

    result = (
        raw_trajectory[
            required_columns
        ]
        .copy()
    )

    for column_name in required_columns:

        result[
            column_name
        ] = pd.to_numeric(
            result[
                column_name
            ],
            errors="coerce",
        )

    missing_value_counts = (
        result
        .isna()
        .sum()
    )

    invalid_columns = (
        missing_value_counts[
            missing_value_counts
            > 0
        ]
        .index
        .tolist()
    )

    if invalid_columns:
        raise FD004InferenceError(
            "Raw FD004 trajectory contains missing or non-numeric "
            f"values in: {invalid_columns}"
        )

    if not np.isfinite(
        result[
            required_columns
        ]
        .to_numpy(dtype=float)
    ).all():
        raise FD004InferenceError(
            "Raw FD004 trajectory contains non-finite values."
        )

    result[
        "engine_id"
    ] = (
        result[
            "engine_id"
        ]
        .astype(int)
    )

    result[
        "cycle"
    ] = (
        result[
            "cycle"
        ]
        .astype(int)
    )

    if (
        result[
            "engine_id"
        ]
        <= 0
    ).any():
        raise FD004InferenceError(
            "engine_id values must be positive integers."
        )

    if (
        result[
            "cycle"
        ]
        <= 0
    ).any():
        raise FD004InferenceError(
            "cycle values must be positive integers."
        )

    result = (
        result
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
    )

    duplicate_count = int(
        result[
            [
                "engine_id",
                "cycle",
            ]
        ]
        .duplicated()
        .sum()
    )

    if duplicate_count > 0:
        raise FD004InferenceError(
            "Raw FD004 trajectory contains "
            f"{duplicate_count} duplicate engine-cycle rows."
        )

    if require_contiguous_cycles:

        for engine_id, engine_frame in (
            result
            .groupby(
                "engine_id",
                sort=True,
            )
        ):

            observed_cycles = (
                engine_frame[
                    "cycle"
                ]
                .to_numpy(dtype=int)
            )

            expected_cycles = np.arange(
                observed_cycles.min(),
                observed_cycles.max() + 1,
                dtype=int,
            )

            if not np.array_equal(
                observed_cycles,
                expected_cycles,
            ):
                raise FD004InferenceError(
                    "Engine "
                    f"{engine_id} does not contain a contiguous "
                    "cycle history."
                )

    return result


def _resolve_canonical_regime(
    raw_label: int,
    label_mapping: dict[Any, Any],
) -> int:
    """
    Resolve a raw KMeans cluster label to the canonical 1-6 label.
    """
    integer_label = int(
        raw_label
    )

    possible_keys = [
        integer_label,
        np.int64(
            integer_label
        ),
        str(
            integer_label
        ),
    ]

    for possible_key in possible_keys:

        if possible_key in label_mapping:

            return int(
                label_mapping[
                    possible_key
                ]
            )

    raise FD004InferenceError(
        "No canonical regime mapping exists for "
        f"raw KMeans label {integer_label}."
    )


def assign_fd004_operating_regimes(
    raw_trajectory: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
) -> pd.DataFrame:
    """
    Assign frozen operating regimes and centroid distances.
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    result = raw_trajectory.copy()

    setting_columns = list(
        artifact[
            "raw_schema"
        ][
            "operating_setting_columns"
        ]
    )

    regime_contract = artifact[
        "operating_regime"
    ]

    setting_scaler = regime_contract[
        "setting_scaler"
    ]

    regime_model = regime_contract[
        "kmeans_model"
    ]

    label_mapping = regime_contract[
        "raw_to_canonical_label_mapping"
    ]

    scaled_settings = (
        setting_scaler
        .transform(
            result[
                setting_columns
            ]
        )
    )

    raw_regime_labels = (
        regime_model
        .predict(
            scaled_settings
        )
    )

    canonical_regimes = np.asarray(
        [
            _resolve_canonical_regime(
                raw_label=raw_label,
                label_mapping=label_mapping,
            )
            for raw_label in raw_regime_labels
        ],
        dtype=int,
    )

    centroid_distances = (
        regime_model
        .transform(
            scaled_settings
        )
    )

    assigned_distances = (
        centroid_distances
        .min(axis=1)
    )

    result[
        "operating_regime"
    ] = canonical_regimes

    result[
        "operating_regime_distance"
    ] = assigned_distances

    regime_count = int(
        regime_contract[
            "canonical_regime_count"
        ]
    )

    for regime_number in range(
        1,
        regime_count + 1,
    ):

        result[
            f"operating_regime_{regime_number}"
        ] = (
            result[
                "operating_regime"
            ]
            .eq(
                regime_number
            )
            .astype(float)
        )

    return result


def _select_fd004_regime_statistics(
    statistics_frame: pd.DataFrame,
    canonical_regimes: np.ndarray,
    sensor_columns: list[str],
) -> np.ndarray:
    """
    Select one frozen regime-statistics row per observation.
    """
    regime_values = np.asarray(
        canonical_regimes,
        dtype=int,
    )

    available_indices = {
        int(index_value)
        for index_value
        in statistics_frame.index
    }

    canonical_set = {
        int(value)
        for value in regime_values
    }

    zero_based_set = {
        int(value - 1)
        for value in regime_values
    }

    if canonical_set.issubset(
        available_indices
    ):

        lookup_values = regime_values

    elif zero_based_set.issubset(
        available_indices
    ):

        lookup_values = (
            regime_values - 1
        )

    else:
        raise FD004InferenceError(
            "Stored regime-statistics indices are incompatible "
            "with canonical operating regimes."
        )

    selected_values = (
        statistics_frame
        .loc[
            lookup_values,
            sensor_columns,
        ]
        .to_numpy(dtype=float)
    )

    expected_shape = (
        len(
            regime_values
        ),
        len(
            sensor_columns
        ),
    )

    if (
        selected_values.shape
        != expected_shape
    ):
        raise FD004InferenceError(
            "Unexpected condition-statistics shape: "
            f"{selected_values.shape}; expected {expected_shape}."
        )

    return selected_values


def add_fd004_condition_normalization(
    regime_frame: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
) -> pd.DataFrame:
    """
    Add regime-specific condition-normalized sensor features.
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    result = regime_frame.copy()

    sensor_columns = list(
        artifact[
            "raw_schema"
        ][
            "sensor_columns"
        ]
    )

    normalization_contract = artifact[
        "condition_normalization"
    ]

    regime_sensor_means = (
        normalization_contract[
            "regime_sensor_means"
        ]
    )

    regime_sensor_scales = (
        normalization_contract[
            "regime_sensor_scales"
        ]
    )

    canonical_regimes = (
        result[
            "operating_regime"
        ]
        .to_numpy(dtype=int)
    )

    row_means = (
        _select_fd004_regime_statistics(
            statistics_frame=(
                regime_sensor_means
            ),
            canonical_regimes=(
                canonical_regimes
            ),
            sensor_columns=(
                sensor_columns
            ),
        )
    )

    row_scales = (
        _select_fd004_regime_statistics(
            statistics_frame=(
                regime_sensor_scales
            ),
            canonical_regimes=(
                canonical_regimes
            ),
            sensor_columns=(
                sensor_columns
            ),
        )
    )

    if (
        row_scales
        <= 0.0
    ).any():
        raise FD004InferenceError(
            "Condition-normalization scales must be positive."
        )

    raw_sensor_matrix = (
        result[
            sensor_columns
        ]
        .to_numpy(dtype=float)
    )

    normalized_matrix = (
        (
            raw_sensor_matrix
            - row_means
        )
        / row_scales
    )

    if not np.isfinite(
        normalized_matrix
    ).all():
        raise FD004InferenceError(
            "Condition-normalized sensor matrix contains "
            "non-finite values."
        )

    for sensor_position, sensor_name in enumerate(
        sensor_columns
    ):

        result[
            f"{sensor_name}_condition_z"
        ] = (
            normalized_matrix[
                :,
                sensor_position,
            ]
        )

    return result


def add_fd004_causal_temporal_features(
    normalized_frame: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
) -> pd.DataFrame:
    """
    Add the exact frozen causal temporal features.

    Per retained sensor:

    - current condition-normalized value;
    - delta at lag 1;
    - delta at lag 5;
    - rolling mean for windows 5, 10, and 20;
    - rolling standard deviation for windows 5, 10, and 20.
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    result = (
        normalized_frame
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    feature_contract = artifact[
        "feature_engineering"
    ]

    retained_sensors = list(
        feature_contract[
            "retained_sensor_columns"
        ]
    )

    delta_lags = list(
        feature_contract[
            "delta_lags"
        ]
    )

    rolling_windows = list(
        feature_contract[
            "rolling_windows"
        ]
    )

    rolling_min_periods = int(
        feature_contract[
            "rolling_min_periods"
        ]
    )

    rolling_ddof = int(
        feature_contract[
            "rolling_standard_deviation_ddof"
        ]
    )

    for raw_sensor_name in retained_sensors:

        base_feature = (
            f"{raw_sensor_name}_condition_z"
        )

        grouped_feature = (
            result
            .groupby(
                "engine_id",
                sort=False,
            )[
                base_feature
            ]
        )

        for lag in delta_lags:

            result[
                f"{base_feature}_delta_{lag}"
            ] = (
                grouped_feature
                .diff(
                    periods=int(
                        lag
                    )
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

        for window_size in rolling_windows:

            rolling_object = (
                grouped_feature
                .rolling(
                    window=int(
                        window_size
                    ),
                    min_periods=(
                        rolling_min_periods
                    ),
                )
            )

            rolling_mean = (
                rolling_object
                .mean()
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reindex(
                    result.index
                )
                .to_numpy(dtype=float)
            )

            rolling_std = (
                rolling_object
                .std(
                    ddof=rolling_ddof
                )
                .reset_index(
                    level=0,
                    drop=True,
                )
                .reindex(
                    result.index
                )
                .fillna(0.0)
                .to_numpy(dtype=float)
            )

            result[
                f"{base_feature}_mean_{window_size}"
            ] = rolling_mean

            result[
                f"{base_feature}_std_{window_size}"
            ] = rolling_std

    return result


def build_fd004_feature_frame(
    raw_trajectory: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
    require_contiguous_cycles: bool = True,
) -> pd.DataFrame:
    """
    Reconstruct the complete frozen FD004 feature frame.
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    validated_frame = (
        validate_fd004_raw_trajectory(
            raw_trajectory=(
                raw_trajectory
            ),
            artifact_or_path=(
                artifact
            ),
            require_contiguous_cycles=(
                require_contiguous_cycles
            ),
        )
    )

    regime_frame = (
        assign_fd004_operating_regimes(
            raw_trajectory=(
                validated_frame
            ),
            artifact_or_path=(
                artifact
            ),
        )
    )

    normalized_frame = (
        add_fd004_condition_normalization(
            regime_frame=(
                regime_frame
            ),
            artifact_or_path=(
                artifact
            ),
        )
    )

    feature_frame = (
        add_fd004_causal_temporal_features(
            normalized_frame=(
                normalized_frame
            ),
            artifact_or_path=(
                artifact
            ),
        )
    )

    feature_contract = artifact[
        "feature_engineering"
    ]

    required_model_features = set(
        feature_contract[
            "regression_feature_columns"
        ]
    )

    required_model_features.update(
        feature_contract[
            "classification_feature_columns"
        ]
    )

    required_model_features.update(
        feature_contract[
            "anomaly_feature_columns"
        ]
    )

    missing_model_features = sorted(
        required_model_features
        - set(
            feature_frame.columns
        )
    )

    if missing_model_features:
        raise FD004InferenceError(
            "Engineered FD004 feature frame is missing: "
            f"{missing_model_features}"
        )

    if not np.isfinite(
        feature_frame[
            sorted(
                required_model_features
            )
        ]
        .to_numpy(dtype=float)
    ).all():
        raise FD004InferenceError(
            "Engineered FD004 features contain non-finite values."
        )

    return feature_frame


def apply_fd004_regression_postprocessing(
    raw_predictions: np.ndarray,
    regression_contract: dict[str, Any],
) -> np.ndarray:
    """
    Apply the frozen ceiling, safety-offset, and non-negative rules.
    """
    predictions = np.asarray(
        raw_predictions,
        dtype=float,
    ).copy()

    prediction_ceiling = (
        regression_contract[
            "prediction_ceiling"
        ]
    )

    if prediction_ceiling is not None:

        predictions = np.minimum(
            predictions,
            float(
                prediction_ceiling
            ),
        )

    predictions = (
        predictions
        - float(
            regression_contract[
                "safety_offset"
            ]
        )
    )

    predictions = np.maximum(
        predictions,
        float(
            regression_contract[
                "minimum_prediction"
            ]
        ),
    )

    return predictions


def _calibrate_fd004_probability(
    calibrator: Any,
    raw_probability: np.ndarray,
) -> np.ndarray:
    """
    Apply the frozen isotonic probability calibrator.
    """
    calibrated_probability = (
        calibrator.predict(
            np.asarray(
                raw_probability,
                dtype=float,
            )
        )
    )

    return np.clip(
        calibrated_probability,
        0.0,
        1.0,
    )


def _calculate_fd004_raw_policy(
    alert_10: np.ndarray,
    alert_20: np.ndarray,
    alert_30: np.ndarray,
    anomaly_alert: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """
    Calculate raw policy state, level, and reason.
    """
    alert_10 = np.asarray(
        alert_10,
        dtype=int,
    )

    alert_20 = np.asarray(
        alert_20,
        dtype=int,
    )

    alert_30 = np.asarray(
        alert_30,
        dtype=int,
    )

    anomaly_alert = np.asarray(
        anomaly_alert,
        dtype=int,
    )

    raw_states = np.select(
        condlist=[
            alert_10 == 1,
            (
                (alert_10 == 0)
                &
                (alert_20 == 1)
            ),
            (
                (alert_10 == 0)
                &
                (alert_20 == 0)
                &
                (
                    (alert_30 == 1)
                    |
                    (anomaly_alert == 1)
                )
            ),
        ],
        choicelist=[
            "Critical",
            "Warning",
            "Watch",
        ],
        default="Normal",
    )

    raw_reasons = np.select(
        condlist=[
            alert_10 == 1,
            (
                (alert_10 == 0)
                &
                (alert_20 == 1)
            ),
            (
                (alert_10 == 0)
                &
                (alert_20 == 0)
                &
                (alert_30 == 1)
            ),
            (
                (alert_10 == 0)
                &
                (alert_20 == 0)
                &
                (alert_30 == 0)
                &
                (anomaly_alert == 1)
            ),
        ],
        choicelist=[
            "10-cycle supervised alert",
            "20-cycle supervised alert",
            "30-cycle supervised alert",
            "Anomaly-only watch",
        ],
        default="No active alert",
    )

    raw_levels = np.asarray(
        [
            FD004_POLICY_STATE_LEVELS[
                state_name
            ]
            for state_name in raw_states
        ],
        dtype=int,
    )

    return (
        raw_states,
        raw_levels,
        raw_reasons,
    )


def apply_fd004_policy_hysteresis(
    prediction_table: pd.DataFrame,
    confirmation_cycles: int,
) -> pd.DataFrame:
    """
    Apply immediate escalation and confirmed de-escalation.
    """
    if confirmation_cycles < 1:
        raise FD004InferenceError(
            "confirmation_cycles must be at least one."
        )

    result = (
        prediction_table
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )

    raw_levels = (
        result[
            "raw_policy_level"
        ]
        .to_numpy(dtype=int)
    )

    operational_levels = np.full(
        shape=len(result),
        fill_value=-1,
        dtype=int,
    )

    hysteresis_hold = np.zeros(
        shape=len(result),
        dtype=bool,
    )

    deescalation_run_length = np.zeros(
        shape=len(result),
        dtype=int,
    )

    for _, engine_indices in (
        result
        .groupby(
            "engine_id",
            sort=True,
        )
        .groups
        .items()
    ):

        ordered_indices = np.asarray(
            list(
                engine_indices
            ),
            dtype=int,
        )

        current_level = int(
            raw_levels[
                ordered_indices[0]
            ]
        )

        consecutive_lower_cycles = 0

        operational_levels[
            ordered_indices[0]
        ] = current_level

        for local_position in range(
            1,
            len(
                ordered_indices
            ),
        ):

            row_index = int(
                ordered_indices[
                    local_position
                ]
            )

            proposed_level = int(
                raw_levels[
                    row_index
                ]
            )

            if proposed_level > current_level:

                current_level = (
                    proposed_level
                )

                consecutive_lower_cycles = 0

            elif proposed_level == current_level:

                consecutive_lower_cycles = 0

            else:

                consecutive_lower_cycles += 1

                if (
                    consecutive_lower_cycles
                    >= confirmation_cycles
                ):

                    current_level = (
                        proposed_level
                    )

                    consecutive_lower_cycles = 0

                else:

                    hysteresis_hold[
                        row_index
                    ] = True

            operational_levels[
                row_index
            ] = current_level

            deescalation_run_length[
                row_index
            ] = consecutive_lower_cycles

    result[
        "policy_level"
    ] = operational_levels

    result[
        "policy_state"
    ] = [
        FD004_POLICY_LEVEL_STATES[
            int(level)
        ]
        for level in operational_levels
    ]

    result[
        "hysteresis_hold"
    ] = hysteresis_hold

    result[
        "deescalation_run_length"
    ] = deescalation_run_length

    result[
        "policy_reason"
    ] = np.where(
        hysteresis_hold,
        "State retained by de-escalation hysteresis",
        result[
            "raw_policy_reason"
        ],
    )

    return result


def predict_fd004_trajectory(
    raw_trajectory: pd.DataFrame,
    artifact_or_path: dict[str, Any] | str | Path,
    require_contiguous_cycles: bool = True,
    return_feature_frame: bool = False,
) -> pd.DataFrame | tuple[
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run complete FD004 trajectory inference.
    """
    artifact = _resolve_artifact(
        artifact_or_path
    )

    feature_frame = (
        build_fd004_feature_frame(
            raw_trajectory=(
                raw_trajectory
            ),
            artifact_or_path=(
                artifact
            ),
            require_contiguous_cycles=(
                require_contiguous_cycles
            ),
        )
    )

    feature_contract = artifact[
        "feature_engineering"
    ]

    regression_columns = list(
        feature_contract[
            "regression_feature_columns"
        ]
    )

    classification_columns = list(
        feature_contract[
            "classification_feature_columns"
        ]
    )

    anomaly_columns = list(
        feature_contract[
            "anomaly_feature_columns"
        ]
    )

    regression_matrix = (
        feature_frame[
            regression_columns
        ]
        .to_numpy(dtype=float)
    )

    classification_matrix = (
        feature_frame[
            classification_columns
        ]
        .to_numpy(dtype=float)
    )

    anomaly_matrix = (
        feature_frame[
            anomaly_columns
        ]
        .to_numpy(dtype=float)
    )

    regression_model = artifact[
        "models"
    ][
        "regression"
    ]

    raw_rul_prediction = (
        regression_model
        .predict(
            regression_matrix
        )
    )

    regression_contract = artifact[
        "regression_contract"
    ]

    locked_rul_prediction = (
        apply_fd004_regression_postprocessing(
            raw_predictions=(
                raw_rul_prediction
            ),
            regression_contract=(
                regression_contract
            ),
        )
    )

    conformal_quantile = float(
        regression_contract[
            "deployment_conformal_quantile"
        ]
    )

    rul_lower = np.maximum(
        float(
            regression_contract[
                "interval_lower_bound"
            ]
        ),
        (
            locked_rul_prediction
            - conformal_quantile
        ),
    )

    rul_upper = (
        locked_rul_prediction
        + conformal_quantile
    )

    classification_contract = artifact[
        "classification_contract"
    ]

    horizons = [
        int(horizon)
        for horizon in (
            classification_contract[
                "horizons"
            ]
        )
    ]

    raw_probability_by_horizon = {}

    calibrated_probability_by_horizon = {}

    for horizon in horizons:

        classification_model = artifact[
            "models"
        ][
            "classification"
        ][
            horizon
        ]

        probability_calibrator = artifact[
            "models"
        ][
            "classification_calibrators"
        ][
            horizon
        ]

        raw_probability = (
            classification_model
            .predict_proba(
                classification_matrix
            )[:, 1]
        )

        calibrated_probability = (
            _calibrate_fd004_probability(
                calibrator=(
                    probability_calibrator
                ),
                raw_probability=(
                    raw_probability
                ),
            )
        )

        raw_probability_by_horizon[
            horizon
        ] = raw_probability

        calibrated_probability_by_horizon[
            horizon
        ] = calibrated_probability

    probability_10 = (
        calibrated_probability_by_horizon[
            10
        ]
    )

    probability_20 = np.maximum(
        probability_10,
        calibrated_probability_by_horizon[
            20
        ],
    )

    probability_30 = np.maximum(
        probability_20,
        calibrated_probability_by_horizon[
            30
        ],
    )

    thresholds = classification_contract[
        "thresholds"
    ]

    alert_10 = (
        probability_10
        >= float(
            thresholds[
                10
            ]
        )
    ).astype(int)

    alert_20 = (
        probability_20
        >= float(
            thresholds[
                20
            ]
        )
    ).astype(int)

    alert_30 = (
        probability_30
        >= float(
            thresholds[
                30
            ]
        )
    ).astype(int)

    anomaly_model = artifact[
        "models"
    ][
        "anomaly"
    ]

    anomaly_severity = -(
        anomaly_model
        .decision_function(
            anomaly_matrix
        )
    )

    anomaly_threshold = float(
        artifact[
            "anomaly_contract"
        ][
            "deployment_threshold"
        ]
    )

    anomaly_alert = (
        anomaly_severity
        >= anomaly_threshold
    ).astype(int)

    (
        raw_policy_state,
        raw_policy_level,
        raw_policy_reason,
    ) = _calculate_fd004_raw_policy(
        alert_10=alert_10,
        alert_20=alert_20,
        alert_30=alert_30,
        anomaly_alert=anomaly_alert,
    )

    prediction_table = (
        feature_frame[
            [
                "engine_id",
                "cycle",
                "operating_regime",
                "operating_regime_distance",
            ]
        ]
        .copy()
    )

    prediction_table[
        "raw_RUL_prediction"
    ] = raw_rul_prediction

    prediction_table[
        "locked_RUL_prediction"
    ] = locked_rul_prediction

    prediction_table[
        "RUL_lower"
    ] = rul_lower

    prediction_table[
        "RUL_upper"
    ] = rul_upper

    prediction_table[
        "interval_width"
    ] = (
        rul_upper
        - rul_lower
    )

    for horizon in horizons:

        prediction_table[
            f"probability_{horizon}_raw"
        ] = raw_probability_by_horizon[
            horizon
        ]

        prediction_table[
            f"probability_{horizon}_calibrated"
        ] = calibrated_probability_by_horizon[
            horizon
        ]

    prediction_table[
        "probability_10"
    ] = probability_10

    prediction_table[
        "probability_20"
    ] = probability_20

    prediction_table[
        "probability_30"
    ] = probability_30

    prediction_table[
        "alert_10"
    ] = alert_10

    prediction_table[
        "alert_20"
    ] = alert_20

    prediction_table[
        "alert_30"
    ] = alert_30

    prediction_table[
        "anomaly_severity"
    ] = anomaly_severity

    prediction_table[
        "anomaly_threshold"
    ] = anomaly_threshold

    prediction_table[
        "anomaly_alert"
    ] = anomaly_alert

    prediction_table[
        "raw_policy_state"
    ] = raw_policy_state

    prediction_table[
        "raw_policy_level"
    ] = raw_policy_level

    prediction_table[
        "raw_policy_reason"
    ] = raw_policy_reason

    confirmation_cycles = int(
        artifact[
            "policy_contract"
        ][
            "deescalation_confirmation_cycles"
        ]
    )

    prediction_table = (
        apply_fd004_policy_hysteresis(
            prediction_table=(
                prediction_table
            ),
            confirmation_cycles=(
                confirmation_cycles
            ),
        )
    )

    if return_feature_frame:

        return (
            prediction_table,
            feature_frame,
        )

    return prediction_table


def create_fd004_terminal_summary(
    trajectory_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return one final-cycle deployment record per engine.
    """
    required_columns = {
        "engine_id",
        "cycle",
        "locked_RUL_prediction",
        "RUL_lower",
        "RUL_upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_severity",
        "anomaly_alert",
        "policy_state",
        "policy_reason",
    }

    missing_columns = (
        required_columns
        - set(
            trajectory_predictions.columns
        )
    )

    if missing_columns:
        raise FD004InferenceError(
            "Prediction table is missing terminal-summary columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        trajectory_predictions
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .groupby(
            "engine_id",
            as_index=False,
            sort=True,
        )
        .tail(1)
        .sort_values(
            "engine_id"
        )
        .reset_index(drop=True)
    )
