from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


DEFAULT_ROLLING_WINDOWS = (5, 10, 20)
DEFAULT_CHANGE_PERIODS = (1, 5)


def build_time_series_features(
    dataframe: pd.DataFrame,
    sensor_columns: Iterable[str],
    rolling_windows: tuple[int, ...] = DEFAULT_ROLLING_WINDOWS,
    change_periods: tuple[int, ...] = DEFAULT_CHANGE_PERIODS,
) -> pd.DataFrame:
    """
    Create leakage-safe time-series features for each engine.

    All temporal features are calculated independently for each engine
    and use only the current and previous cycles. Future observations
    are never used.
    """

    required_columns = {"engine_id", "cycle"}
    missing_required = required_columns - set(dataframe.columns)

    if missing_required:
        raise ValueError(
            f"Missing required columns: {sorted(missing_required)}"
        )

    # Remove accidental duplicate sensor names while preserving order.
    sensor_columns = list(dict.fromkeys(sensor_columns))

    missing_sensors = [
        column
        for column in sensor_columns
        if column not in dataframe.columns
    ]

    if missing_sensors:
        raise ValueError(
            f"Missing sensor columns: {missing_sensors}"
        )

    if not rolling_windows:
        raise ValueError(
            "At least one rolling window must be provided."
        )

    if any(window <= 0 for window in rolling_windows):
        raise ValueError(
            "Rolling-window sizes must be positive integers."
        )

    if any(period <= 0 for period in change_periods):
        raise ValueError(
            "Change periods must be positive integers."
        )

    # Sort observations so every engine follows chronological order.
    result = (
        dataframe
        .sort_values(["engine_id", "cycle"])
        .reset_index(drop=True)
        .copy()
    )

    grouped = result.groupby(
        "engine_id",
        sort=False,
    )

    # Store all new columns here instead of inserting them one by one.
    generated_features: dict[str, pd.Series] = {}

    for sensor_name in sensor_columns:
        sensor_group = grouped[sensor_name]

        # Difference from earlier cycles of the same engine.
        for period in change_periods:
            column_name = f"{sensor_name}_delta_{period}"

            generated_features[column_name] = (
                sensor_group
                .diff(periods=period)
                .fillna(0.0)
            )

        # Rolling statistics based only on current and past cycles.
        for window in rolling_windows:
            rolling_group = sensor_group.rolling(
                window=window,
                min_periods=1,
            )

            mean_column = f"{sensor_name}_mean_{window}"
            std_column = f"{sensor_name}_std_{window}"

            generated_features[mean_column] = (
                rolling_group
                .mean()
                .reset_index(level=0, drop=True)
                .reindex(result.index)
            )

            generated_features[std_column] = (
                rolling_group
                .std(ddof=0)
                .reset_index(level=0, drop=True)
                .reindex(result.index)
            )

    # Build all generated features in one DataFrame.
    temporal_features = pd.DataFrame(
        generated_features,
        index=result.index,
    )

    temporal_features = temporal_features.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    if temporal_features.isna().any().any():
        missing_columns = (
            temporal_features.columns[
                temporal_features.isna().any()
            ]
            .tolist()
        )

        raise ValueError(
            "Temporal feature generation produced missing values "
            f"in columns: {missing_columns}"
        )

    # Add all generated columns at once to avoid fragmentation.
    result = pd.concat(
        [result, temporal_features],
        axis=1,
    ).copy()

    return result


def get_model_feature_columns(
    dataframe: pd.DataFrame,
    constant_features: Iterable[str] | None = None,
) -> list[str]:
    """
    Return columns that may be supplied to machine-learning models.

    Identifiers, future information, and target variables are excluded.
    """

    excluded_columns = {
        "engine_id",
        "final_cycle",
        "RUL",
        "failure_within_10",
        "failure_within_20",
        "failure_within_30",
    }

    if constant_features is not None:
        excluded_columns.update(constant_features)

    numeric_columns = dataframe.select_dtypes(
        include=np.number
    ).columns

    return [
        column
        for column in numeric_columns
        if column not in excluded_columns
    ]