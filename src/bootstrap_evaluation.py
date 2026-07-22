from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def calculate_fd001_nasa_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate the asymmetric NASA C-MAPSS score.

    Late predictions, where predicted RUL exceeds actual RUL,
    receive the stronger exponential penalty.
    """
    actual = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    predicted = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError(
            "y_true and y_pred must have equal lengths."
        )

    errors = predicted - actual

    penalties = np.where(
        errors < 0.0,
        np.expm1(
            -errors / 13.0
        ),
        np.expm1(
            errors / 10.0
        ),
    )

    return float(
        penalties.sum()
    )


def calculate_fd001_regression_metrics(
    evaluation_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate locked FD001 RUL regression metrics.
    """
    required_columns = {
        "RUL",
        "RUL prediction",
    }

    missing_columns = required_columns.difference(
        evaluation_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing regression metric columns: "
            f"{sorted(missing_columns)}"
        )

    actual = evaluation_df[
        "RUL"
    ].to_numpy(
        dtype=float
    )

    predicted = evaluation_df[
        "RUL prediction"
    ].to_numpy(
        dtype=float
    )

    if not np.isfinite(actual).all():
        raise ValueError(
            "Actual RUL contains invalid values."
        )

    if not np.isfinite(predicted).all():
        raise ValueError(
            "Predicted RUL contains invalid values."
        )

    near_failure_mask = (
        actual <= 30.0
    )

    nasa_score = (
        calculate_fd001_nasa_score(
            actual,
            predicted,
        )
    )

    return {
        "MAE": float(
            mean_absolute_error(
                actual,
                predicted,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    actual,
                    predicted,
                )
            )
        ),
        "R2": float(
            r2_score(
                actual,
                predicted,
            )
        ),
        "NASA score": nasa_score,
        "NASA penalty per row": float(
            nasa_score / len(
                evaluation_df
            )
        ),
        "Near-failure MAE": float(
            mean_absolute_error(
                actual[
                    near_failure_mask
                ],
                predicted[
                    near_failure_mask
                ],
            )
        ),
        "Late-prediction rate": float(
            (
                predicted > actual
            ).mean()
        ),
    }


def calculate_fd001_interval_metrics(
    interval_df: pd.DataFrame,
) -> dict[str, float]:
    """
    Calculate overall and regional conformal interval metrics.
    """
    required_columns = {
        "RUL",
        "RUL lower",
        "RUL upper",
    }

    missing_columns = required_columns.difference(
        interval_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing interval metric columns: "
            f"{sorted(missing_columns)}"
        )

    actual = interval_df[
        "RUL"
    ].to_numpy(
        dtype=float
    )

    lower = interval_df[
        "RUL lower"
    ].to_numpy(
        dtype=float
    )

    upper = interval_df[
        "RUL upper"
    ].to_numpy(
        dtype=float
    )

    if not (
        np.isfinite(actual).all()
        and np.isfinite(lower).all()
        and np.isfinite(upper).all()
    ):
        raise ValueError(
            "Interval data contains invalid values."
        )

    if (lower > upper).any():
        raise ValueError(
            "RUL lower cannot exceed RUL upper."
        )

    covered = (
        (actual >= lower)
        & (actual <= upper)
    )

    width = upper - lower

    region_masks = {
        "RUL 0-30": (
            actual <= 30.0
        ),
        "RUL 31-125": (
            (actual > 30.0)
            & (actual <= 125.0)
        ),
        "RUL >125": (
            actual > 125.0
        ),
    }

    metrics = {
        "Overall coverage": float(
            covered.mean()
        ),
        "Average interval width": float(
            width.mean()
        ),
        "Median interval width": float(
            np.median(width)
        ),
    }

    for region_name, region_mask in (
        region_masks.items()
    ):
        if not region_mask.any():
            metrics[
                f"Coverage {region_name}"
            ] = np.nan

            metrics[
                f"Average width {region_name}"
            ] = np.nan

            continue

        metrics[
            f"Coverage {region_name}"
        ] = float(
            covered[
                region_mask
            ].mean()
        )

        metrics[
            f"Average width {region_name}"
        ] = float(
            width[
                region_mask
            ].mean()
        )

    return metrics


def _validate_bootstrap_parameters(
    n_bootstrap: int,
    confidence_level: float,
) -> None:
    """
    Validate common bootstrap settings.
    """
    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )


def _bootstrap_engine_metrics(
    data: pd.DataFrame,
    metric_function: Callable[
        [pd.DataFrame],
        dict[str, float],
    ],
    n_bootstrap: int,
    confidence_level: float,
    random_state: int,
) -> pd.DataFrame:
    """
    Generic engine-level percentile bootstrap.
    """
    _validate_bootstrap_parameters(
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
    )

    if "engine_id" not in data.columns:
        raise ValueError(
            "engine_id is required for engine bootstrap."
        )

    engine_ids = (
        data["engine_id"]
        .drop_duplicates()
        .to_numpy()
    )

    if len(engine_ids) < 2:
        raise ValueError(
            "At least two engines are required."
        )

    engine_frames = {
        engine_id: (
            data.loc[
                data["engine_id"]
                == engine_id
            ]
            .copy()
        )
        for engine_id in engine_ids
    }

    point_estimates = metric_function(
        data
    )

    bootstrap_values: dict[
        str,
        list[float],
    ] = {
        metric_name: []
        for metric_name
        in point_estimates
    }

    random_generator = (
        np.random.default_rng(
            random_state
        )
    )

    for _ in range(n_bootstrap):
        sampled_engine_ids = (
            random_generator.choice(
                engine_ids,
                size=len(engine_ids),
                replace=True,
            )
        )

        sampled_data = pd.concat(
            [
                engine_frames[
                    engine_id
                ]
                for engine_id
                in sampled_engine_ids
            ],
            ignore_index=True,
        )

        sampled_metrics = metric_function(
            sampled_data
        )

        for (
            metric_name,
            metric_value,
        ) in sampled_metrics.items():
            bootstrap_values[
                metric_name
            ].append(
                float(metric_value)
            )

    alpha = (
        1.0
        - confidence_level
    )

    result_rows = []

    for (
        metric_name,
        point_estimate,
    ) in point_estimates.items():
        metric_values = np.asarray(
            bootstrap_values[
                metric_name
            ],
            dtype=float,
        )

        metric_values = metric_values[
            np.isfinite(
                metric_values
            )
        ]

        if len(metric_values) == 0:
            ci_lower = np.nan
            ci_upper = np.nan
        else:
            ci_lower = float(
                np.quantile(
                    metric_values,
                    alpha / 2.0,
                )
            )

            ci_upper = float(
                np.quantile(
                    metric_values,
                    1.0 - alpha / 2.0,
                )
            )

        result_rows.append(
            {
                "Metric": metric_name,
                "Estimate": float(
                    point_estimate
                ),
                "CI lower": ci_lower,
                "CI upper": ci_upper,
                "Confidence level": (
                    confidence_level
                ),
                "Method": (
                    "Engine percentile bootstrap"
                ),
                "Bootstrap samples": (
                    n_bootstrap
                ),
                "Engine count": int(
                    len(engine_ids)
                ),
            }
        )

    return pd.DataFrame(
        result_rows
    )


def bootstrap_fd001_regression_metrics(
    evaluation_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Engine-bootstrap confidence intervals for RUL metrics.
    """
    return _bootstrap_engine_metrics(
        data=evaluation_df,
        metric_function=(
            calculate_fd001_regression_metrics
        ),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )


def bootstrap_fd001_interval_metrics(
    interval_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Engine-bootstrap confidence intervals for conformal metrics.
    """
    return _bootstrap_engine_metrics(
        data=interval_df,
        metric_function=(
            calculate_fd001_interval_metrics
        ),
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=random_state,
    )