from __future__ import annotations

from math import ceil
from typing import Any

import numpy as np
import pandas as pd

from src.config import RANDOM_STATE


def calculate_absolute_residuals(
    y_true: pd.Series | np.ndarray,
    y_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Calculate absolute residuals for conformal calibration.
    """
    y_true_array = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    ).reshape(-1)

    if len(y_true_array) != len(y_pred_array):
        raise ValueError(
            "y_true and y_pred must have the same length."
        )

    if len(y_true_array) == 0:
        raise ValueError(
            "Input arrays must not be empty."
        )

    if not np.isfinite(y_true_array).all():
        raise ValueError(
            "y_true contains NaN or infinite values."
        )

    if not np.isfinite(y_pred_array).all():
        raise ValueError(
            "y_pred contains NaN or infinite values."
        )

    return np.abs(
        y_true_array - y_pred_array
    )


def calculate_conformal_quantile(
    calibration_scores: pd.Series | np.ndarray,
    confidence_level: float = 0.90,
) -> float:
    """
    Calculate the finite-sample split-conformal quantile.

    The quantile uses the conservative 'higher' order
    statistic required for finite-sample conformal coverage.
    """
    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    scores = np.asarray(
        calibration_scores,
        dtype=float,
    ).reshape(-1)

    if len(scores) == 0:
        raise ValueError(
            "Calibration scores must not be empty."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Calibration scores contain NaN or infinity."
        )

    scores = np.sort(scores)

    sample_count = len(scores)

    rank = ceil(
        (sample_count + 1)
        * confidence_level
    )

    rank = min(
        max(rank, 1),
        sample_count,
    )

    return float(
        scores[rank - 1]
    )


def create_conformal_intervals(
    predictions: pd.Series | np.ndarray,
    conformal_quantile: float,
    lower_bound: float = 0.0,
) -> pd.DataFrame:
    """
    Create symmetric split-conformal prediction intervals.
    """
    prediction_array = np.asarray(
        predictions,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(prediction_array).all():
        raise ValueError(
            "Predictions contain NaN or infinity."
        )

    if conformal_quantile < 0.0:
        raise ValueError(
            "conformal_quantile must be non-negative."
        )

    lower = np.maximum(
        lower_bound,
        prediction_array - conformal_quantile,
    )

    upper = (
        prediction_array + conformal_quantile
    )

    return pd.DataFrame(
        {
            "RUL prediction": prediction_array,
            "RUL lower": lower,
            "RUL upper": upper,
            "Interval width": upper - lower,
        }
    )


def evaluate_prediction_intervals(
    y_true: pd.Series | np.ndarray,
    interval_df: pd.DataFrame,
) -> dict[str, Any]:
    """
    Evaluate empirical coverage and interval width.
    """
    required_columns = {
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "Interval width",
    }

    missing_columns = required_columns.difference(
        interval_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing interval columns: "
            f"{sorted(missing_columns)}"
        )

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    ).reshape(-1)

    if len(y_true_array) != len(interval_df):
        raise ValueError(
            "y_true and interval_df must have "
            "the same length."
        )

    lower = interval_df[
        "RUL lower"
    ].to_numpy(dtype=float)

    upper = interval_df[
        "RUL upper"
    ].to_numpy(dtype=float)

    widths = interval_df[
        "Interval width"
    ].to_numpy(dtype=float)

    prediction = interval_df[
        "RUL prediction"
    ].to_numpy(dtype=float)

    covered = (
        (y_true_array >= lower)
        & (y_true_array <= upper)
    )

    lower_miss = (
        y_true_array < lower
    )

    upper_miss = (
        y_true_array > upper
    )

    return {
        "Samples": len(y_true_array),
        "Empirical coverage": float(
            covered.mean()
        ),
        "Average interval width": float(
            widths.mean()
        ),
        "Median interval width": float(
            np.median(widths)
        ),
        "Minimum interval width": float(
            widths.min()
        ),
        "Maximum interval width": float(
            widths.max()
        ),
        "Lower misses": int(
            lower_miss.sum()
        ),
        "Upper misses": int(
            upper_miss.sum()
        ),
        "Total misses": int(
            (~covered).sum()
        ),
        "Mean prediction": float(
            prediction.mean()
        ),
        "Mean actual RUL": float(
            y_true_array.mean()
        ),
    }


def evaluate_interval_regions(
    evaluation_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate interval coverage in operational RUL regions.
    """
    required_columns = {
        "RUL",
        "RUL lower",
        "RUL upper",
        "Interval width",
    }

    missing_columns = required_columns.difference(
        evaluation_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    result_df = evaluation_df.copy()

    result_df["RUL region"] = pd.cut(
        result_df["RUL"],
        bins=[
            -1,
            30,
            125,
            np.inf,
        ],
        labels=[
            "Near failure: 0–30",
            "Mid life: 31–125",
            "Early life: >125",
        ],
    )

    result_df["Covered"] = (
        (
            result_df["RUL"]
            >= result_df["RUL lower"]
        )
        & (
            result_df["RUL"]
            <= result_df["RUL upper"]
        )
    )

    result_df["Lower miss"] = (
        result_df["RUL"]
        < result_df["RUL lower"]
    )

    result_df["Upper miss"] = (
        result_df["RUL"]
        > result_df["RUL upper"]
    )

    return (
        result_df
        .groupby(
            "RUL region",
            observed=True,
        )
        .agg(
            Samples=(
                "Covered",
                "size",
            ),
            Coverage=(
                "Covered",
                "mean",
            ),
            Average_width=(
                "Interval width",
                "mean",
            ),
            Median_width=(
                "Interval width",
                "median",
            ),
            Lower_misses=(
                "Lower miss",
                "sum",
            ),
            Upper_misses=(
                "Upper miss",
                "sum",
            ),
        )
        .reset_index()
    )


def bootstrap_interval_metrics(
    evaluation_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate confidence intervals by resampling engines.
    """
    required_columns = {
        "engine_id",
        "RUL",
        "RUL lower",
        "RUL upper",
        "Interval width",
    }

    missing_columns = required_columns.difference(
        evaluation_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    engine_ids = (
        evaluation_df["engine_id"]
        .drop_duplicates()
        .to_numpy()
    )

    if len(engine_ids) < 2:
        raise ValueError(
            "At least two engines are required."
        )

    engine_frames = {
        engine_id: evaluation_df.loc[
            evaluation_df["engine_id"]
            == engine_id
        ]
        for engine_id in engine_ids
    }

    covered = (
        (
            evaluation_df["RUL"]
            >= evaluation_df["RUL lower"]
        )
        & (
            evaluation_df["RUL"]
            <= evaluation_df["RUL upper"]
        )
    )

    point_estimates = {
        "Empirical coverage": float(
            covered.mean()
        ),
        "Average interval width": float(
            evaluation_df[
                "Interval width"
            ].mean()
        ),
        "Median interval width": float(
            evaluation_df[
                "Interval width"
            ].median()
        ),
    }

    bootstrap_values = {
        metric: []
        for metric in point_estimates
    }

    rng = np.random.default_rng(
        random_state
    )

    for _ in range(n_bootstrap):
        sampled_engine_ids = rng.choice(
            engine_ids,
            size=len(engine_ids),
            replace=True,
        )

        sampled_df = pd.concat(
            [
                engine_frames[engine_id]
                for engine_id
                in sampled_engine_ids
            ],
            ignore_index=True,
        )

        sampled_covered = (
            (
                sampled_df["RUL"]
                >= sampled_df["RUL lower"]
            )
            & (
                sampled_df["RUL"]
                <= sampled_df["RUL upper"]
            )
        )

        bootstrap_values[
            "Empirical coverage"
        ].append(
            float(
                sampled_covered.mean()
            )
        )

        bootstrap_values[
            "Average interval width"
        ].append(
            float(
                sampled_df[
                    "Interval width"
                ].mean()
            )
        )

        bootstrap_values[
            "Median interval width"
        ].append(
            float(
                sampled_df[
                    "Interval width"
                ].median()
            )
        )

    alpha = 1.0 - confidence_level

    result_rows = []

    for metric_name, estimate in (
        point_estimates.items()
    ):
        values = np.asarray(
            bootstrap_values[metric_name],
            dtype=float,
        )

        result_rows.append(
            {
                "Metric": metric_name,
                "Estimate": estimate,
                "CI lower": float(
                    np.quantile(
                        values,
                        alpha / 2.0,
                    )
                ),
                "CI upper": float(
                    np.quantile(
                        values,
                        1.0 - alpha / 2.0,
                    )
                ),
                "Confidence level": (
                    confidence_level
                ),
                "Method": (
                    "Engine bootstrap"
                ),
            }
        )

    return pd.DataFrame(
        result_rows
    )