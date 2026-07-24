from __future__ import annotations

import numpy as np
import pandas as pd
from statistics import NormalDist

from src.config import RANDOM_STATE


def evaluate_persistent_early_warnings(
    risk_df: pd.DataFrame,
    horizon: int,
    alert_column: str,
    alerts_required: int = 2,
    window_size: int = 3,
) -> pd.DataFrame:
    """
    Evaluate persistent early warnings at engine level.

    A persistent warning occurs when at least
    `alerts_required` alerts appear within the most recent
    `window_size` cycles.

    Lead time:
        failure_cycle - first_persistent_alert_cycle
    """
    if alerts_required <= 0:
        raise ValueError(
            "alerts_required must be positive."
        )

    if window_size <= 0:
        raise ValueError(
            "window_size must be positive."
        )

    if alerts_required > window_size:
        raise ValueError(
            "alerts_required cannot exceed window_size."
        )

    required_columns = {
        "engine_id",
        "cycle",
        "RUL",
        alert_column,
    }

    missing_columns = (
        required_columns.difference(
            risk_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    engine_results = []

    for engine_id, engine_df in risk_df.groupby(
        "engine_id",
        sort=True,
    ):
        engine_df = (
            engine_df
            .sort_values("cycle")
            .copy()
        )

        alerts = (
            engine_df[alert_column]
            .astype(int)
        )

        persistent_alert = (
            alerts
            .rolling(
                window=window_size,
                min_periods=window_size,
            )
            .sum()
            >= alerts_required
        )

        failure_cycle_values = (
            engine_df["cycle"]
            + engine_df["RUL"]
        )

        failure_cycle = int(
            round(
                failure_cycle_values.median()
            )
        )

        if persistent_alert.any():
            first_alert_index = (
                persistent_alert[
                    persistent_alert
                ].index[0]
            )

            first_alert_cycle = int(
                engine_df.loc[
                    first_alert_index,
                    "cycle",
                ]
            )

            lead_time = (
                failure_cycle
                - first_alert_cycle
            )

            missed_warning = 0

            late_warning_delay = max(
                0,
                horizon - lead_time,
            )

            early_warning_burden = max(
                0,
                lead_time - horizon,
            )

        else:
            first_alert_cycle = np.nan
            lead_time = np.nan
            missed_warning = 1
            late_warning_delay = horizon
            early_warning_burden = 0

        engine_results.append(
            {
                "engine_id": int(engine_id),
                "Horizon": int(horizon),
                "Failure cycle": failure_cycle,
                "First persistent alert cycle": (
                    first_alert_cycle
                ),
                "Lead time": lead_time,
                "Missed warning": missed_warning,
                "Late-warning delay": (
                    late_warning_delay
                ),
                "Early-warning burden": (
                    early_warning_burden
                ),
                "Persistence rule": (
                    f"{alerts_required}-of-"
                    f"{window_size}"
                ),
            }
        )

    return pd.DataFrame(
        engine_results
    )


def summarize_early_warnings(
    engine_warning_df: pd.DataFrame,
) -> dict[str, float | int | str]:
    """
    Summarize engine-level early-warning performance.
    """
    detected_df = engine_warning_df.loc[
        engine_warning_df[
            "Missed warning"
        ] == 0
    ]

    total_engines = len(
        engine_warning_df
    )

    missed_engines = int(
        engine_warning_df[
            "Missed warning"
        ].sum()
    )

    if detected_df.empty:
        mean_lead_time = np.nan
        median_lead_time = np.nan
        minimum_lead_time = np.nan
        maximum_lead_time = np.nan
    else:
        mean_lead_time = float(
            detected_df[
                "Lead time"
            ].mean()
        )

        median_lead_time = float(
            detected_df[
                "Lead time"
            ].median()
        )

        minimum_lead_time = float(
            detected_df[
                "Lead time"
            ].min()
        )

        maximum_lead_time = float(
            detected_df[
                "Lead time"
            ].max()
        )

    return {
        "Horizon": int(
            engine_warning_df[
                "Horizon"
            ].iloc[0]
        ),
        "Persistence rule": (
            engine_warning_df[
                "Persistence rule"
            ].iloc[0]
        ),
        "Total engines": total_engines,
        "Detected engines": (
            total_engines - missed_engines
        ),
        "Missed engines": missed_engines,
        "Miss rate": (
            missed_engines
            / total_engines
        ),
        "Mean lead time": mean_lead_time,
        "Median lead time": median_lead_time,
        "Minimum lead time": minimum_lead_time,
        "Maximum lead time": maximum_lead_time,
        "Mean late-warning delay": float(
            engine_warning_df[
                "Late-warning delay"
            ].mean()
        ),
        "Mean early-warning burden": float(
            engine_warning_df[
                "Early-warning burden"
            ].mean()
        ),
    }

def wilson_proportion_interval(
    successes: int,
    total: int,
    confidence_level: float = 0.95,
) -> tuple[float, float]:
    """
    Calculate a Wilson confidence interval for a proportion.

    This is more reliable than a normal interval when the
    sample is small or the observed proportion is 0 or 1.
    """
    if total <= 0:
        raise ValueError(
            "total must be positive."
        )

    if not 0 <= successes <= total:
        raise ValueError(
            "successes must be between 0 and total."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    alpha = 1.0 - confidence_level

    z_value = NormalDist().inv_cdf(
        1.0 - alpha / 2.0
    )

    proportion = successes / total

    denominator = (
        1.0
        + z_value ** 2 / total
    )

    center = (
        proportion
        + z_value ** 2
        / (2.0 * total)
    ) / denominator

    half_width = (
        z_value
        * np.sqrt(
            (
                proportion
                * (1.0 - proportion)
                / total
            )
            + (
                z_value ** 2
                / (4.0 * total ** 2)
            )
        )
        / denominator
    )

    lower_bound = max(
        0.0,
        center - half_width,
    )

    upper_bound = min(
        1.0,
        center + half_width,
    )

    return (
        float(lower_bound),
        float(upper_bound),
    )


def bootstrap_early_warning_intervals(
    engine_warning_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate engine-level confidence intervals for
    persistent early-warning metrics.

    Engines are sampled with replacement. Since the input
    contains one row per engine, this is an engine-level
    bootstrap rather than a row-level bootstrap.
    """
    required_columns = {
        "Lead time",
        "Missed warning",
        "Late-warning delay",
        "Early-warning burden",
    }

    missing_columns = (
        required_columns.difference(
            engine_warning_df.columns
        )
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if len(engine_warning_df) < 2:
        raise ValueError(
            "At least two engines are required."
        )

    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    rng = np.random.default_rng(
        random_state
    )

    engine_warning_df = (
        engine_warning_df
        .reset_index(drop=True)
        .copy()
    )

    sample_count = len(
        engine_warning_df
    )

    alpha = (
        1.0 - confidence_level
    )

    lower_quantile = (
        alpha / 2.0
    )

    upper_quantile = (
        1.0 - alpha / 2.0
    )

    def calculate_metrics(
        sample_df: pd.DataFrame,
    ) -> dict[str, float]:
        detected_df = sample_df.loc[
            sample_df[
                "Missed warning"
            ] == 0
        ]

        return {
            "Mean lead time": float(
                detected_df[
                    "Lead time"
                ].mean()
            )
            if not detected_df.empty
            else np.nan,

            "Median lead time": float(
                detected_df[
                    "Lead time"
                ].median()
            )
            if not detected_df.empty
            else np.nan,

            "Mean late-warning delay": float(
                sample_df[
                    "Late-warning delay"
                ].mean()
            ),

            "Mean early-warning burden": float(
                sample_df[
                    "Early-warning burden"
                ].mean()
            ),
        }

    point_estimates = calculate_metrics(
        engine_warning_df
    )

    bootstrap_values = {
        metric_name: []
        for metric_name in point_estimates
    }

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(
            low=0,
            high=sample_count,
            size=sample_count,
        )

        bootstrap_sample = (
            engine_warning_df.iloc[
                sampled_indices
            ]
        )

        sampled_metrics = (
            calculate_metrics(
                bootstrap_sample
            )
        )

        for (
            metric_name,
            metric_value,
        ) in sampled_metrics.items():
            if np.isfinite(
                metric_value
            ):
                bootstrap_values[
                    metric_name
                ].append(
                    metric_value
                )

    result_rows = []

    for (
        metric_name,
        estimate,
    ) in point_estimates.items():
        values = np.asarray(
            bootstrap_values[
                metric_name
            ],
            dtype=float,
        )

        result_rows.append(
            {
                "Metric": metric_name,
                "Estimate": estimate,
                "CI lower": float(
                    np.quantile(
                        values,
                        lower_quantile,
                    )
                ),
                "CI upper": float(
                    np.quantile(
                        values,
                        upper_quantile,
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

    missed_engines = int(
        engine_warning_df[
            "Missed warning"
        ].sum()
    )

    miss_rate_lower, miss_rate_upper = (
        wilson_proportion_interval(
            successes=missed_engines,
            total=sample_count,
            confidence_level=(
                confidence_level
            ),
        )
    )

    result_rows.append(
        {
            "Metric": "Miss rate",
            "Estimate": (
                missed_engines
                / sample_count
            ),
            "CI lower": miss_rate_lower,
            "CI upper": miss_rate_upper,
            "Confidence level": (
                confidence_level
            ),
            "Method": (
                "Wilson interval"
            ),
        }
    )

    return pd.DataFrame(
        result_rows
    )