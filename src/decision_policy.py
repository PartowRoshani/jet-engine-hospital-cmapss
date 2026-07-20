from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


ACTION_ORDER = {
    "CONTINUE": 0,
    "INSPECT": 1,
    "STOP": 2,
}


@dataclass(frozen=True)
class DecisionPolicyConfig:
    """
    Configuration for the maintenance decision policy.
    """

    probability_threshold_10: float = 0.30
    probability_threshold_20: float = 0.29
    probability_threshold_30: float = 0.27

    anomaly_threshold: float = 0.9999

    persistence_alerts_required: int = 2
    persistence_window_size: int = 3

    stop_rul_boundary: float = 10.0
    inspect_rul_boundary: float = 30.0


def validate_decision_inputs(
    evidence_df: pd.DataFrame,
) -> None:
    """
    Validate the evidence table used by the policy.
    """
    required_columns = {
        "engine_id",
        "cycle",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_percentile",
    }

    missing_columns = required_columns.difference(
        evidence_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing decision-policy columns: "
            f"{sorted(missing_columns)}"
        )

    numeric_columns = [
        "cycle",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_percentile",
    ]

    numeric_values = evidence_df[
        numeric_columns
    ].to_numpy(dtype=float)

    if not np.isfinite(numeric_values).all():
        raise ValueError(
            "Decision evidence contains NaN "
            "or infinite values."
        )

    probability_columns = [
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_percentile",
    ]

    for column in probability_columns:
        if not evidence_df[column].between(
            0.0,
            1.0,
            inclusive="both",
        ).all():
            raise ValueError(
                f"{column} must be between 0 and 1."
            )

    probability_hierarchy_valid = (
        (
            evidence_df["probability_10"]
            <= evidence_df["probability_20"]
        )
        & (
            evidence_df["probability_20"]
            <= evidence_df["probability_30"]
        )
    )

    if not probability_hierarchy_valid.all():
        raise ValueError(
            "Failure probabilities must satisfy "
            "P10 <= P20 <= P30."
        )

    interval_valid = (
        evidence_df["RUL lower"]
        <= evidence_df["RUL upper"]
    )

    if not interval_valid.all():
        raise ValueError(
            "RUL lower must not exceed RUL upper."
        )


def add_persistent_signal(
    data: pd.DataFrame,
    signal_column: str,
    output_column: str,
    alerts_required: int = 2,
    window_size: int = 3,
) -> pd.DataFrame:
    """
    Add a persistent binary signal independently
    within each engine.

    A signal becomes persistent when at least
    alerts_required positive values occur in the
    latest window_size cycles.
    """
    required_columns = {
        "engine_id",
        "cycle",
        signal_column,
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing persistence columns: "
            f"{sorted(missing_columns)}"
        )

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
            "alerts_required cannot exceed "
            "window_size."
        )

    result = (
        data
        .sort_values(
            ["engine_id", "cycle"]
        )
        .copy()
    )

    signal_values = (
        result[signal_column]
        .astype(int)
    )

    rolling_count = (
        signal_values
        .groupby(
            result["engine_id"]
        )
        .rolling(
            window=window_size,
            min_periods=window_size,
        )
        .sum()
        .reset_index(
            level=0,
            drop=True,
        )
    )

    result[output_column] = (
        rolling_count >= alerts_required
    ).fillna(False).astype(int)

    return result


def prepare_decision_signals(
    evidence_df: pd.DataFrame,
    config: DecisionPolicyConfig,
) -> pd.DataFrame:
    """
    Convert probabilities and anomaly percentiles
    into raw and persistent alert signals.
    """
    validate_decision_inputs(
        evidence_df
    )

    result = (
        evidence_df
        .sort_values(
            ["engine_id", "cycle"]
        )
        .copy()
        .reset_index(drop=True)
    )

    result["risk_alert_10"] = (
        result["probability_10"]
        >= config.probability_threshold_10
    ).astype(int)

    result["risk_alert_20"] = (
        result["probability_20"]
        >= config.probability_threshold_20
    ).astype(int)

    result["risk_alert_30"] = (
        result["probability_30"]
        >= config.probability_threshold_30
    ).astype(int)

    result["anomaly_alert"] = (
        result["anomaly_percentile"]
        >= config.anomaly_threshold
    ).astype(int)

    persistence_arguments = {
        "alerts_required": (
            config.persistence_alerts_required
        ),
        "window_size": (
            config.persistence_window_size
        ),
    }

    result = add_persistent_signal(
        data=result,
        signal_column="risk_alert_10",
        output_column="persistent_risk_10",
        **persistence_arguments,
    )

    result = add_persistent_signal(
        data=result,
        signal_column="risk_alert_20",
        output_column="persistent_risk_20",
        **persistence_arguments,
    )

    result = add_persistent_signal(
        data=result,
        signal_column="risk_alert_30",
        output_column="persistent_risk_30",
        **persistence_arguments,
    )

    result = add_persistent_signal(
        data=result,
        signal_column="anomaly_alert",
        output_column="persistent_anomaly",
        **persistence_arguments,
    )

    result["interval_crosses_10"] = (
        (
            result["RUL lower"]
            <= config.stop_rul_boundary
        )
        & (
            result["RUL upper"]
            > config.stop_rul_boundary
        )
    ).astype(int)

    result["interval_crosses_30"] = (
        (
            result["RUL lower"]
            <= config.inspect_rul_boundary
        )
        & (
            result["RUL upper"]
            > config.inspect_rul_boundary
        )
    ).astype(int)

    result["supervised_signal_count"] = (
        result[
            [
                "persistent_risk_10",
                "persistent_risk_20",
                "persistent_risk_30",
            ]
        ]
        .sum(axis=1)
    )

    result["total_signal_count"] = (
        result["supervised_signal_count"]
        + result["persistent_anomaly"]
    )

    return result


def assign_reference_action(
    rul: pd.Series | np.ndarray,
    stop_boundary: float = 10.0,
    inspect_boundary: float = 30.0,
) -> np.ndarray:
    """
    Convert actual RUL into the operational reference
    action used only for evaluation.
    """
    rul_array = np.asarray(
        rul,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(rul_array).all():
        raise ValueError(
            "RUL contains NaN or infinity."
        )

    if stop_boundary >= inspect_boundary:
        raise ValueError(
            "stop_boundary must be smaller than "
            "inspect_boundary."
        )

    return np.select(
        condlist=[
            rul_array <= stop_boundary,
            rul_array <= inspect_boundary,
        ],
        choicelist=[
            "STOP",
            "INSPECT",
        ],
        default="CONTINUE",
    )


def apply_maintenance_policy(
    evidence_df: pd.DataFrame,
    policy_name: str,
    config: DecisionPolicyConfig | None = None,
) -> pd.DataFrame:
    """
    Apply one of three documented decision policies.

    Supported policies:
    - supervised_only
    - uncertainty_aware
    - full_fusion
    """
    if config is None:
        config = DecisionPolicyConfig()

    supported_policies = {
        "supervised_only",
        "uncertainty_aware",
        "full_fusion",
    }

    if policy_name not in supported_policies:
        raise ValueError(
            "policy_name must be one of "
            f"{sorted(supported_policies)}"
        )

    result = prepare_decision_signals(
        evidence_df=evidence_df,
        config=config,
    )

    persistent_10 = (
        result["persistent_risk_10"] == 1
    )

    persistent_20 = (
        result["persistent_risk_20"] == 1
    )

    persistent_30 = (
        result["persistent_risk_30"] == 1
    )

    persistent_anomaly = (
        result["persistent_anomaly"] == 1
    )

    low_rul_10 = (
        result["RUL lower"]
        <= config.stop_rul_boundary
    )

    low_rul_30 = (
        result["RUL lower"]
        <= config.inspect_rul_boundary
    )

    if policy_name == "supervised_only":
        stop_condition = persistent_10

        inspect_condition = (
            ~stop_condition
            & (
                persistent_20
                | persistent_30
            )
        )

    elif policy_name == "uncertainty_aware":
        stop_condition = (
            persistent_10
            | (
                persistent_20
                & low_rul_10
            )
        )

        inspect_condition = (
            ~stop_condition
            & (
                persistent_20
                | persistent_30
                | (
                    low_rul_30
                    & (
                        result["probability_30"]
                        >= (
                            0.5
                            * config
                            .probability_threshold_30
                        )
                    )
                )
            )
        )

    else:
        stop_condition = (
            persistent_10
            | (
                persistent_20
                & low_rul_10
            )
        )

        inspect_condition = (
            ~stop_condition
            & (
                persistent_20
                | persistent_30
                | persistent_anomaly
                | (
                    low_rul_30
                    & (
                        result["probability_30"]
                        >= (
                            0.5
                            * config
                            .probability_threshold_30
                        )
                    )
                )
            )
        )

    result["Action"] = np.select(
        condlist=[
            stop_condition,
            inspect_condition,
        ],
        choicelist=[
            "STOP",
            "INSPECT",
        ],
        default="CONTINUE",
    )

    result["Action level"] = (
        result["Action"]
        .map(ACTION_ORDER)
        .astype(int)
    )

    result["Signal disagreement"] = (
        (
            persistent_anomaly
            & ~persistent_30
        )
        | (
            persistent_30
            & ~persistent_anomaly
        )
    ).astype(int)

    result["Confidence"] = np.select(
        condlist=[
            (
                stop_condition
                & persistent_10
                & low_rul_10
            ),
            (
                inspect_condition
                & (
                    result[
                        "total_signal_count"
                    ] >= 2
                )
            ),
            (
                result[
                    "Signal disagreement"
                ] == 1
            ),
        ],
        choicelist=[
            "HIGH",
            "HIGH",
            "LOW",
        ],
        default="MEDIUM",
    )

    result["Trigger"] = "No validated risk rule fired"

    result.loc[
        persistent_anomaly
        & ~persistent_30,
        "Trigger",
    ] = (
        "Persistent anomaly with low supervised risk"
    )

    result.loc[
        persistent_30
        & ~persistent_anomaly,
        "Trigger",
    ] = (
        "Persistent 30-cycle supervised risk"
    )

    result.loc[
        persistent_20,
        "Trigger",
    ] = (
        "Persistent 20-cycle failure risk"
    )

    result.loc[
        persistent_20 & low_rul_10,
        "Trigger",
    ] = (
        "20-cycle risk confirmed by low RUL bound"
    )

    result.loc[
        persistent_10,
        "Trigger",
    ] = (
        "Persistent validated 10-cycle failure risk"
    )

    result.loc[
        (
            result["Action"] == "CONTINUE"
        ),
        "Next review cycles",
    ] = 10

    result.loc[
        (
            result["Action"] == "INSPECT"
        ),
        "Next review cycles",
    ] = 1

    result.loc[
        (
            result["Action"] == "STOP"
        ),
        "Next review cycles",
    ] = 0

    result["Next review cycles"] = (
        result["Next review cycles"]
        .astype(int)
    )

    result["Policy"] = policy_name

    return result


def calculate_decision_cost(
    actual_action: pd.Series | np.ndarray,
    predicted_action: pd.Series | np.ndarray,
) -> pd.DataFrame:
    """
    Calculate an asymmetric operational cost.

    Missing a dangerous engine is much more expensive
    than performing an unnecessary inspection.
    """
    actual = np.asarray(
        actual_action,
        dtype=str,
    ).reshape(-1)

    predicted = np.asarray(
        predicted_action,
        dtype=str,
    ).reshape(-1)

    if len(actual) != len(predicted):
        raise ValueError(
            "Actual and predicted actions must have "
            "the same length."
        )

    cost_matrix = {
        ("CONTINUE", "CONTINUE"): 0.0,
        ("CONTINUE", "INSPECT"): 5.0,
        ("CONTINUE", "STOP"): 15.0,

        ("INSPECT", "CONTINUE"): 40.0,
        ("INSPECT", "INSPECT"): 0.0,
        ("INSPECT", "STOP"): 10.0,

        ("STOP", "CONTINUE"): 100.0,
        ("STOP", "INSPECT"): 30.0,
        ("STOP", "STOP"): 0.0,
    }

    rows = []

    for actual_value, predicted_value in zip(
        actual,
        predicted,
        strict=True,
    ):
        key = (
            actual_value,
            predicted_value,
        )

        if key not in cost_matrix:
            raise ValueError(
                f"Unsupported action pair: {key}"
            )

        cost = cost_matrix[key]

        rows.append(
            {
                "Actual action": actual_value,
                "Predicted action": predicted_value,
                "Decision cost": cost,
                "Unsafe miss": int(
                    actual_value == "STOP"
                    and predicted_value
                    == "CONTINUE"
                ),
                "Delayed inspection": int(
                    actual_value == "INSPECT"
                    and predicted_value
                    == "CONTINUE"
                ),
                "Unnecessary inspection": int(
                    actual_value == "CONTINUE"
                    and predicted_value
                    == "INSPECT"
                ),
                "Unnecessary stop": int(
                    actual_value == "CONTINUE"
                    and predicted_value
                    == "STOP"
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_decision_policy(
    policy_df: pd.DataFrame,
    actual_rul_column: str = "RUL",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a decision policy with an asymmetric cost
    and a CONTINUE/INSPECT/STOP confusion matrix.
    """
    required_columns = {
        actual_rul_column,
        "Action",
        "Policy",
    }

    missing_columns = required_columns.difference(
        policy_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing policy evaluation columns: "
            f"{sorted(missing_columns)}"
        )

    actual_actions = assign_reference_action(
        rul=policy_df[actual_rul_column]
    )

    cost_df = calculate_decision_cost(
        actual_action=actual_actions,
        predicted_action=policy_df["Action"],
    )

    summary_df = pd.DataFrame(
        [
            {
                "Policy": (
                    policy_df["Policy"].iloc[0]
                ),
                "Samples": len(policy_df),
                "Average cost": float(
                    cost_df[
                        "Decision cost"
                    ].mean()
                ),
                "Total cost": float(
                    cost_df[
                        "Decision cost"
                    ].sum()
                ),
                "Unsafe misses": int(
                    cost_df[
                        "Unsafe miss"
                    ].sum()
                ),
                "Delayed inspections": int(
                    cost_df[
                        "Delayed inspection"
                    ].sum()
                ),
                "Unnecessary inspections": int(
                    cost_df[
                        "Unnecessary inspection"
                    ].sum()
                ),
                "Unnecessary stops": int(
                    cost_df[
                        "Unnecessary stop"
                    ].sum()
                ),
                "STOP recall": float(
                    (
                        cost_df.loc[
                            cost_df[
                                "Actual action"
                            ] == "STOP",
                            "Predicted action",
                        ]
                        == "STOP"
                    ).mean()
                ),
                "INSPECT-or-higher recall": float(
                    (
                        cost_df.loc[
                            cost_df[
                                "Actual action"
                            ].isin(
                                [
                                    "INSPECT",
                                    "STOP",
                                ]
                            ),
                            "Predicted action",
                        ]
                        != "CONTINUE"
                    ).mean()
                ),
            }
        ]
    )

    confusion_df = pd.crosstab(
        pd.Series(
            actual_actions,
            name="Actual action",
        ),
        pd.Series(
            policy_df[
                "Action"
            ].to_numpy(),
            name="Predicted action",
        ),
        dropna=False,
    )

    confusion_df = confusion_df.reindex(
        index=[
            "CONTINUE",
            "INSPECT",
            "STOP",
        ],
        columns=[
            "CONTINUE",
            "INSPECT",
            "STOP",
        ],
        fill_value=0,
    )

    return summary_df, confusion_df
def enforce_action_hysteresis(
    policy_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prevent maintenance actions from becoming less severe
    within an engine trajectory.

    Without an explicit maintenance/reset event, actions may
    remain unchanged or escalate:

        CONTINUE -> INSPECT -> STOP

    They may not move backward.
    """
    required_columns = {
        "engine_id",
        "cycle",
        "Action",
        "Action level",
        "Trigger",
        "Confidence",
        "Next review cycles",
    }

    missing_columns = required_columns.difference(
        policy_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing hysteresis columns: "
            f"{sorted(missing_columns)}"
        )

    result = (
        policy_df
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .copy()
        .reset_index(drop=True)
    )

    valid_actions = set(
        ACTION_ORDER
    )

    if not set(
        result["Action"].unique()
    ).issubset(valid_actions):
        raise ValueError(
            "Unsupported action found in policy_df."
        )

    reverse_action_order = {
        level: action
        for action, level in ACTION_ORDER.items()
    }

    result["Raw Action"] = result["Action"]
    result["Raw Action level"] = (
        result["Action level"]
    )
    result["Raw Trigger"] = result["Trigger"]
    result["Raw Confidence"] = (
        result["Confidence"]
    )

    result["Hysteresis applied"] = False

    review_cycle_mapping = {
        "CONTINUE": 10,
        "INSPECT": 1,
        "STOP": 0,
    }

    for engine_id, engine_indices in (
        result
        .groupby(
            "engine_id",
            sort=False,
        )
        .groups
        .items()
    ):
        retained_level = -1
        retained_trigger = ""
        retained_confidence = "MEDIUM"

        for row_index in engine_indices:
            raw_level = int(
                result.at[
                    row_index,
                    "Raw Action level",
                ]
            )

            raw_trigger = str(
                result.at[
                    row_index,
                    "Raw Trigger",
                ]
            )

            raw_confidence = str(
                result.at[
                    row_index,
                    "Raw Confidence",
                ]
            )

            if raw_level >= retained_level:
                retained_level = raw_level
                retained_trigger = raw_trigger
                retained_confidence = (
                    raw_confidence
                )
                hysteresis_applied = False
            else:
                hysteresis_applied = True

            retained_action = (
                reverse_action_order[
                    retained_level
                ]
            )

            result.at[
                row_index,
                "Action level",
            ] = retained_level

            result.at[
                row_index,
                "Action",
            ] = retained_action

            result.at[
                row_index,
                "Hysteresis applied",
            ] = hysteresis_applied

            if hysteresis_applied:
                result.at[
                    row_index,
                    "Trigger",
                ] = (
                    f"Retained {retained_action} "
                    "by hysteresis after: "
                    f"{retained_trigger}"
                )

                result.at[
                    row_index,
                    "Confidence",
                ] = retained_confidence

            result.at[
                row_index,
                "Next review cycles",
            ] = review_cycle_mapping[
                retained_action
            ]

    result["Action level"] = (
        result["Action level"]
        .astype(int)
    )

    result["Next review cycles"] = (
        result["Next review cycles"]
        .astype(int)
    )

    result["Hysteresis applied"] = (
        result["Hysteresis applied"]
        .astype(bool)
    )

    return result
def bootstrap_decision_policy_metrics(
    policy_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate engine-bootstrap confidence intervals for
    row-level maintenance decision metrics.
    """
    required_columns = {
        "engine_id",
        "RUL",
        "Action",
        "Policy",
    }

    missing_columns = required_columns.difference(
        policy_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing decision bootstrap columns: "
            f"{sorted(missing_columns)}"
        )

    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    def calculate_metrics(
        sampled_df: pd.DataFrame,
    ) -> dict[str, float]:
        actual_actions = assign_reference_action(
            sampled_df["RUL"]
        )

        predicted_actions = (
            sampled_df["Action"]
            .astype(str)
            .to_numpy()
        )

        cost_df = calculate_decision_cost(
            actual_action=actual_actions,
            predicted_action=predicted_actions,
        )

        actual_stop = (
            actual_actions == "STOP"
        )

        actual_inspect = (
            actual_actions == "INSPECT"
        )

        actual_continue = (
            actual_actions == "CONTINUE"
        )

        actual_warning = np.isin(
            actual_actions,
            [
                "INSPECT",
                "STOP",
            ],
        )

        stop_recall = (
            float(
                (
                    predicted_actions[
                        actual_stop
                    ] == "STOP"
                ).mean()
            )
            if actual_stop.any()
            else np.nan
        )

        inspect_or_higher_recall = (
            float(
                (
                    predicted_actions[
                        actual_warning
                    ] != "CONTINUE"
                ).mean()
            )
            if actual_warning.any()
            else np.nan
        )

        unsafe_miss_rate = (
            float(
                (
                    predicted_actions[
                        actual_stop
                    ] == "CONTINUE"
                ).mean()
            )
            if actual_stop.any()
            else np.nan
        )

        delayed_inspection_rate = (
            float(
                (
                    predicted_actions[
                        actual_inspect
                    ] == "CONTINUE"
                ).mean()
            )
            if actual_inspect.any()
            else np.nan
        )

        unnecessary_inspection_rate = (
            float(
                (
                    predicted_actions[
                        actual_continue
                    ] == "INSPECT"
                ).mean()
            )
            if actual_continue.any()
            else np.nan
        )

        unnecessary_stop_rate = (
            float(
                (
                    predicted_actions[
                        actual_continue
                    ] == "STOP"
                ).mean()
            )
            if actual_continue.any()
            else np.nan
        )

        return {
            "Average decision cost": float(
                cost_df[
                    "Decision cost"
                ].mean()
            ),
            "STOP recall": stop_recall,
            "INSPECT-or-higher recall": (
                inspect_or_higher_recall
            ),
            "Unsafe miss rate": (
                unsafe_miss_rate
            ),
            "Delayed inspection rate": (
                delayed_inspection_rate
            ),
            "Unnecessary inspection rate": (
                unnecessary_inspection_rate
            ),
            "Unnecessary stop rate": (
                unnecessary_stop_rate
            ),
        }

    point_estimates = calculate_metrics(
        policy_df
    )

    engine_ids = (
        policy_df["engine_id"]
        .drop_duplicates()
        .to_numpy()
    )

    if len(engine_ids) < 2:
        raise ValueError(
            "At least two engines are required."
        )

    engine_frames = {
        engine_id: policy_df.loc[
            policy_df["engine_id"]
            == engine_id
        ]
        for engine_id in engine_ids
    }

    bootstrap_values = {
        metric_name: []
        for metric_name in point_estimates
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

        sampled_metrics = calculate_metrics(
            sampled_df
        )

        for metric_name, metric_value in (
            sampled_metrics.items()
        ):
            bootstrap_values[
                metric_name
            ].append(
                metric_value
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

        values = values[
            np.isfinite(values)
        ]

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


def bootstrap_decision_timing_metrics(
    timing_df: pd.DataFrame,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Calculate engine-bootstrap confidence intervals for
    INSPECT and STOP decision timing.
    """
    required_columns = {
        "engine_id",
        "INSPECT lead time",
        "INSPECT missed",
        "INSPECT late delay",
        "INSPECT early burden",
        "STOP lead time",
        "STOP missed",
        "STOP late delay",
        "STOP early burden",
    }

    missing_columns = required_columns.difference(
        timing_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing timing bootstrap columns: "
            f"{sorted(missing_columns)}"
        )

    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    def calculate_metrics(
        sampled_df: pd.DataFrame,
    ) -> dict[str, float]:
        inspect_detected_df = (
            sampled_df.loc[
                sampled_df[
                    "INSPECT missed"
                ] == 0
            ]
        )

        stop_detected_df = (
            sampled_df.loc[
                sampled_df[
                    "STOP missed"
                ] == 0
            ]
        )

        return {
            "INSPECT miss rate": float(
                sampled_df[
                    "INSPECT missed"
                ].mean()
            ),
            "Mean INSPECT lead time": float(
                inspect_detected_df[
                    "INSPECT lead time"
                ].mean()
            ),
            "Median INSPECT lead time": float(
                inspect_detected_df[
                    "INSPECT lead time"
                ].median()
            ),
            "Mean INSPECT late delay": float(
                sampled_df[
                    "INSPECT late delay"
                ].mean()
            ),
            "Mean INSPECT early burden": float(
                sampled_df[
                    "INSPECT early burden"
                ].mean()
            ),
            "STOP miss rate": float(
                sampled_df[
                    "STOP missed"
                ].mean()
            ),
            "Mean STOP lead time": float(
                stop_detected_df[
                    "STOP lead time"
                ].mean()
            ),
            "Median STOP lead time": float(
                stop_detected_df[
                    "STOP lead time"
                ].median()
            ),
            "Mean STOP late delay": float(
                sampled_df[
                    "STOP late delay"
                ].mean()
            ),
            "Mean STOP early burden": float(
                sampled_df[
                    "STOP early burden"
                ].mean()
            ),
        }

    point_estimates = calculate_metrics(
        timing_df
    )

    engine_rows = (
        timing_df
        .drop_duplicates(
            subset=["engine_id"]
        )
        .reset_index(drop=True)
    )

    if len(engine_rows) < 2:
        raise ValueError(
            "At least two engines are required."
        )

    bootstrap_values = {
        metric_name: []
        for metric_name in point_estimates
    }

    rng = np.random.default_rng(
        random_state
    )

    for _ in range(n_bootstrap):
        sampled_indices = rng.integers(
            low=0,
            high=len(engine_rows),
            size=len(engine_rows),
        )

        sampled_df = (
            engine_rows
            .iloc[sampled_indices]
            .reset_index(drop=True)
        )

        sampled_metrics = calculate_metrics(
            sampled_df
        )

        for metric_name, metric_value in (
            sampled_metrics.items()
        ):
            bootstrap_values[
                metric_name
            ].append(
                metric_value
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

        values = values[
            np.isfinite(values)
        ]

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