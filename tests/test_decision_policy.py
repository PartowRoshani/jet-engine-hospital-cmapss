import numpy as np
import pandas as pd
import pytest

from src.decision_policy import (
    DecisionPolicyConfig,
    add_persistent_signal,
    apply_maintenance_policy,
    assign_reference_action,
    bootstrap_decision_policy_metrics,
    bootstrap_decision_timing_metrics,
    calculate_decision_cost,
    enforce_action_hysteresis,
    evaluate_decision_policy,
    validate_decision_inputs,
)


def make_valid_evidence() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5, 6],
            "RUL": [40, 35, 30, 20, 10, 5],
            "RUL prediction": [
                42.0, 37.0, 31.0,
                22.0, 11.0, 6.0,
            ],
            "RUL lower": [
                30.0, 25.0, 20.0,
                10.0, 2.0, 0.0,
            ],
            "RUL upper": [
                54.0, 49.0, 42.0,
                34.0, 23.0, 18.0,
            ],
            "probability_10": [
                0.01, 0.02, 0.05,
                0.10, 0.40, 0.50,
            ],
            "probability_20": [
                0.02, 0.05, 0.10,
                0.35, 0.55, 0.70,
            ],
            "probability_30": [
                0.05, 0.10, 0.30,
                0.50, 0.75, 0.90,
            ],
            "anomaly_percentile": [
                0.10, 0.20, 0.30,
                0.80, 0.9999, 1.00,
            ],
        }
    )


def test_assign_reference_action_boundaries():
    rul = np.array(
        [31.0, 30.0, 11.0, 10.0, 0.0]
    )

    actions = assign_reference_action(rul)

    assert actions.tolist() == [
        "CONTINUE",
        "INSPECT",
        "INSPECT",
        "STOP",
        "STOP",
    ]


def test_validate_decision_inputs_accepts_valid_data():
    evidence = make_valid_evidence()

    validate_decision_inputs(evidence)


def test_validate_decision_inputs_rejects_bad_hierarchy():
    evidence = make_valid_evidence()

    evidence.loc[
        0,
        "probability_10",
    ] = 0.50

    evidence.loc[
        0,
        "probability_20",
    ] = 0.20

    with pytest.raises(
        ValueError,
        match="P10 <= P20 <= P30",
    ):
        validate_decision_inputs(evidence)


def test_persistent_signal_is_engine_specific():
    data = pd.DataFrame(
        {
            "engine_id": [
                1, 1, 1,
                2, 2, 2,
            ],
            "cycle": [
                1, 2, 3,
                1, 2, 3,
            ],
            "alert": [
                1, 1, 0,
                1, 0, 0,
            ],
        }
    )

    result = add_persistent_signal(
        data=data,
        signal_column="alert",
        output_column="persistent",
        alerts_required=2,
        window_size=3,
    )

    engine_1 = result.loc[
        result["engine_id"] == 1,
        "persistent",
    ].tolist()

    engine_2 = result.loc[
        result["engine_id"] == 2,
        "persistent",
    ].tolist()

    assert engine_1 == [0, 0, 1]
    assert engine_2 == [0, 0, 0]


def test_supervised_policy_creates_valid_actions():
    evidence = make_valid_evidence()

    policy_df = apply_maintenance_policy(
        evidence_df=evidence,
        policy_name="supervised_only",
        config=DecisionPolicyConfig(),
    )

    assert set(
        policy_df["Action"]
    ).issubset(
        {
            "CONTINUE",
            "INSPECT",
            "STOP",
        }
    )

    assert (
        policy_df["Policy"]
        == "supervised_only"
    ).all()


def test_anomaly_alone_does_not_trigger_stop():
    evidence = make_valid_evidence()

    evidence[
        "probability_10"
    ] = 0.0

    evidence[
        "probability_20"
    ] = 0.0

    evidence[
        "probability_30"
    ] = 0.0

    evidence[
        "anomaly_percentile"
    ] = 1.0

    policy_df = apply_maintenance_policy(
        evidence_df=evidence,
        policy_name="full_fusion",
        config=DecisionPolicyConfig(),
    )

    assert not (
        policy_df["Action"] == "STOP"
    ).any()

    assert (
        policy_df["Action"] == "INSPECT"
    ).any()


def test_hysteresis_prevents_action_downgrade():
    policy_df = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1],
            "cycle": [1, 2, 3, 4],
            "Action": [
                "CONTINUE",
                "INSPECT",
                "CONTINUE",
                "STOP",
            ],
            "Action level": [0, 1, 0, 2],
            "Trigger": [
                "None",
                "Risk",
                "None",
                "High risk",
            ],
            "Confidence": [
                "MEDIUM",
                "HIGH",
                "MEDIUM",
                "HIGH",
            ],
            "Next review cycles": [
                10, 1, 10, 0,
            ],
        }
    )

    result = enforce_action_hysteresis(
        policy_df
    )

    assert result["Action"].tolist() == [
        "CONTINUE",
        "INSPECT",
        "INSPECT",
        "STOP",
    ]

    assert result[
        "Action level"
    ].is_monotonic_increasing

    assert result.loc[
        2,
        "Hysteresis applied",
    ]


def test_decision_cost_is_asymmetric():
    actual = np.array(
        [
            "STOP",
            "CONTINUE",
            "INSPECT",
        ]
    )

    predicted = np.array(
        [
            "CONTINUE",
            "INSPECT",
            "STOP",
        ]
    )

    result = calculate_decision_cost(
        actual_action=actual,
        predicted_action=predicted,
    )

    assert result[
        "Decision cost"
    ].tolist() == [
        100.0,
        5.0,
        10.0,
    ]

    assert result[
        "Unsafe miss"
    ].sum() == 1


def test_evaluate_decision_policy():
    evidence = make_valid_evidence()

    policy_df = apply_maintenance_policy(
        evidence_df=evidence,
        policy_name="supervised_only",
    )

    policy_df = enforce_action_hysteresis(
        policy_df
    )

    summary_df, confusion_df = (
        evaluate_decision_policy(
            policy_df=policy_df,
            actual_rul_column="RUL",
        )
    )

    assert len(summary_df) == 1

    assert np.isfinite(
        summary_df["Average cost"].iloc[0]
    )

    assert confusion_df.shape == (3, 3)

    assert confusion_df.to_numpy().sum() == len(
        policy_df
    )


def test_decision_metric_bootstrap():
    evidence = pd.concat(
        [
            make_valid_evidence().assign(
                engine_id=1
            ),
            make_valid_evidence().assign(
                engine_id=2
            ),
        ],
        ignore_index=True,
    )

    policy_df = apply_maintenance_policy(
        evidence_df=evidence,
        policy_name="supervised_only",
    )

    policy_df = enforce_action_hysteresis(
        policy_df
    )

    result = bootstrap_decision_policy_metrics(
        policy_df=policy_df,
        n_bootstrap=100,
        confidence_level=0.95,
        random_state=42,
    )

    assert {
        "Metric",
        "Estimate",
        "CI lower",
        "CI upper",
    }.issubset(result.columns)

    assert np.isfinite(
        result[
            [
                "Estimate",
                "CI lower",
                "CI upper",
            ]
        ].to_numpy()
    ).all()


def test_decision_timing_bootstrap():
    timing_df = pd.DataFrame(
        {
            "engine_id": [1, 2, 3],
            "INSPECT lead time": [
                30.0, 35.0, 25.0
            ],
            "INSPECT missed": [0, 0, 0],
            "INSPECT late delay": [
                0.0, 0.0, 5.0
            ],
            "INSPECT early burden": [
                0.0, 5.0, 0.0
            ],
            "STOP lead time": [
                10.0, 12.0, 8.0
            ],
            "STOP missed": [0, 0, 0],
            "STOP late delay": [
                0.0, 0.0, 2.0
            ],
            "STOP early burden": [
                0.0, 2.0, 0.0
            ],
        }
    )

    result = bootstrap_decision_timing_metrics(
        timing_df=timing_df,
        n_bootstrap=100,
        confidence_level=0.95,
        random_state=42,
    )

    assert len(result) == 10

    assert np.isfinite(
        result[
            [
                "Estimate",
                "CI lower",
                "CI upper",
            ]
        ].to_numpy()
    ).all()