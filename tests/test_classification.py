import numpy as np
import pandas as pd
import pytest

from src.classification import (
    evaluate_binary_classifier,
    make_failure_target,
)
from src.evaluation import (
    evaluate_persistent_early_warnings,
    wilson_proportion_interval,
)


def test_make_failure_target():
    rul = np.array([40, 30, 20, 10, 0])

    target = make_failure_target(
        rul,
        horizon=20,
    )

    expected = np.array(
        [0, 0, 1, 1, 1]
    )

    np.testing.assert_array_equal(
        target,
        expected,
    )


def test_failure_targets_are_nested():
    rul = np.arange(0, 50)

    target_10 = make_failure_target(
        rul,
        horizon=10,
    )

    target_20 = make_failure_target(
        rul,
        horizon=20,
    )

    target_30 = make_failure_target(
        rul,
        horizon=30,
    )

    assert np.all(
        target_10 <= target_20
    )

    assert np.all(
        target_20 <= target_30
    )


def test_invalid_failure_horizon():
    with pytest.raises(ValueError):
        make_failure_target(
            np.array([1, 2, 3]),
            horizon=0,
        )


def test_binary_classifier_metrics():
    y_true = np.array(
        [0, 0, 1, 1]
    )

    probabilities = np.array(
        [0.1, 0.7, 0.8, 0.2]
    )

    metrics = evaluate_binary_classifier(
        model_name="test",
        horizon=10,
        y_true=y_true,
        probabilities=probabilities,
        threshold=0.5,
    )

    assert metrics[
        "True negatives"
    ] == 1

    assert metrics[
        "False positives"
    ] == 1

    assert metrics[
        "False negatives"
    ] == 1

    assert metrics[
        "True positives"
    ] == 1


def test_persistent_warning_two_of_three():
    risk_df = pd.DataFrame(
        {
            "engine_id": [
                1, 1, 1, 1, 1
            ],
            "cycle": [
                1, 2, 3, 4, 5
            ],
            "RUL": [
                4, 3, 2, 1, 0
            ],
            "alert_10": [
                0, 1, 0, 1, 1
            ],
        }
    )

    result = (
        evaluate_persistent_early_warnings(
            risk_df=risk_df,
            horizon=10,
            alert_column="alert_10",
            alerts_required=2,
            window_size=3,
        )
    )

    assert len(result) == 1

    assert result.loc[
        0,
        "First persistent alert cycle",
    ] == 4

    assert result.loc[
        0,
        "Lead time",
    ] == 1

    assert result.loc[
        0,
        "Missed warning",
    ] == 0


def test_wilson_interval_for_zero_misses():
    lower, upper = (
        wilson_proportion_interval(
            successes=0,
            total=15,
            confidence_level=0.95,
        )
    )

    assert lower == pytest.approx(
        0.0
    )

    assert upper == pytest.approx(
        0.2039,
        abs=0.001,
    )