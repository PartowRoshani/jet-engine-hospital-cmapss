from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.bootstrap_evaluation import (
    bootstrap_fd001_interval_metrics,
    bootstrap_fd001_regression_metrics,
    calculate_fd001_interval_metrics,
    calculate_fd001_nasa_score,
    calculate_fd001_regression_metrics,
)


def make_regression_test_data() -> pd.DataFrame:
    """
    Create a small deterministic engine-level regression dataset.
    """
    return pd.DataFrame(
        {
            "engine_id": [
                1,
                1,
                1,
                2,
                2,
                2,
                3,
                3,
                3,
            ],
            "cycle": [
                1,
                2,
                3,
                1,
                2,
                3,
                1,
                2,
                3,
            ],
            "RUL": [
                40.0,
                20.0,
                0.0,
                50.0,
                25.0,
                0.0,
                60.0,
                30.0,
                0.0,
            ],
            "RUL prediction": [
                38.0,
                22.0,
                3.0,
                47.0,
                27.0,
                2.0,
                57.0,
                33.0,
                4.0,
            ],
        }
    )


def make_interval_test_data() -> pd.DataFrame:
    """
    Create deterministic conformal interval test data.
    """
    regression_df = (
        make_regression_test_data()
    )

    interval_df = (
        regression_df.copy()
    )

    interval_df[
        "RUL lower"
    ] = np.maximum(
        interval_df[
            "RUL prediction"
        ]
        - 10.0,
        0.0,
    )

    interval_df[
        "RUL upper"
    ] = (
        interval_df[
            "RUL prediction"
        ]
        + 10.0
    )

    return interval_df


def test_nasa_score_penalizes_late_predictions_more() -> None:
    actual = np.array(
        [
            50.0,
        ]
    )

    early_prediction = np.array(
        [
            40.0,
        ]
    )

    late_prediction = np.array(
        [
            60.0,
        ]
    )

    early_score = (
        calculate_fd001_nasa_score(
            actual,
            early_prediction,
        )
    )

    late_score = (
        calculate_fd001_nasa_score(
            actual,
            late_prediction,
        )
    )

    assert late_score > early_score


def test_regression_metrics_are_finite() -> None:
    metrics = (
        calculate_fd001_regression_metrics(
            make_regression_test_data()
        )
    )

    expected_metrics = {
        "MAE",
        "RMSE",
        "R2",
        "NASA score",
        "NASA penalty per row",
        "Near-failure MAE",
        "Late-prediction rate",
    }

    assert set(
        metrics
    ) == expected_metrics

    assert all(
        np.isfinite(
            value
        )
        for value in metrics.values()
    )

    assert metrics[
        "MAE"
    ] >= 0.0

    assert 0.0 <= metrics[
        "Late-prediction rate"
    ] <= 1.0


def test_interval_metrics_are_valid() -> None:
    metrics = (
        calculate_fd001_interval_metrics(
            make_interval_test_data()
        )
    )

    assert (
        metrics[
            "Overall coverage"
        ]
        == pytest.approx(
            1.0
        )
    )

    assert metrics[
        "Average interval width"
    ] > 0.0

    assert 0.0 <= metrics[
        "Coverage RUL 0-30"
    ] <= 1.0


def test_regression_bootstrap_is_reproducible() -> None:
    data = (
        make_regression_test_data()
    )

    first_result = (
        bootstrap_fd001_regression_metrics(
            evaluation_df=data,
            n_bootstrap=200,
            confidence_level=0.95,
            random_state=42,
        )
    )

    second_result = (
        bootstrap_fd001_regression_metrics(
            evaluation_df=data,
            n_bootstrap=200,
            confidence_level=0.95,
            random_state=42,
        )
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )

    assert (
        first_result[
            "Engine count"
        ]
        == 3
    ).all()

    assert (
        first_result[
            "CI lower"
        ]
        <= first_result[
            "CI upper"
        ]
    ).all()


def test_interval_bootstrap_is_reproducible() -> None:
    data = (
        make_interval_test_data()
    )

    first_result = (
        bootstrap_fd001_interval_metrics(
            interval_df=data,
            n_bootstrap=200,
            confidence_level=0.95,
            random_state=43,
        )
    )

    second_result = (
        bootstrap_fd001_interval_metrics(
            interval_df=data,
            n_bootstrap=200,
            confidence_level=0.95,
            random_state=43,
        )
    )

    pd.testing.assert_frame_equal(
        first_result,
        second_result,
    )


def test_bootstrap_rejects_too_few_samples() -> None:
    with pytest.raises(
        ValueError,
        match="at least 100",
    ):
        bootstrap_fd001_regression_metrics(
            evaluation_df=(
                make_regression_test_data()
            ),
            n_bootstrap=99,
        )


def test_bootstrap_requires_multiple_engines() -> None:
    single_engine_df = (
        make_regression_test_data()
        .loc[
            lambda frame:
            frame[
                "engine_id"
            ]
            == 1
        ]
        .copy()
    )

    with pytest.raises(
        ValueError,
        match="At least two engines",
    ):
        bootstrap_fd001_regression_metrics(
            evaluation_df=(
                single_engine_df
            ),
            n_bootstrap=100,
        )