import numpy as np
import pandas as pd
import pytest

from src.uncertainty import (
    bootstrap_interval_metrics,
    calculate_absolute_residuals,
    calculate_conformal_quantile,
    create_conformal_intervals,
    evaluate_interval_regions,
    evaluate_prediction_intervals,
)


def test_absolute_residuals():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 17.0, 30.0])

    residuals = calculate_absolute_residuals(
        y_true=y_true,
        y_pred=y_pred,
    )

    np.testing.assert_allclose(
        residuals,
        np.array([2.0, 3.0, 0.0]),
    )


def test_absolute_residual_length_error():
    with pytest.raises(ValueError):
        calculate_absolute_residuals(
            y_true=np.array([1.0, 2.0]),
            y_pred=np.array([1.0]),
        )


def test_conformal_quantile_finite_sample():
    scores = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0]
    )

    quantile = calculate_conformal_quantile(
        calibration_scores=scores,
        confidence_level=0.80,
    )

    # ceil((5 + 1) * 0.80) = 5
    assert quantile == pytest.approx(5.0)


def test_conformal_quantile_invalid_level():
    with pytest.raises(ValueError):
        calculate_conformal_quantile(
            calibration_scores=np.array(
                [1.0, 2.0]
            ),
            confidence_level=1.0,
        )


def test_create_conformal_intervals():
    predictions = np.array(
        [10.0, 50.0]
    )

    intervals = create_conformal_intervals(
        predictions=predictions,
        conformal_quantile=20.0,
        lower_bound=0.0,
    )

    np.testing.assert_allclose(
        intervals["RUL lower"],
        np.array([0.0, 30.0]),
    )

    np.testing.assert_allclose(
        intervals["RUL upper"],
        np.array([30.0, 70.0]),
    )

    np.testing.assert_allclose(
        intervals["Interval width"],
        np.array([30.0, 40.0]),
    )


def test_evaluate_prediction_intervals():
    y_true = np.array(
        [10.0, 20.0, 50.0]
    )

    interval_df = pd.DataFrame(
        {
            "RUL prediction": [
                10.0, 25.0, 30.0
            ],
            "RUL lower": [
                5.0, 15.0, 20.0
            ],
            "RUL upper": [
                15.0, 35.0, 40.0
            ],
            "Interval width": [
                10.0, 20.0, 20.0
            ],
        }
    )

    metrics = evaluate_prediction_intervals(
        y_true=y_true,
        interval_df=interval_df,
    )

    assert metrics[
        "Empirical coverage"
    ] == pytest.approx(2.0 / 3.0)

    assert metrics[
        "Upper misses"
    ] == 1

    assert metrics[
        "Lower misses"
    ] == 0

    assert metrics[
        "Total misses"
    ] == 1


def test_evaluate_interval_regions():
    evaluation_df = pd.DataFrame(
        {
            "RUL": [
                10.0, 50.0, 150.0
            ],
            "RUL lower": [
                0.0, 40.0, 100.0
            ],
            "RUL upper": [
                20.0, 60.0, 140.0
            ],
            "Interval width": [
                20.0, 20.0, 40.0
            ],
        }
    )

    regions = evaluate_interval_regions(
        evaluation_df=evaluation_df
    )

    assert len(regions) == 3

    near_coverage = regions.loc[
        regions["RUL region"]
        == "Near failure: 0–30",
        "Coverage",
    ].iloc[0]

    early_coverage = regions.loc[
        regions["RUL region"]
        == "Early life: >125",
        "Coverage",
    ].iloc[0]

    assert near_coverage == pytest.approx(
        1.0
    )

    assert early_coverage == pytest.approx(
        0.0
    )


def test_bootstrap_interval_metrics():
    evaluation_df = pd.DataFrame(
        {
            "engine_id": [
                1, 1, 2, 2
            ],
            "RUL": [
                10.0, 20.0, 15.0, 25.0
            ],
            "RUL lower": [
                5.0, 15.0, 10.0, 20.0
            ],
            "RUL upper": [
                15.0, 25.0, 20.0, 30.0
            ],
            "Interval width": [
                10.0, 10.0, 10.0, 10.0
            ],
        }
    )

    result = bootstrap_interval_metrics(
        evaluation_df=evaluation_df,
        n_bootstrap=100,
        confidence_level=0.95,
        random_state=42,
    )

    assert set(result["Metric"]) == {
        "Empirical coverage",
        "Average interval width",
        "Median interval width",
    }

    assert np.isfinite(
        result[
            [
                "Estimate",
                "CI lower",
                "CI upper",
            ]
        ].to_numpy()
    ).all()

    coverage = result.loc[
        result["Metric"]
        == "Empirical coverage",
        "Estimate",
    ].iloc[0]

    assert coverage == pytest.approx(
        1.0
    )