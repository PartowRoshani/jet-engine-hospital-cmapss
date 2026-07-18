import numpy as np
import pandas as pd
import pytest

from src.anomaly_detection import (
    PCAReconstructionDetector,
    calculate_anomaly_threshold,
    calculate_raw_anomaly_score,
    calculate_reference_percentile,
    evaluate_anomaly_scores,
    make_anomaly_models,
    select_healthy_cycles,
)


def test_select_healthy_cycles():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 2, 2, 2],
            "cycle": [1, 2, 3, 1, 2, 3],
            "sensor_2": [1, 2, 3, 4, 5, 6],
        }
    )

    healthy = select_healthy_cycles(
        data=data,
        maximum_cycle=2,
    )

    assert len(healthy) == 4
    assert healthy["cycle"].max() == 2
    assert healthy["engine_id"].nunique() == 2


def test_select_healthy_cycles_invalid_limit():
    data = pd.DataFrame(
        {
            "engine_id": [1],
            "cycle": [1],
        }
    )

    with pytest.raises(ValueError):
        select_healthy_cycles(
            data=data,
            maximum_cycle=0,
        )


def test_reference_percentile_ordering():
    reference_scores = np.array(
        [1.0, 2.0, 3.0, 4.0]
    )

    scores = np.array(
        [0.0, 2.0, 4.0, 10.0]
    )

    percentiles = calculate_reference_percentile(
        scores=scores,
        healthy_reference_scores=reference_scores,
    )

    expected = np.array(
        [0.0, 0.5, 1.0, 1.0]
    )

    np.testing.assert_allclose(
        percentiles,
        expected,
    )


def test_anomaly_threshold():
    reference_scores = np.arange(
        1,
        101,
        dtype=float,
    )

    threshold = calculate_anomaly_threshold(
        healthy_reference_scores=reference_scores,
        quantile=0.99,
    )

    assert threshold == pytest.approx(
        np.quantile(
            reference_scores,
            0.99,
        )
    )


def test_pca_reconstruction_scores_are_valid():
    normal_data = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    detector = PCAReconstructionDetector(
        variance_to_keep=0.95,
    )

    detector.fit(normal_data)

    scores = detector.anomaly_score(
        normal_data
    )

    assert len(scores) == len(normal_data)
    assert np.isfinite(scores).all()
    assert np.all(scores >= 0.0)


def test_raw_score_direction_is_reversed():
    class DummyNormalityDetector:
        def decision_function(self, X):
            return np.array(
                [2.0, 0.5, -3.0]
            )

    detector = DummyNormalityDetector()

    scores = calculate_raw_anomaly_score(
        model=detector,
        X=np.zeros((3, 2)),
    )

    np.testing.assert_allclose(
        scores,
        np.array(
            [-2.0, -0.5, 3.0]
        ),
    )

    assert scores[2] > scores[0]


def test_anomaly_model_factory():
    models = make_anomaly_models()

    assert set(models) == {
        "Isolation Forest",
        "Local Outlier Factor",
        "One-Class SVM",
        "PCA Reconstruction",
    }

    lof_detector = (
        models["Local Outlier Factor"]
        .named_steps["detector"]
    )

    assert lof_detector.novelty is True


def test_evaluate_anomaly_scores():
    data = pd.DataFrame(
        {
            "engine_id": [
                1, 1, 1, 1,
                2, 2, 2, 2,
            ],
            "cycle": [
                1, 2, 3, 4,
                1, 2, 3, 4,
            ],
            "RUL": [
                50, 40, 20, 0,
                50, 40, 20, 0,
            ],
        }
    )

    raw_scores = np.array(
        [
            0.1, 0.2, 0.8, 1.0,
            0.1, 0.3, 0.9, 1.1,
        ]
    )

    percentile_scores = np.array(
        [
            0.1, 0.2, 0.99, 1.0,
            0.1, 0.3, 0.99, 1.0,
        ]
    )

    metrics = evaluate_anomaly_scores(
        model_name="test detector",
        data=data,
        raw_scores=raw_scores,
        percentile_scores=percentile_scores,
        percentile_threshold=0.99,
        healthy_cycle_limit=2,
        near_failure_rul=30,
    )

    assert metrics[
        "Healthy false-alarm rate"
    ] == pytest.approx(0.0)

    assert metrics[
        "Near-failure detection rate"
    ] == pytest.approx(1.0)

    assert metrics["Recall"] == pytest.approx(
        1.0
    )

    assert metrics["ROC-AUC"] == pytest.approx(
        1.0
    )