from pathlib import Path

import numpy as np

from src.artifact_loader import (
    load_fd001_artifacts,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def test_fd001_artifact_bundle_loads():
    bundle = load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )

    assert set(
        bundle.classification_models
    ) == {
        10,
        20,
        30,
    }

    assert len(
        bundle.anomaly_reference_scores
    ) == 2100

    assert np.isfinite(
        bundle.anomaly_reference_scores
    ).all()


def test_fd001_feature_schema():
    bundle = load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )

    assert len(
        bundle.feature_schema[
            "regression"
        ]["columns"]
    ) == 18

    assert len(
        bundle.feature_schema[
            "classification"
        ]["columns"]
    ) == 138

    assert len(
        bundle.feature_schema[
            "anomaly_detection"
        ]["columns"]
    ) == 15


def test_fd001_locked_thresholds():
    bundle = load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )

    horizons = bundle.classification_config[
        "horizons"
    ]

    assert horizons["10"]["threshold"] == 0.30
    assert horizons["20"]["threshold"] == 0.29
    assert horizons["30"]["threshold"] == 0.27

    assert (
        bundle.anomaly_config[
            "percentile_threshold"
        ]
        == 0.9999
    )

    assert (
        bundle.conformal_config[
            "confidence_level"
        ]
        == 0.95
    )

    assert (
        bundle.decision_policy_config[
            "policy_name"
        ]
        == "supervised_only"
    )