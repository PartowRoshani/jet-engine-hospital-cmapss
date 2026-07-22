from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.artifact_loader import (
    load_fd001_artifacts,
)
from src.inference import (
    create_fd001_dashboard_view,
    run_fd001_inference,
    validate_fd001_inference_frame,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


@pytest.fixture(scope="module")
def artifact_bundle():
    return load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )


def make_synthetic_feature_frame(
    artifact_bundle,
) -> pd.DataFrame:
    regression_columns = (
        artifact_bundle
        .feature_schema[
            "regression"
        ][
            "columns"
        ]
    )

    classification_columns = (
        artifact_bundle
        .feature_schema[
            "classification"
        ][
            "columns"
        ]
    )

    anomaly_columns = (
        artifact_bundle
        .feature_schema[
            "anomaly_detection"
        ][
            "columns"
        ]
    )

    feature_columns = sorted(
        set(
            regression_columns
            + classification_columns
            + anomaly_columns
        )
    )

    number_of_rows = 8

    data = pd.DataFrame(
        0.0,
        index=range(number_of_rows),
        columns=feature_columns,
    )

    data["engine_id"] = [
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
    ]

    data["cycle"] = [
        1,
        2,
        3,
        4,
        1,
        2,
        3,
        4,
    ]

    return data


def test_validate_inference_frame():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1, 2],
            "cycle": [1, 2, 1],
        }
    )

    validate_fd001_inference_frame(
        data
    )


def test_duplicate_engine_cycle_rejected():
    data = pd.DataFrame(
        {
            "engine_id": [1, 1],
            "cycle": [1, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="Duplicate",
    ):
        validate_fd001_inference_frame(
            data
        )


def test_unified_inference_outputs(
    artifact_bundle,
):
    data = make_synthetic_feature_frame(
        artifact_bundle
    )

    result = run_fd001_inference(
        bundle=artifact_bundle,
        feature_df=data,
        model_scope="evaluation",
    )

    required_columns = {
        "engine_id",
        "cycle",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_raw_score",
        "anomaly_percentile",
        "Action",
        "Action level",
        "Trigger",
        "Confidence",
    }

    assert required_columns.issubset(
        result.columns
    )

    assert len(result) == len(data)

    assert np.isfinite(
        result[
            [
                "RUL prediction",
                "RUL lower",
                "RUL upper",
                "probability_10",
                "probability_20",
                "probability_30",
                "anomaly_raw_score",
                "anomaly_percentile",
            ]
        ].to_numpy()
    ).all()

    assert (
        result["RUL prediction"] >= 0.0
    ).all()

    assert (
        result["RUL lower"] >= 0.0
    ).all()

    assert (
        result["RUL lower"]
        <= result["RUL prediction"]
    ).all()

    assert (
        result["RUL prediction"]
        <= result["RUL upper"]
    ).all()


def test_failure_probabilities_are_monotonic(
    artifact_bundle,
):
    data = make_synthetic_feature_frame(
        artifact_bundle
    )

    result = run_fd001_inference(
        bundle=artifact_bundle,
        feature_df=data,
    )

    assert (
        result["probability_10"]
        <= result["probability_20"]
    ).all()

    assert (
        result["probability_20"]
        <= result["probability_30"]
    ).all()


def test_actions_do_not_downgrade(
    artifact_bundle,
):
    data = make_synthetic_feature_frame(
        artifact_bundle
    )

    result = run_fd001_inference(
        bundle=artifact_bundle,
        feature_df=data,
    )

    for _, engine_df in result.groupby(
        "engine_id"
    ):
        level_changes = (
            engine_df[
                "Action level"
            ]
            .diff()
            .fillna(0)
        )

        assert (
            level_changes >= 0
        ).all()


def test_dashboard_has_one_row_per_engine(
    artifact_bundle,
):
    data = make_synthetic_feature_frame(
        artifact_bundle
    )

    inference_df = run_fd001_inference(
        bundle=artifact_bundle,
        feature_df=data,
    )

    dashboard_df = (
        create_fd001_dashboard_view(
            inference_df
        )
    )

    assert len(dashboard_df) == 2

    assert (
        dashboard_df[
            "engine_id"
        ].nunique()
        == 2
    )


def test_full_train_scope_is_rejected(
    artifact_bundle,
):
    data = make_synthetic_feature_frame(
        artifact_bundle
    )

    with pytest.raises(
        ValueError,
        match="evaluation",
    ):
        run_fd001_inference(
            bundle=artifact_bundle,
            feature_df=data,
            model_scope="full_train",
        )