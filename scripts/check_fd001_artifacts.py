from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import numpy as np

from src.artifact_loader import load_fd001_artifacts


def main() -> None:
    bundle = load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )

    assert set(bundle.classification_models) == {
        10,
        20,
        30,
    }

    assert len(bundle.anomaly_reference_scores) == 2100

    assert np.isfinite(
        bundle.anomaly_reference_scores
    ).all()

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

    print(
        "FD001 artifact integrity check passed."
    )

    print(
        "Artifact version:",
        bundle.manifest[
            "artifact_version"
        ],
    )

    print(
        "Classification horizons:",
        sorted(
            bundle.classification_models.keys()
        ),
    )

    print(
        "Anomaly reference rows:",
        len(
            bundle.anomaly_reference_scores
        ),
    )


if __name__ == "__main__":
    main()