from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.anomaly_detection import (
    calculate_raw_anomaly_score,
    calculate_reference_percentile,
)
from src.artifact_management import (
    calculate_sha256,
    load_json,
)


@dataclass(frozen=True)
class FD001ArtifactBundle:
    """
    All locked FD001 models, schemas, thresholds,
    and reference data required for inference.
    """

    project_root: Path
    manifest: dict[str, Any]

    rul_evaluation_model: Any
    rul_full_train_model: Any

    classification_models: dict[int, Any]

    anomaly_model: Any
    anomaly_reference_scores: np.ndarray

    classification_config: dict[str, Any]
    anomaly_config: dict[str, Any]
    conformal_config: dict[str, Any]
    decision_policy_config: dict[str, Any]
    feature_schema: dict[str, Any]


def _resolve_artifact_path(
    project_root: Path,
    artifact_information: dict[str, Any],
) -> Path:
    """
    Resolve an artifact path using the portable relative path
    first, then the original path, then a filename search.
    """
    relative_path = artifact_information.get(
        "relative_path"
    )

    if relative_path:
        candidate = (
            project_root
            / Path(relative_path)
        )

        if candidate.exists():
            return candidate.resolve()

    stored_path = artifact_information.get(
        "path"
    )

    if stored_path:
        candidate = Path(stored_path)

        if candidate.exists():
            return candidate.resolve()

        relative_candidate = (
            project_root
            / candidate
        )

        if relative_candidate.exists():
            return relative_candidate.resolve()

    filename = artifact_information.get(
        "filename"
    )

    if filename:
        matches = list(
            (
                project_root
                / "artifacts"
            ).rglob(filename)
        )

        if len(matches) == 1:
            return matches[0].resolve()

        if len(matches) > 1:
            raise RuntimeError(
                "Multiple artifact files have the same "
                f"filename: {filename}"
            )

    raise FileNotFoundError(
        "Could not resolve artifact file: "
        f"{artifact_information}"
    )


def _verify_artifact_file(
    artifact_path: Path,
    artifact_information: dict[str, Any],
) -> None:
    """
    Verify artifact size and SHA-256 checksum.
    """
    expected_size = artifact_information.get(
        "size_bytes"
    )

    if expected_size is not None:
        actual_size = artifact_path.stat().st_size

        if actual_size != int(expected_size):
            raise RuntimeError(
                "Artifact size mismatch for "
                f"{artifact_path.name}. "
                f"Expected {expected_size}, "
                f"found {actual_size}."
            )

    expected_sha256 = artifact_information.get(
        "sha256"
    )

    if expected_sha256:
        actual_sha256 = calculate_sha256(
            artifact_path
        )

        if actual_sha256 != expected_sha256:
            raise RuntimeError(
                "Artifact checksum mismatch for "
                f"{artifact_path.name}."
            )


def _load_artifact_path(
    project_root: Path,
    artifact_inventory: dict[str, Any],
    artifact_name: str,
    verify_checksum: bool,
) -> Path:
    """
    Resolve and optionally verify one manifest artifact.
    """
    if artifact_name not in artifact_inventory:
        raise KeyError(
            "Artifact missing from manifest: "
            f"{artifact_name}"
        )

    artifact_information = (
        artifact_inventory[artifact_name]
    )

    artifact_path = _resolve_artifact_path(
        project_root=project_root,
        artifact_information=artifact_information,
    )

    if verify_checksum:
        _verify_artifact_file(
            artifact_path=artifact_path,
            artifact_information=artifact_information,
        )

    return artifact_path


def load_fd001_artifacts(
    project_root: str | Path,
    verify_checksums: bool = True,
) -> FD001ArtifactBundle:
    """
    Load the complete locked FD001 inference bundle.
    """
    resolved_project_root = Path(
        project_root
    ).resolve()

    manifest_path = (
        resolved_project_root
        / "artifacts"
        / "metadata"
        / "fd001_manifest.json"
    )

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}"
        )

    manifest = load_json(
        manifest_path
    )

    artifact_inventory = manifest.get(
        "artifacts",
        {}
    )

    required_artifacts = {
        "rul_evaluation_model",
        "rul_full_train_model",
        "classifier_10",
        "classifier_20",
        "classifier_30",
        "anomaly_model",
        "anomaly_reference_scores",
        "classification_config",
        "anomaly_config",
        "conformal_config",
        "decision_policy_config",
        "feature_schema",
    }

    missing_artifacts = required_artifacts.difference(
        artifact_inventory
    )

    if missing_artifacts:
        raise KeyError(
            "Required artifacts missing from manifest: "
            f"{sorted(missing_artifacts)}"
        )

    resolved_paths = {
        artifact_name: _load_artifact_path(
            project_root=resolved_project_root,
            artifact_inventory=artifact_inventory,
            artifact_name=artifact_name,
            verify_checksum=verify_checksums,
        )
        for artifact_name in required_artifacts
    }

    classification_models = {
        horizon: joblib.load(
            resolved_paths[
                f"classifier_{horizon}"
            ]
        )
        for horizon in [
            10,
            20,
            30,
        ]
    }

    anomaly_reference_scores = np.load(
        resolved_paths[
            "anomaly_reference_scores"
        ]
    )

    anomaly_reference_scores = np.asarray(
        anomaly_reference_scores,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(
        anomaly_reference_scores
    ).all():
        raise ValueError(
            "Anomaly reference scores contain "
            "NaN or infinity."
        )

    bundle = FD001ArtifactBundle(
        project_root=resolved_project_root,
        manifest=manifest,

        rul_evaluation_model=joblib.load(
            resolved_paths[
                "rul_evaluation_model"
            ]
        ),

        rul_full_train_model=joblib.load(
            resolved_paths[
                "rul_full_train_model"
            ]
        ),

        classification_models=(
            classification_models
        ),

        anomaly_model=joblib.load(
            resolved_paths[
                "anomaly_model"
            ]
        ),

        anomaly_reference_scores=(
            anomaly_reference_scores
        ),

        classification_config=load_json(
            resolved_paths[
                "classification_config"
            ]
        ),

        anomaly_config=load_json(
            resolved_paths[
                "anomaly_config"
            ]
        ),

        conformal_config=load_json(
            resolved_paths[
                "conformal_config"
            ]
        ),

        decision_policy_config=load_json(
            resolved_paths[
                "decision_policy_config"
            ]
        ),

        feature_schema=load_json(
            resolved_paths[
                "feature_schema"
            ]
        ),
    )

    return bundle


def select_feature_columns(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Select model features in the exact training order.
    """
    missing_columns = set(
        feature_columns
    ).difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Missing inference features: "
            f"{sorted(missing_columns)}"
        )

    selected_data = data.loc[
        :,
        feature_columns,
    ].copy()

    if selected_data.isna().any().any():
        raise ValueError(
            "Inference features contain missing values."
        )

    values = selected_data.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "Inference features contain NaN "
            "or infinite values."
        )

    return selected_data


def predict_fd001_rul(
    bundle: FD001ArtifactBundle,
    data: pd.DataFrame,
    model_scope: str = "evaluation",
) -> np.ndarray:
    """
    Predict non-negative RUL.

    model_scope:
    - evaluation: 70-engine model used for validation/test
    - full_train: 100-engine model used for deployment
    """
    feature_columns = bundle.feature_schema[
        "regression"
    ]["columns"]

    features = select_feature_columns(
        data=data,
        feature_columns=feature_columns,
    )

    if model_scope == "evaluation":
        model = bundle.rul_evaluation_model

    elif model_scope == "full_train":
        model = bundle.rul_full_train_model

    else:
        raise ValueError(
            "model_scope must be 'evaluation' "
            "or 'full_train'."
        )

    predictions = np.asarray(
        model.predict(features),
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(predictions).all():
        raise ValueError(
            "RUL model produced invalid predictions."
        )

    if (predictions < 0.0).any():
        raise ValueError(
            "RUL artifact produced negative predictions."
        )

    return predictions


def predict_fd001_failure_probabilities(
    bundle: FD001ArtifactBundle,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict calibrated monotonic failure probabilities.
    """
    feature_columns = bundle.feature_schema[
        "classification"
    ]["columns"]

    features = select_feature_columns(
        data=data,
        feature_columns=feature_columns,
    )

    raw_probability_matrix = np.column_stack(
        [
            bundle.classification_models[
                horizon
            ].predict_proba(features)[:, 1]
            for horizon in [
                10,
                20,
                30,
            ]
        ]
    )

    monotonic_probability_matrix = (
        np.maximum.accumulate(
            raw_probability_matrix,
            axis=1,
        )
    )

    return pd.DataFrame(
        monotonic_probability_matrix,
        columns=[
            "probability_10",
            "probability_20",
            "probability_30",
        ],
        index=data.index,
    )


def predict_fd001_anomaly(
    bundle: FD001ArtifactBundle,
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate anomaly raw scores and reference percentiles.
    """
    feature_columns = bundle.feature_schema[
        "anomaly_detection"
    ]["columns"]

    features = select_feature_columns(
        data=data,
        feature_columns=feature_columns,
    )

    raw_scores = calculate_raw_anomaly_score(
        model=bundle.anomaly_model,
        X=features,
    )

    raw_scores = np.asarray(
        raw_scores,
        dtype=float,
    ).reshape(-1)

    percentiles = calculate_reference_percentile(
            raw_scores,
            bundle.anomaly_reference_scores,
        )
    

    return pd.DataFrame(
        {
            "anomaly_raw_score": raw_scores,
            "anomaly_percentile": percentiles,
        },
        index=data.index,
    )


def create_fd001_rul_intervals(
    bundle: FD001ArtifactBundle,
    rul_predictions: np.ndarray,
) -> pd.DataFrame:
    """
    Create the locked 95% conformal RUL intervals.
    """
    predictions = np.asarray(
        rul_predictions,
        dtype=float,
    ).reshape(-1)

    quantile = float(
        bundle.conformal_config[
            "quantile"
        ]
    )

    lower_bound = float(
        bundle.conformal_config[
            "lower_physical_bound"
        ]
    )

    interval_lower = np.maximum(
        lower_bound,
        predictions - quantile,
    )

    interval_upper = (
        predictions + quantile
    )

    return pd.DataFrame(
        {
            "RUL prediction": predictions,
            "RUL lower": interval_lower,
            "RUL upper": interval_upper,
            "Interval width": (
                interval_upper
                - interval_lower
            ),
        }
    )