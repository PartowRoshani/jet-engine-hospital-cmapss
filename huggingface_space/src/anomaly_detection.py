from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from scipy.stats import spearmanr

from sklearn.base import BaseEstimator
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neighbors import LocalOutlierFactor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from src.config import RANDOM_STATE


class PCAReconstructionDetector(BaseEstimator):
    """
    PCA anomaly detector based on reconstruction error.

    The detector is fitted only on healthy training cycles.
    Larger reconstruction error means more abnormal behavior.
    """

    def __init__(
        self,
        variance_to_keep: float = 0.95,
    ) -> None:
        self.variance_to_keep = variance_to_keep

    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: None = None,
    ) -> "PCAReconstructionDetector":
        if not 0.0 < self.variance_to_keep <= 1.0:
            raise ValueError(
                "variance_to_keep must be in (0, 1]."
            )

        X_array = np.asarray(
            X,
            dtype=float,
        )

        if X_array.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional matrix."
            )

        if not np.isfinite(X_array).all():
            raise ValueError(
                "X contains NaN or infinite values."
            )

        self.scaler_ = StandardScaler()

        X_scaled = self.scaler_.fit_transform(
            X_array
        )

        self.pca_ = PCA(
            n_components=self.variance_to_keep,
            svd_solver="full",
        )

        self.pca_.fit(X_scaled)

        self.n_features_in_ = X_array.shape[1]
        self.n_components_ = self.pca_.n_components_

        return self

    def anomaly_score(
        self,
        X: pd.DataFrame | np.ndarray,
    ) -> np.ndarray:
        if not hasattr(self, "pca_"):
            raise RuntimeError(
                "The detector must be fitted first."
            )

        X_array = np.asarray(
            X,
            dtype=float,
        )

        if X_array.ndim != 2:
            raise ValueError(
                "X must be a two-dimensional matrix."
            )

        X_scaled = self.scaler_.transform(
            X_array
        )

        X_reduced = self.pca_.transform(
            X_scaled
        )

        X_reconstructed = self.pca_.inverse_transform(
            X_reduced
        )

        reconstruction_error = np.mean(
            (
                X_scaled
                - X_reconstructed
            )
            ** 2,
            axis=1,
        )

        return reconstruction_error


def select_healthy_cycles(
    data: pd.DataFrame,
    maximum_cycle: int = 30,
) -> pd.DataFrame:
    """
    Select an early-life healthy fitting region.

    Only the absolute cycle number is used. No RUL,
    final cycle, or held-out information is needed.
    """
    required_columns = {
        "engine_id",
        "cycle",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if maximum_cycle <= 0:
        raise ValueError(
            "maximum_cycle must be positive."
        )

    healthy_data = (
        data.loc[
            data["cycle"] <= maximum_cycle
        ]
        .copy()
        .sort_values(
            ["engine_id", "cycle"]
        )
        .reset_index(drop=True)
    )

    if healthy_data.empty:
        raise ValueError(
            "No healthy fitting rows were selected."
        )

    return healthy_data


def make_anomaly_models(
) -> dict[str, BaseEstimator]:
    """
    Create the four required unsupervised detectors.
    """
    return {
        "Isolation Forest": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "detector",
                    IsolationForest(
                        n_estimators=300,
                        max_samples="auto",
                        contamination="auto",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),

        "Local Outlier Factor": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "detector",
                    LocalOutlierFactor(
                        n_neighbors=35,
                        novelty=True,
                        contamination="auto",
                        n_jobs=-1,
                    ),
                ),
            ]
        ),

        "One-Class SVM": Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "detector",
                    OneClassSVM(
                        kernel="rbf",
                        nu=0.05,
                        gamma="scale",
                    ),
                ),
            ]
        ),

        "PCA Reconstruction": (
            PCAReconstructionDetector(
                variance_to_keep=0.95,
            )
        ),
    }


def calculate_raw_anomaly_score(
    model: BaseEstimator,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Return anomaly scores where larger always means
    more abnormal.
    """
    if isinstance(
        model,
        PCAReconstructionDetector,
    ):
        scores = model.anomaly_score(X)

    elif hasattr(
        model,
        "decision_function",
    ):
        # Isolation Forest, LOF and One-Class SVM
        # return larger values for more normal samples.
        scores = -np.asarray(
            model.decision_function(X),
            dtype=float,
        ).reshape(-1)

    else:
        raise TypeError(
            "Unsupported anomaly detector."
        )

    if not np.isfinite(scores).all():
        raise ValueError(
            "Anomaly scores contain NaN or infinity."
        )

    return scores


def calculate_reference_percentile(
    scores: pd.Series | np.ndarray,
    healthy_reference_scores: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Convert raw scores to percentiles relative to the
    healthy Train score distribution.

    Values near 1 are more abnormal.
    """
    score_array = np.asarray(
        scores,
        dtype=float,
    ).reshape(-1)

    reference_array = np.asarray(
        healthy_reference_scores,
        dtype=float,
    ).reshape(-1)

    if len(reference_array) == 0:
        raise ValueError(
            "Healthy reference scores are empty."
        )

    if not np.isfinite(score_array).all():
        raise ValueError(
            "Scores contain NaN or infinity."
        )

    if not np.isfinite(reference_array).all():
        raise ValueError(
            "Reference scores contain NaN or infinity."
        )

    sorted_reference = np.sort(
        reference_array
    )

    percentile_scores = (
        np.searchsorted(
            sorted_reference,
            score_array,
            side="right",
        )
        / len(sorted_reference)
    )

    return np.clip(
        percentile_scores,
        a_min=0.0,
        a_max=1.0,
    )


def calculate_anomaly_threshold(
    healthy_reference_scores: pd.Series | np.ndarray,
    quantile: float = 0.99,
) -> float:
    """
    Select a raw anomaly threshold using only healthy
    Train scores.
    """
    if not 0.0 < quantile < 1.0:
        raise ValueError(
            "quantile must be between 0 and 1."
        )

    reference_array = np.asarray(
        healthy_reference_scores,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(reference_array).all():
        raise ValueError(
            "Reference scores contain NaN or infinity."
        )

    return float(
        np.quantile(
            reference_array,
            quantile,
        )
    )


def evaluate_anomaly_scores(
    model_name: str,
    data: pd.DataFrame,
    raw_scores: pd.Series | np.ndarray,
    percentile_scores: pd.Series | np.ndarray,
    percentile_threshold: float = 0.99,
    healthy_cycle_limit: int = 30,
    near_failure_rul: int = 30,
) -> dict[str, Any]:
    """
    Evaluate an unsupervised anomaly detector.

    Raw scores are used for ranking metrics because
    percentile scores may saturate at zero or one.

    Percentile scores are used for threshold decisions
    and human-readable comparisons.
    """
    required_columns = {
        "cycle",
        "RUL",
    }

    missing_columns = required_columns.difference(
        data.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    raw_score_array = np.asarray(
        raw_scores,
        dtype=float,
    ).reshape(-1)

    percentile_array = np.asarray(
        percentile_scores,
        dtype=float,
    ).reshape(-1)

    if len(raw_score_array) != len(data):
        raise ValueError(
            "Raw score count must match data rows."
        )

    if len(percentile_array) != len(data):
        raise ValueError(
            "Percentile score count must match data rows."
        )

    if not np.isfinite(
        raw_score_array
    ).all():
        raise ValueError(
            "Raw scores contain NaN or infinity."
        )

    if not np.isfinite(
        percentile_array
    ).all():
        raise ValueError(
            "Percentile scores contain NaN or infinity."
        )

    if not 0.0 < percentile_threshold < 1.0:
        raise ValueError(
            "percentile_threshold must be in (0, 1)."
        )

    rul_array = data[
        "RUL"
    ].to_numpy(dtype=float)

    cycle_array = data[
        "cycle"
    ].to_numpy(dtype=int)

    healthy_mask = (
        cycle_array <= healthy_cycle_limit
    )

    near_failure_mask = (
        rul_array <= near_failure_rul
    )

    if not healthy_mask.any():
        raise ValueError(
            "No healthy evaluation rows found."
        )

    if not near_failure_mask.any():
        raise ValueError(
            "No near-failure evaluation rows found."
        )

    predictions = (
        percentile_array
        >= percentile_threshold
    ).astype(int)

    near_failure_labels = (
        near_failure_mask.astype(int)
    )

    tn, fp, fn, tp = confusion_matrix(
        near_failure_labels,
        predictions,
        labels=[0, 1],
    ).ravel()

    correlation_result = spearmanr(
        raw_score_array,
        -rul_array,
    )

    healthy_percentile_median = float(
        np.median(
            percentile_array[
                healthy_mask
            ]
        )
    )

    near_failure_percentile_median = float(
        np.median(
            percentile_array[
                near_failure_mask
            ]
        )
    )

    healthy_raw_median = float(
        np.median(
            raw_score_array[
                healthy_mask
            ]
        )
    )

    near_failure_raw_median = float(
        np.median(
            raw_score_array[
                near_failure_mask
            ]
        )
    )

    return {
        "Model": model_name,
        "Samples": len(data),
        "Percentile threshold": (
            percentile_threshold
        ),
        "Healthy rows": int(
            healthy_mask.sum()
        ),
        "Near-failure rows": int(
            near_failure_mask.sum()
        ),
        "Healthy false-alarm rate": float(
            predictions[
                healthy_mask
            ].mean()
        ),
        "Near-failure detection rate": float(
            predictions[
                near_failure_mask
            ].mean()
        ),
        "Precision": precision_score(
            near_failure_labels,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            near_failure_labels,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            near_failure_labels,
            raw_score_array,
        ),
        "PR-AUC": average_precision_score(
            near_failure_labels,
            raw_score_array,
        ),
        "Healthy median percentile": (
            healthy_percentile_median
        ),
        "Near-failure median percentile": (
            near_failure_percentile_median
        ),
        "Percentile separation": (
            near_failure_percentile_median
            - healthy_percentile_median
        ),
        "Healthy median raw score": (
            healthy_raw_median
        ),
        "Near-failure median raw score": (
            near_failure_raw_median
        ),
        "Raw score separation": (
            near_failure_raw_median
            - healthy_raw_median
        ),
        "Percentile saturation at zero": float(
            np.isclose(
                percentile_array,
                0.0,
            ).mean()
        ),
        "Percentile saturation at one": float(
            np.isclose(
                percentile_array,
                1.0,
            ).mean()
        ),
        "Unique percentile values": int(
            np.unique(
                percentile_array
            ).size
        ),
        "True negatives": int(tn),
        "False positives": int(fp),
        "False negatives": int(fn),
        "True positives": int(tp),
        "Spearman raw score vs degradation": float(
            correlation_result.statistic
        ),
    }
def bootstrap_anomaly_metric_intervals(
    data: pd.DataFrame,
    raw_scores: pd.Series | np.ndarray,
    percentile_scores: pd.Series | np.ndarray,
    percentile_threshold: float,
    healthy_cycle_limit: int = 30,
    near_failure_rul: int = 30,
    n_bootstrap: int = 5000,
    confidence_level: float = 0.95,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Calculate confidence intervals by resampling complete
    engines rather than independent rows.
    """
    if "engine_id" not in data.columns:
        raise ValueError(
            "data must contain engine_id."
        )

    if n_bootstrap < 100:
        raise ValueError(
            "n_bootstrap must be at least 100."
        )

    if not 0.0 < confidence_level < 1.0:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    raw_score_array = np.asarray(
        raw_scores,
        dtype=float,
    ).reshape(-1)

    percentile_array = np.asarray(
        percentile_scores,
        dtype=float,
    ).reshape(-1)

    if len(raw_score_array) != len(data):
        raise ValueError(
            "Raw score count must match data rows."
        )

    if len(percentile_array) != len(data):
        raise ValueError(
            "Percentile score count must match data rows."
        )

    data = data.reset_index(
        drop=True
    ).copy()

    engine_ids = (
        data["engine_id"]
        .drop_duplicates()
        .to_numpy()
    )

    engine_row_indices = {
        engine_id: np.flatnonzero(
            data["engine_id"].to_numpy()
            == engine_id
        )
        for engine_id in engine_ids
    }

    point_metrics = evaluate_anomaly_scores(
        model_name="Isolation Forest",
        data=data,
        raw_scores=raw_score_array,
        percentile_scores=percentile_array,
        percentile_threshold=(
            percentile_threshold
        ),
        healthy_cycle_limit=(
            healthy_cycle_limit
        ),
        near_failure_rul=(
            near_failure_rul
        ),
    )

    metric_names = [
        "ROC-AUC",
        "PR-AUC",
        "Healthy false-alarm rate",
        "Near-failure detection rate",
        "Precision",
        "Recall",
        "Spearman raw score vs degradation",
    ]

    bootstrap_values = {
        metric_name: []
        for metric_name in metric_names
    }

    rng = np.random.default_rng(
        random_state
    )

    for _ in range(n_bootstrap):
        sampled_engines = rng.choice(
            engine_ids,
            size=len(engine_ids),
            replace=True,
        )

        sampled_indices = np.concatenate(
            [
                engine_row_indices[
                    engine_id
                ]
                for engine_id
                in sampled_engines
            ]
        )

        sampled_data = (
            data.iloc[
                sampled_indices
            ]
            .reset_index(drop=True)
        )

        sampled_raw_scores = (
            raw_score_array[
                sampled_indices
            ]
        )

        sampled_percentiles = (
            percentile_array[
                sampled_indices
            ]
        )

        sampled_metrics = (
            evaluate_anomaly_scores(
                model_name=(
                    "Isolation Forest"
                ),
                data=sampled_data,
                raw_scores=(
                    sampled_raw_scores
                ),
                percentile_scores=(
                    sampled_percentiles
                ),
                percentile_threshold=(
                    percentile_threshold
                ),
                healthy_cycle_limit=(
                    healthy_cycle_limit
                ),
                near_failure_rul=(
                    near_failure_rul
                ),
            )
        )

        for metric_name in metric_names:
            metric_value = float(
                sampled_metrics[
                    metric_name
                ]
            )

            if np.isfinite(metric_value):
                bootstrap_values[
                    metric_name
                ].append(
                    metric_value
                )

    alpha = 1.0 - confidence_level
    lower_quantile = alpha / 2.0
    upper_quantile = 1.0 - alpha / 2.0

    result_rows = []

    for metric_name in metric_names:
        values = np.asarray(
            bootstrap_values[
                metric_name
            ],
            dtype=float,
        )

        result_rows.append(
            {
                "Metric": metric_name,
                "Estimate": float(
                    point_metrics[
                        metric_name
                    ]
                ),
                "CI lower": float(
                    np.quantile(
                        values,
                        lower_quantile,
                    )
                ),
                "CI upper": float(
                    np.quantile(
                        values,
                        upper_quantile,
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