from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sklearn.base import ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    brier_score_loss,
    log_loss,
)
from sklearn.calibration import (
    CalibratedClassifierCV,
)

from sklearn.model_selection import (
    GroupKFold,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from src.config import RANDOM_STATE


def make_failure_target(
    rul: pd.Series | np.ndarray,
    horizon: int,
) -> np.ndarray:
    """
    Create a binary target indicating whether failure occurs
    within the specified number of future cycles.

    Label 1:
        RUL <= horizon

    Label 0:
        RUL > horizon
    """
    if horizon <= 0:
        raise ValueError(
            "horizon must be a positive integer."
        )

    rul_array = np.asarray(
        rul,
        dtype=float,
    ).reshape(-1)

    if not np.isfinite(rul_array).all():
        raise ValueError(
            "RUL contains NaN or infinite values."
        )

    return (
        rul_array <= horizon
    ).astype(int)


def make_dummy_classifier() -> DummyClassifier:
    """
    Probability baseline based on the training class prior.
    """
    return DummyClassifier(
        strategy="prior",
        random_state=RANDOM_STATE,
    )


def make_logistic_classifier() -> Pipeline:
    """
    Class-weighted logistic regression with scaling.
    """
    return Pipeline(
        steps=[
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=3000,
                    solver="lbfgs",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def make_random_forest_classifier(
    n_estimators: int = 250,
) -> RandomForestClassifier:
    """
    Random Forest classifier for imbalanced failure labels.
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.7,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_gradient_boosting_classifier(
) -> HistGradientBoostingClassifier:
    """
    Efficient histogram-based gradient boosting classifier.

    Class imbalance is handled through sample weights
    during fitting.
    """
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=200,
        max_leaf_nodes=31,
        min_samples_leaf=20,
        l2_regularization=1.0,
        random_state=RANDOM_STATE,
    )


def predict_positive_probability(
    model: ClassifierMixin,
    X: pd.DataFrame | np.ndarray,
) -> np.ndarray:
    """
    Return the predicted probability of the positive class.
    """
    if not hasattr(model, "predict_proba"):
        raise TypeError(
            "The classifier must provide predict_proba()."
        )

    probabilities = model.predict_proba(X)

    if probabilities.ndim != 2:
        raise ValueError(
            "predict_proba returned an invalid shape."
        )

    class_labels = np.asarray(
        model.classes_
    )

    positive_indices = np.where(
        class_labels == 1
    )[0]

    if len(positive_indices) != 1:
        raise ValueError(
            "The fitted classifier does not contain "
            "exactly one positive class labelled 1."
        )

    positive_probability = probabilities[
        :,
        positive_indices[0],
    ]

    return np.clip(
        positive_probability,
        a_min=0.0,
        a_max=1.0,
    )


def evaluate_binary_classifier(
    model_name: str,
    horizon: int,
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """
    Evaluate binary failure classification.

    PR-AUC is especially important because the positive
    failure class is imbalanced.
    """
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(
            "threshold must be between 0 and 1."
        )

    y_true_array = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true_array) != len(probability_array):
        raise ValueError(
            "y_true and probabilities must have "
            "the same length."
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "Probabilities contain NaN or infinity."
        )

    if not np.isin(
        y_true_array,
        [0, 1],
    ).all():
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    predictions = (
        probability_array >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true_array,
        predictions,
        labels=[0, 1],
    ).ravel()

    positive_rate = float(
        y_true_array.mean()
    )

    predicted_positive_rate = float(
        predictions.mean()
    )

    return {
        "Model": model_name,
        "Horizon": horizon,
        "Threshold": threshold,
        "Samples": len(y_true_array),
        "Positive rate": positive_rate,
        "Predicted positive rate": (
            predicted_positive_rate
        ),
        "Accuracy": accuracy_score(
            y_true_array,
            predictions,
        ),
        "Precision": precision_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "Recall": recall_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "F1": f1_score(
            y_true_array,
            predictions,
            zero_division=0,
        ),
        "ROC-AUC": roc_auc_score(
            y_true_array,
            probability_array,
        ),
        "PR-AUC": average_precision_score(
            y_true_array,
            probability_array,
        ),
        "True negatives": int(tn),
        "False positives": int(fp),
        "False negatives": int(fn),
        "True positives": int(tp),
    }


def get_balanced_sample_weights(
    y: pd.Series | np.ndarray,
) -> np.ndarray:
    """
    Generate balanced sample weights for estimators
    that do not have a class_weight parameter.
    """
    y_array = np.asarray(
        y,
        dtype=int,
    ).reshape(-1)

    return compute_sample_weight(
        class_weight="balanced",
        y=y_array,
    )
def evaluate_probability_calibration(
    model_name: str,
    horizon: int,
    y_true: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate the quality and calibration of predicted
    failure probabilities.

    Lower Brier score and lower log loss are better.
    """
    y_true_array = np.asarray(
        y_true,
        dtype=int,
    ).reshape(-1)

    probability_array = np.asarray(
        probabilities,
        dtype=float,
    ).reshape(-1)

    if len(y_true_array) != len(
        probability_array
    ):
        raise ValueError(
            "y_true and probabilities must "
            "have the same length."
        )

    if not np.isin(
        y_true_array,
        [0, 1],
    ).all():
        raise ValueError(
            "y_true must contain only 0 and 1."
        )

    if not np.isfinite(
        probability_array
    ).all():
        raise ValueError(
            "Probabilities contain NaN "
            "or infinite values."
        )

    probability_array = np.clip(
        probability_array,
        a_min=1e-8,
        a_max=1 - 1e-8,
    )

    observed_positive_rate = float(
        y_true_array.mean()
    )

    mean_predicted_probability = float(
        probability_array.mean()
    )

    brier_score = brier_score_loss(
        y_true_array,
        probability_array,
    )

    baseline_brier_score = (
        observed_positive_rate
        * (
            1.0
            - observed_positive_rate
        )
    )

    if baseline_brier_score > 0:
        brier_skill_score = (
            1.0
            - brier_score
            / baseline_brier_score
        )
    else:
        brier_skill_score = np.nan

    return {
        "Model": model_name,
        "Horizon": horizon,
        "Samples": len(y_true_array),
        "Observed positive rate": (
            observed_positive_rate
        ),
        "Mean predicted probability": (
            mean_predicted_probability
        ),
        "Brier score": brier_score,
        "Baseline Brier score": (
            baseline_brier_score
        ),
        "Brier skill score": (
            brier_skill_score
        ),
        "Log loss": log_loss(
            y_true_array,
            probability_array,
            labels=[0, 1],
        ),
    }
def make_group_calibration_splits(
    X: pd.DataFrame | np.ndarray,
    y: pd.Series | np.ndarray,
    groups: pd.Series | np.ndarray,
    n_splits: int = 5,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Create group-aware cross-validation splits for
    probability calibration.

    All cycles belonging to the same engine remain in
    the same fold.
    """
    y_array = np.asarray(
        y,
        dtype=int,
    ).reshape(-1)

    groups_array = np.asarray(
        groups
    ).reshape(-1)

    if len(X) != len(y_array):
        raise ValueError(
            "X and y must have the same length."
        )

    if len(y_array) != len(groups_array):
        raise ValueError(
            "y and groups must have the same length."
        )

    unique_group_count = np.unique(
        groups_array
    ).size

    if n_splits < 2:
        raise ValueError(
            "n_splits must be at least 2."
        )

    if n_splits > unique_group_count:
        raise ValueError(
            "n_splits cannot exceed the number "
            "of unique engine groups."
        )

    group_splitter = GroupKFold(
        n_splits=n_splits
    )

    return list(
        group_splitter.split(
            X,
            y_array,
            groups_array,
        )
    )


def make_calibrated_logistic_classifier(
    method: str,
    cv_splits: list[
        tuple[np.ndarray, np.ndarray]
    ],
) -> CalibratedClassifierCV:
    """
    Build a group-aware calibrated Logistic Regression.

    Supported calibration methods:
    - sigmoid: Platt scaling
    - isotonic: non-parametric calibration
    """
    if method not in {
        "sigmoid",
        "isotonic",
    }:
        raise ValueError(
            "method must be either "
            "'sigmoid' or 'isotonic'."
        )

    return CalibratedClassifierCV(
        estimator=make_logistic_classifier(),
        method=method,
        cv=cv_splits,
        ensemble=True,
    )