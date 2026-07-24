from __future__ import annotations

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    PolynomialFeatures,
    StandardScaler,
)

from typing import Any
from sklearn.base import BaseEstimator, RegressorMixin

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin

from src.config import RANDOM_STATE

def make_dummy_regressor() -> DummyRegressor:
    """
    Baseline model that always predicts the mean training RUL.
    """

    return DummyRegressor(strategy="mean")


def make_ridge_regressor(alpha: float = 1.0) -> Pipeline:
    """
    Create a Ridge regression pipeline.

    The scaler and model are fitted only on the training split.
    """

    if alpha < 0:
        raise ValueError("Ridge alpha must be non-negative.")

    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "regressor",
                Ridge(alpha=alpha),
            ),
        ]
    )


def nasa_score(
    y_true,
    y_pred,
) -> float:
    """
    Calculate the asymmetric NASA C-MAPSS score.

    Overestimating RUL is penalized more strongly because it may
    delay maintenance beyond the actual safe operating period.
    """

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    errors = y_pred_array - y_true_array

    penalties = np.where(
        errors < 0,
        np.exp(-errors / 13.0) - 1.0,
        np.exp(errors / 10.0) - 1.0,
    )

    return float(penalties.sum())


def evaluate_regression(
    model_name: str,
    y_true,
    y_pred,
) -> dict[str, float | str]:
    """
    Calculate regression metrics for one experiment.
    """

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    if y_true_array.shape != y_pred_array.shape:
        raise ValueError(
            "y_true and y_pred must have the same shape."
        )

    errors = y_pred_array - y_true_array

    near_failure_mask = y_true_array <= 30

    if near_failure_mask.any():
        near_failure_mae = mean_absolute_error(
            y_true_array[near_failure_mask],
            y_pred_array[near_failure_mask],
        )
    else:
        near_failure_mae = np.nan

    return {
        "Model": model_name,
        "MAE": float(
            mean_absolute_error(
                y_true_array,
                y_pred_array,
            )
        ),
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true_array,
                    y_pred_array,
                )
            )
        ),
        "R2": float(
            r2_score(
                y_true_array,
                y_pred_array,
            )
        ),
        "NASA Score": nasa_score(
            y_true_array,
            y_pred_array,
        ),
        "Near-failure MAE": float(
            near_failure_mae
        ),
        "Late prediction rate": float(
            np.mean(errors > 0)
        ),
    }

def make_polynomial_ridge_regressor(
    degree: int = 2,
    alpha: float = 10.0,
) -> Pipeline:
    """
    Polynomial regression implemented as polynomial features
    followed by standardized Ridge regression.
    """

    if degree < 2:
        raise ValueError(
            "Polynomial degree must be at least 2."
        )

    if alpha < 0:
        raise ValueError(
            "Ridge alpha must be non-negative."
        )

    return Pipeline(
        steps=[
            (
                "polynomial_features",
                PolynomialFeatures(
                    degree=degree,
                    include_bias=False,
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
            (
                "regressor",
                Ridge(alpha=alpha),
            ),
        ]
    )


def make_random_forest_regressor(
    n_estimators: int = 250,
) -> RandomForestRegressor:
    """
    Random Forest baseline for nonlinear RUL prediction.
    """

    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_leaf=2,
        max_features=0.7,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_gradient_boosting_regressor(
) -> GradientBoostingRegressor:
    """
    Gradient Boosting baseline for nonlinear RUL prediction.
    """

    return GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=3,
        loss="squared_error",
        random_state=RANDOM_STATE,
    )


def evaluate_rul_regions(
    model_name: str,
    y_true,
    y_pred,
) -> list[dict[str, float | str | int]]:
    """
    Evaluate prediction errors in early-life, mid-life,
    and near-failure regions.
    """

    y_true_array = np.asarray(
        y_true,
        dtype=float,
    )

    y_pred_array = np.asarray(
        y_pred,
        dtype=float,
    )

    regions = {
        "Near failure: RUL 0–30": (
            y_true_array <= 30
        ),
        "Mid life: RUL 31–125": (
            (y_true_array > 30)
            & (y_true_array <= 125)
        ),
        "Early life: RUL > 125": (
            y_true_array > 125
        ),
    }

    results = []

    for region_name, mask in regions.items():
        if not mask.any():
            continue

        region_true = y_true_array[mask]
        region_pred = y_pred_array[mask]

        results.append(
            {
                "Model": model_name,
                "Region": region_name,
                "Samples": int(mask.sum()),
                "MAE": float(
                    mean_absolute_error(
                        region_true,
                        region_pred,
                    )
                ),
                "RMSE": float(
                    np.sqrt(
                        mean_squared_error(
                            region_true,
                            region_pred,
                        )
                    )
                ),
                "Bias": float(
                    np.mean(
                        region_pred - region_true
                    )
                ),
            }
        )

    return results
def prepare_official_test_evaluation(
    test_df,
    official_rul,
):
    """
    Select the final observed cycle of each official test engine
    and attach its true remaining useful life.

    The rows of RUL_FD001 correspond to engine IDs sorted in
    ascending order.
    """
    required_columns = {
        "engine_id",
        "cycle",
    }

    missing_columns = required_columns.difference(
        test_df.columns
    )

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{sorted(missing_columns)}"
        )

    last_cycle_rows = (
        test_df
        .sort_values(
            ["engine_id", "cycle"]
        )
        .groupby(
            "engine_id",
            as_index=False,
        )
        .tail(1)
        .sort_values("engine_id")
        .reset_index(drop=True)
    )

    true_rul = np.asarray(
        official_rul
    ).reshape(-1)

    if len(last_cycle_rows) != len(true_rul):
        raise ValueError(
            "The number of official test engines "
            "does not match the number of RUL values. "
            f"Engines={len(last_cycle_rows)}, "
            f"RUL values={len(true_rul)}"
        )

    if not np.isfinite(true_rul).all():
        raise ValueError(
            "Official RUL values contain "
            "NaN or infinite values."
        )

    evaluation_df = (
        last_cycle_rows.copy()
    )

    evaluation_df[
        "actual_RUL"
    ] = true_rul.astype(float)

    return evaluation_df


class NonNegativeRegressor(
    BaseEstimator,
    RegressorMixin,
):
    """
    Wrap a fitted regression model and enforce a physical
    lower bound on its predictions.

    For RUL prediction, the default lower bound is zero
    because negative remaining useful life is not meaningful.
    """

    def __init__(
        self,
        estimator: Any,
        lower_bound: float = 0.0,
    ) -> None:
        self.estimator = estimator
        self.lower_bound = lower_bound

    def fit(
        self,
        X: Any,
        y: Any,
    ) -> "NonNegativeRegressor":
        self.estimator.fit(X, y)
        return self

    def predict(
        self,
        X: Any,
    ) -> np.ndarray:
        predictions = np.asarray(
            self.estimator.predict(X),
            dtype=float,
        )

        return np.maximum(
            predictions,
            float(self.lower_bound),
        )