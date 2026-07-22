from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = PROJECT_ROOT / "reports" / "tables"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.decision_policy import (
    DecisionPolicyConfig,
    apply_maintenance_policy,
    enforce_action_hysteresis,
)


def require_columns(
    frame: pd.DataFrame,
    required_columns: list[str],
    frame_name: str,
) -> None:
    missing_columns = sorted(
        set(required_columns).difference(
            frame.columns
        )
    )

    if missing_columns:
        raise ValueError(
            f"{frame_name} is missing columns: "
            f"{missing_columns}"
        )


def calculate_sha256_bytes(
    file_bytes: bytes,
) -> str:
    return hashlib.sha256(
        file_bytes
    ).hexdigest()


def load_table(
    filename: str,
) -> pd.DataFrame:
    path = TABLES_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Required table not found: {path}"
        )

    return pd.read_csv(path)


def write_table(
    frame: pd.DataFrame,
    filename: str,
) -> None:
    output_path = TABLES_DIR / filename

    frame.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Updated: {output_path.relative_to(PROJECT_ROOT)}"
    )


# ---------------------------------------------------------------------
# Load corrected audit results.
# ---------------------------------------------------------------------

corrected_regression_summary = load_table(
    "fd001_internal_test_regression_corrected.csv"
)

corrected_regression_predictions = load_table(
    "fd001_internal_test_regression_predictions_corrected.csv"
)

corrected_regression_ci = load_table(
    "fd001_internal_test_regression_ci_corrected.csv"
)

corrected_interval_rows = load_table(
    "fd001_internal_test_conformal_intervals_corrected.csv"
)

corrected_interval_summary = load_table(
    "fd001_internal_test_conformal_summary_corrected.csv"
)

corrected_interval_ci = load_table(
    "fd001_internal_test_conformal_ci_corrected.csv"
)


require_columns(
    corrected_regression_summary,
    [
        "MAE",
        "RMSE",
        "R2",
        "NASA score",
        "Near-failure MAE",
        "Late-prediction rate",
    ],
    "corrected regression summary",
)

require_columns(
    corrected_regression_predictions,
    [
        "engine_id",
        "cycle",
        "RUL",
        "RUL prediction",
    ],
    "corrected regression predictions",
)

require_columns(
    corrected_interval_rows,
    [
        "engine_id",
        "cycle",
        "RUL",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "Interval width",
    ],
    "corrected interval rows",
)


key_columns = [
    "engine_id",
    "cycle",
]

if corrected_regression_predictions.duplicated(
    key_columns
).any():
    raise ValueError(
        "Corrected regression predictions contain "
        "duplicate engine-cycle keys."
    )

corrected_regression_predictions = (
    corrected_regression_predictions
    .sort_values(key_columns)
    .reset_index(drop=True)
)

corrected_interval_rows = (
    corrected_interval_rows
    .sort_values(key_columns)
    .reset_index(drop=True)
)


pd.testing.assert_frame_equal(
    corrected_regression_predictions[
        [
            "engine_id",
            "cycle",
            "RUL",
        ]
    ],
    corrected_interval_rows[
        [
            "engine_id",
            "cycle",
            "RUL",
        ]
    ],
    check_dtype=False,
)


np.testing.assert_allclose(
    corrected_regression_predictions[
        "RUL prediction"
    ].to_numpy(dtype=float),
    corrected_interval_rows[
        "RUL prediction"
    ].to_numpy(dtype=float),
    rtol=0.0,
    atol=1e-10,
)


# ---------------------------------------------------------------------
# Reconcile official regression tables.
# ---------------------------------------------------------------------

corrected_metrics = (
    corrected_regression_summary.iloc[0]
)


official_regression_results = load_table(
    "fd001_internal_test_regression_results.csv"
)

if len(official_regression_results) != 1:
    raise ValueError(
        "Expected exactly one official regression result row."
    )

official_regression_results.loc[
    0,
    "MAE",
] = float(
    corrected_metrics["MAE"]
)

official_regression_results.loc[
    0,
    "RMSE",
] = float(
    corrected_metrics["RMSE"]
)

official_regression_results.loc[
    0,
    "R2",
] = float(
    corrected_metrics["R2"]
)

official_regression_results.loc[
    0,
    "NASA Score",
] = float(
    corrected_metrics["NASA score"]
)

official_regression_results.loc[
    0,
    "Near-failure MAE",
] = float(
    corrected_metrics[
        "Near-failure MAE"
    ]
)

official_regression_results.loc[
    0,
    "Late prediction rate",
] = float(
    corrected_metrics[
        "Late-prediction rate"
    ]
)


validation_vs_internal = load_table(
    "fd001_validation_vs_internal_test.csv"
)

internal_mask = (
    validation_vs_internal[
        "Dataset"
    ]
    .astype(str)
    .str.strip()
    .str.lower()
    .eq("internal test")
)

if int(internal_mask.sum()) != 1:
    raise ValueError(
        "Expected exactly one Internal test row."
    )

validation_vs_internal.loc[
    internal_mask,
    "MAE",
] = float(
    corrected_metrics["MAE"]
)

validation_vs_internal.loc[
    internal_mask,
    "RMSE",
] = float(
    corrected_metrics["RMSE"]
)

validation_vs_internal.loc[
    internal_mask,
    "R2",
] = float(
    corrected_metrics["R2"]
)

validation_vs_internal.loc[
    internal_mask,
    "NASA Score",
] = float(
    corrected_metrics["NASA score"]
)

validation_vs_internal.loc[
    internal_mask,
    "Near-failure MAE",
] = float(
    corrected_metrics[
        "Near-failure MAE"
    ]
)

validation_vs_internal.loc[
    internal_mask,
    "Late prediction rate",
] = float(
    corrected_metrics[
        "Late-prediction rate"
    ]
)


# ---------------------------------------------------------------------
# Rebuild internal-test conformal tables for every confidence level.
# Validation rows and quantiles remain locked and unchanged.
# ---------------------------------------------------------------------

existing_conformal_evaluation = load_table(
    "fd001_conformal_evaluation.csv"
)

validation_conformal_rows = (
    existing_conformal_evaluation.loc[
        existing_conformal_evaluation[
            "Dataset"
        ].eq("Validation")
    ]
    .copy()
)

if len(validation_conformal_rows) < 1:
    raise ValueError(
        "No validation conformal rows were found."
    )


actual_rul = (
    corrected_regression_predictions[
        "RUL"
    ].to_numpy(dtype=float)
)

point_predictions = (
    corrected_regression_predictions[
        "RUL prediction"
    ].to_numpy(dtype=float)
)

internal_prediction_frames = []
internal_evaluation_rows = []


for _, validation_row in (
    validation_conformal_rows.iterrows()
):
    confidence_level = float(
        validation_row[
            "Confidence level"
        ]
    )

    conformal_quantile = float(
        validation_row[
            "Conformal quantile"
        ]
    )

    lower_bounds = np.maximum(
        point_predictions
        - conformal_quantile,
        0.0,
    )

    upper_bounds = (
        point_predictions
        + conformal_quantile
    )

    interval_widths = (
        upper_bounds
        - lower_bounds
    )

    covered = (
        (actual_rul >= lower_bounds)
        & (actual_rul <= upper_bounds)
    )

    lower_misses = (
        actual_rul < lower_bounds
    )

    upper_misses = (
        actual_rul > upper_bounds
    )

    interval_frame = pd.DataFrame(
        {
            "Confidence level": (
                confidence_level
            ),
            "engine_id": (
                corrected_regression_predictions[
                    "engine_id"
                ].to_numpy()
            ),
            "cycle": (
                corrected_regression_predictions[
                    "cycle"
                ].to_numpy()
            ),
            "RUL": actual_rul,
            "RUL prediction": (
                point_predictions
            ),
            "RUL lower": lower_bounds,
            "RUL upper": upper_bounds,
            "Interval width": (
                interval_widths
            ),
            "Covered": covered,
        }
    )

    internal_prediction_frames.append(
        interval_frame
    )

    internal_evaluation_rows.append(
        {
            "Dataset": "Internal test",
            "Confidence level": (
                confidence_level
            ),
            "Conformal quantile": (
                conformal_quantile
            ),
            "Samples": int(
                len(interval_frame)
            ),
            "Empirical coverage": float(
                covered.mean()
            ),
            "Average interval width": float(
                interval_widths.mean()
            ),
            "Median interval width": float(
                np.median(
                    interval_widths
                )
            ),
            "Minimum interval width": float(
                interval_widths.min()
            ),
            "Maximum interval width": float(
                interval_widths.max()
            ),
            "Lower misses": int(
                lower_misses.sum()
            ),
            "Upper misses": int(
                upper_misses.sum()
            ),
            "Total misses": int(
                (
                    lower_misses
                    | upper_misses
                ).sum()
            ),
            "Mean prediction": float(
                point_predictions.mean()
            ),
            "Mean actual RUL": float(
                actual_rul.mean()
            ),
        }
    )


all_internal_conformal_predictions = (
    pd.concat(
        internal_prediction_frames,
        ignore_index=True,
    )
    .sort_values(
        [
            "Confidence level",
            "engine_id",
            "cycle",
        ]
    )
    .reset_index(drop=True)
)


internal_conformal_evaluation = (
    pd.DataFrame(
        internal_evaluation_rows
    )
)


combined_conformal_evaluation = (
    pd.concat(
        [
            validation_conformal_rows,
            internal_conformal_evaluation,
        ],
        ignore_index=True,
    )
)

combined_conformal_evaluation[
    "_dataset_order"
] = (
    combined_conformal_evaluation[
        "Dataset"
    ].map(
        {
            "Validation": 0,
            "Internal test": 1,
        }
    )
)

combined_conformal_evaluation = (
    combined_conformal_evaluation
    .sort_values(
        [
            "Confidence level",
            "_dataset_order",
        ]
    )
    .drop(
        columns=[
            "_dataset_order",
        ]
    )
    .reset_index(drop=True)
)


confidence_95_mask = np.isclose(
    all_internal_conformal_predictions[
        "Confidence level"
    ].to_numpy(dtype=float),
    0.95,
)

conformal_95 = (
    all_internal_conformal_predictions.loc[
        confidence_95_mask
    ]
    .copy()
    .sort_values(key_columns)
    .reset_index(drop=True)
)


conformal_95[
    "Uncertainty risk 30"
] = (
    conformal_95[
        "RUL lower"
    ] <= 30.0
)

conformal_95[
    "Uncertainty risk 20"
] = (
    conformal_95[
        "RUL lower"
    ] <= 20.0
)

conformal_95[
    "Uncertainty risk 10"
] = (
    conformal_95[
        "RUL lower"
    ] <= 10.0
)

conformal_95[
    "Lower miss"
] = (
    conformal_95["RUL"]
    < conformal_95["RUL lower"]
)

conformal_95[
    "Upper miss"
] = (
    conformal_95["RUL"]
    > conformal_95["RUL upper"]
)


conformal_95 = conformal_95[
    [
        "Confidence level",
        "engine_id",
        "cycle",
        "RUL",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "Interval width",
        "Covered",
        "Uncertainty risk 30",
        "Uncertainty risk 20",
        "Uncertainty risk 10",
        "Lower miss",
        "Upper miss",
    ]
]


# Confirm rebuilt 95% intervals match the corrected audit file.
np.testing.assert_allclose(
    conformal_95[
        [
            "RUL prediction",
            "RUL lower",
            "RUL upper",
            "Interval width",
        ]
    ].to_numpy(dtype=float),
    corrected_interval_rows[
        [
            "RUL prediction",
            "RUL lower",
            "RUL upper",
            "Interval width",
        ]
    ].to_numpy(dtype=float),
    rtol=0.0,
    atol=1e-10,
)


corrected_summary_row = (
    corrected_interval_summary.iloc[0]
)

rebuilt_95_coverage = float(
    conformal_95[
        "Covered"
    ].mean()
)

rebuilt_95_average_width = float(
    conformal_95[
        "Interval width"
    ].mean()
)

if not np.isclose(
    rebuilt_95_coverage,
    float(
        corrected_summary_row[
            "Coverage"
        ]
    ),
    atol=1e-12,
):
    raise AssertionError(
        "Rebuilt 95% coverage does not match "
        "the corrected audit result."
    )

if not np.isclose(
    rebuilt_95_average_width,
    float(
        corrected_summary_row[
            "Average width"
        ]
    ),
    atol=1e-12,
):
    raise AssertionError(
        "Rebuilt 95% average width does not match "
        "the corrected audit result."
    )


# ---------------------------------------------------------------------
# Rebuild 95% per-engine conformal summaries.
# ---------------------------------------------------------------------

conformal_per_engine = (
    conformal_95
    .groupby(
        "engine_id",
        as_index=False,
    )
    .agg(
        Samples=(
            "RUL",
            "size",
        ),
        Coverage=(
            "Covered",
            "mean",
        ),
        Average_width=(
            "Interval width",
            "mean",
        ),
        Median_width=(
            "Interval width",
            "median",
        ),
        Lower_misses=(
            "Lower miss",
            "sum",
        ),
        Upper_misses=(
            "Upper miss",
            "sum",
        ),
        Minimum_RUL=(
            "RUL",
            "min",
        ),
        Maximum_RUL=(
            "RUL",
            "max",
        ),
    )
    .sort_values(
        "engine_id"
    )
    .reset_index(drop=True)
)


conformal_worst_engines = (
    conformal_per_engine
    .sort_values(
        [
            "Coverage",
            "engine_id",
        ],
        ascending=[
            True,
            True,
        ],
    )
    .head(5)
    .reset_index(drop=True)
)


# ---------------------------------------------------------------------
# Replace the official compact conformal CI table.
# ---------------------------------------------------------------------

required_ci_metrics = {
    "Overall coverage": (
        "Empirical coverage"
    ),
    "Average interval width": (
        "Average interval width"
    ),
    "Median interval width": (
        "Median interval width"
    ),
}

official_conformal_ci_rows = []

for source_metric, output_metric in (
    required_ci_metrics.items()
):
    metric_rows = (
        corrected_interval_ci.loc[
            corrected_interval_ci[
                "Metric"
            ].eq(source_metric)
        ]
    )

    if len(metric_rows) != 1:
        raise ValueError(
            "Expected one corrected CI row for "
            f"{source_metric}."
        )

    metric_row = metric_rows.iloc[0]

    official_conformal_ci_rows.append(
        {
            "Metric": output_metric,
            "Estimate": float(
                metric_row["Estimate"]
            ),
            "CI lower": float(
                metric_row["CI lower"]
            ),
            "CI upper": float(
                metric_row["CI upper"]
            ),
            "Confidence level": float(
                metric_row[
                    "Confidence level"
                ]
            ),
            "Method": str(
                metric_row["Method"]
            ),
        }
    )


official_conformal_ci = pd.DataFrame(
    official_conformal_ci_rows
)


# ---------------------------------------------------------------------
# Correct decision evidence and rerun the locked decision policy.
# ---------------------------------------------------------------------

old_decision_evidence = (
    load_table(
        "fd001_internal_test_decision_evidence.csv"
    )
    .sort_values(key_columns)
    .reset_index(drop=True)
)

corrected_decision_evidence = (
    old_decision_evidence.copy()
)


pd.testing.assert_frame_equal(
    corrected_decision_evidence[
        [
            "engine_id",
            "cycle",
            "RUL",
        ]
    ],
    conformal_95[
        [
            "engine_id",
            "cycle",
            "RUL",
        ]
    ],
    check_dtype=False,
)


for column in [
    "RUL prediction",
    "RUL lower",
    "RUL upper",
]:
    corrected_decision_evidence[
        column
    ] = conformal_95[
        column
    ].to_numpy()


old_decision_policy = (
    load_table(
        "fd001_internal_test_decision_policy.csv"
    )
    .sort_values(key_columns)
    .reset_index(drop=True)
)


decision_config_path = (
    PROJECT_ROOT
    / "artifacts"
    / "thresholds"
    / "fd001_decision_policy.json"
)

with decision_config_path.open(
    "r",
    encoding="utf-8",
) as decision_config_file:
    decision_config_dict = json.load(
        decision_config_file
    )


classification_config_path = (
    PROJECT_ROOT
    / "artifacts"
    / "thresholds"
    / "fd001_classification_config.json"
)

with classification_config_path.open(
    "r",
    encoding="utf-8",
) as classification_config_file:
    classification_config = json.load(
        classification_config_file
    )


anomaly_config_path = (
    PROJECT_ROOT
    / "artifacts"
    / "thresholds"
    / "fd001_anomaly_config.json"
)

with anomaly_config_path.open(
    "r",
    encoding="utf-8",
) as anomaly_config_file:
    anomaly_config = json.load(
        anomaly_config_file
    )


required_horizons = {
    "10",
    "20",
    "30",
}

available_horizons = set(
    classification_config[
        "horizons"
    ]
)

missing_horizons = sorted(
    required_horizons.difference(
        available_horizons
    )
)

if missing_horizons:
    raise ValueError(
        "Classification configuration is missing horizons: "
        f"{missing_horizons}"
    )


persistence_config = (
    decision_config_dict[
        "persistence"
    ]
)


decision_config = DecisionPolicyConfig(
    probability_threshold_10=float(
        classification_config[
            "horizons"
        ][
            "10"
        ][
            "threshold"
        ]
    ),
    probability_threshold_20=float(
        classification_config[
            "horizons"
        ][
            "20"
        ][
            "threshold"
        ]
    ),
    probability_threshold_30=float(
        classification_config[
            "horizons"
        ][
            "30"
        ][
            "threshold"
        ]
    ),
    anomaly_threshold=float(
        anomaly_config[
            "percentile_threshold"
        ]
    ),
    persistence_alerts_required=int(
        persistence_config[
            "alerts_required"
        ]
    ),
    persistence_window_size=int(
        persistence_config[
            "window_size"
        ]
    ),
    stop_rul_boundary=10.0,
    inspect_rul_boundary=30.0,
)


print(
    "Locked decision configuration:"
)

print(
    decision_config
)


policy_names = (
    old_decision_policy[
        "Policy"
    ]
    .dropna()
    .astype(str)
    .unique()
)

if len(policy_names) != 1:
    raise ValueError(
        "Expected exactly one locked policy name."
    )

locked_policy_name = str(
    policy_names[0]
)


rebuilt_raw_policy = (
    apply_maintenance_policy(
        evidence_df=(
            corrected_decision_evidence
        ),
        policy_name=(
            locked_policy_name
        ),
        config=decision_config,
    )
)

rebuilt_decision_policy = (
    enforce_action_hysteresis(
        policy_df=(
            rebuilt_raw_policy
        )
    )
    .sort_values(key_columns)
    .reset_index(drop=True)
)


missing_policy_columns = sorted(
    set(
        old_decision_policy.columns
    ).difference(
        rebuilt_decision_policy.columns
    )
)

if missing_policy_columns:
    raise ValueError(
        "Rebuilt policy is missing columns: "
        f"{missing_policy_columns}"
    )


rebuilt_decision_policy = (
    rebuilt_decision_policy[
        old_decision_policy.columns
    ]
)


old_actions = (
    old_decision_policy[
        "Action"
    ]
    .astype(str)
    .to_numpy()
)

new_actions = (
    rebuilt_decision_policy[
        "Action"
    ]
    .astype(str)
    .to_numpy()
)

changed_actions = int(
    (
        old_actions
        != new_actions
    ).sum()
)

if changed_actions != 0:
    raise AssertionError(
        "Correcting RUL evidence changed "
        f"{changed_actions} maintenance actions."
    )


old_action_levels = (
    old_decision_policy[
        "Action level"
    ].to_numpy()
)

new_action_levels = (
    rebuilt_decision_policy[
        "Action level"
    ].to_numpy()
)

if not np.array_equal(
    old_action_levels,
    new_action_levels,
):
    raise AssertionError(
        "Correcting RUL evidence changed "
        "maintenance action levels."
    )


# ---------------------------------------------------------------------
# Update conformal configuration and artifact manifest.
# ---------------------------------------------------------------------

conformal_config_path = (
    PROJECT_ROOT
    / "artifacts"
    / "thresholds"
    / "fd001_conformal_config.json"
)

with conformal_config_path.open(
    "r",
    encoding="utf-8",
) as conformal_config_file:
    conformal_config = json.load(
        conformal_config_file
    )


conformal_config[
    "internal_test_average_width"
] = rebuilt_95_average_width

conformal_config[
    "internal_test_coverage"
] = rebuilt_95_coverage


conformal_config_text = (
    json.dumps(
        conformal_config,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n"
)

conformal_config_bytes = (
    conformal_config_text.encode(
        "utf-8"
    )
)


manifest_path = (
    PROJECT_ROOT
    / "artifacts"
    / "metadata"
    / "fd001_manifest.json"
)

with manifest_path.open(
    "r",
    encoding="utf-8",
) as manifest_file:
    manifest = json.load(
        manifest_file
    )


manifest[
    "artifact_version"
] = "1.0.2"

manifest[
    "created_at_utc"
] = datetime.now(
    timezone.utc
).isoformat()


conformal_manifest_entry = (
    manifest[
        "artifacts"
    ][
        "conformal_config"
    ]
)

conformal_manifest_entry[
    "sha256"
] = calculate_sha256_bytes(
    conformal_config_bytes
)

conformal_manifest_entry[
    "size_bytes"
] = len(
    conformal_config_bytes
)


manifest_text = (
    json.dumps(
        manifest,
        indent=2,
        ensure_ascii=False,
        sort_keys=True,
    )
    + "\n"
)


# ---------------------------------------------------------------------
# All calculations and consistency checks have passed.
# Write official files only now.
# ---------------------------------------------------------------------

write_table(
    official_regression_results,
    "fd001_internal_test_regression_results.csv",
)

write_table(
    validation_vs_internal,
    "fd001_validation_vs_internal_test.csv",
)

write_table(
    all_internal_conformal_predictions,
    "fd001_conformal_internal_test_predictions.csv",
)

write_table(
    conformal_95,
    "fd001_conformal_95_internal_test.csv",
)

write_table(
    combined_conformal_evaluation,
    "fd001_conformal_evaluation.csv",
)

write_table(
    official_conformal_ci,
    "fd001_conformal_confidence_intervals.csv",
)

write_table(
    conformal_per_engine,
    "fd001_conformal_per_engine.csv",
)

write_table(
    conformal_worst_engines,
    "fd001_conformal_worst_engines.csv",
)

write_table(
    corrected_decision_evidence,
    "fd001_internal_test_decision_evidence.csv",
)

write_table(
    rebuilt_decision_policy,
    "fd001_internal_test_decision_policy.csv",
)


conformal_config_path.write_bytes(
    conformal_config_bytes
)

manifest_path.write_text(
    manifest_text,
    encoding="utf-8",
)


print()
print("=" * 72)
print("FD001 reconciliation completed successfully.")
print("=" * 72)

print(
    "Corrected regression MAE:",
    float(
        corrected_metrics["MAE"]
    ),
)

print(
    "Corrected regression RMSE:",
    float(
        corrected_metrics["RMSE"]
    ),
)

print(
    "Corrected NASA score:",
    float(
        corrected_metrics["NASA score"]
    ),
)

print(
    "Corrected 95% coverage:",
    rebuilt_95_coverage,
)

print(
    "Corrected 95% average width:",
    rebuilt_95_average_width,
)

print(
    "Changed maintenance actions:",
    changed_actions,
)

print(
    "Artifact version:",
    manifest["artifact_version"],
)

print(
    "New conformal SHA-256:",
    conformal_manifest_entry[
        "sha256"
    ],
)
