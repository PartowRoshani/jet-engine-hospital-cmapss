from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, SPLITS_DIR


def split_engine_ids(
    dataframe: pd.DataFrame,
    train_size: float = 0.70,
    validation_size: float = 0.15,
    test_size: float = 0.15,
    random_state: int = RANDOM_STATE,
) -> dict[str, list[int]]:
    """
    Split unique engine IDs into train, validation, and test sets.

    All rows belonging to one engine remain in exactly one split.
    """

    if "engine_id" not in dataframe.columns:
        raise ValueError("The dataframe must contain an 'engine_id' column.")

    total_ratio = train_size + validation_size + test_size

    if abs(total_ratio - 1.0) > 1e-9:
        raise ValueError(
            "train_size, validation_size, and test_size must sum to 1."
        )

    engine_ids = sorted(
        int(engine_id)
        for engine_id in dataframe["engine_id"].unique()
    )

    if len(engine_ids) < 3:
        raise ValueError("At least three engines are required for splitting.")

    # First split: 70% train and 30% temporary
    train_ids, temporary_ids = train_test_split(
        engine_ids,
        train_size=train_size,
        random_state=random_state,
        shuffle=True,
    )

    # Split the remaining 30% equally into validation and test
    validation_fraction = validation_size / (
        validation_size + test_size
    )

    validation_ids, test_ids = train_test_split(
        temporary_ids,
        train_size=validation_fraction,
        random_state=random_state,
        shuffle=True,
    )

    splits = {
        "train": sorted(int(value) for value in train_ids),
        "validation": sorted(
            int(value) for value in validation_ids
        ),
        "test": sorted(int(value) for value in test_ids),
    }

    validate_engine_splits(
        splits=splits,
        all_engine_ids=set(engine_ids),
    )

    return splits


def validate_engine_splits(
    splits: dict[str, list[int]],
    all_engine_ids: set[int],
) -> None:
    """
    Confirm that split engine IDs are disjoint and complete.
    """

    train_set = set(splits["train"])
    validation_set = set(splits["validation"])
    test_set = set(splits["test"])

    if train_set & validation_set:
        raise ValueError(
            "Some engines appear in both train and validation."
        )

    if train_set & test_set:
        raise ValueError(
            "Some engines appear in both train and test."
        )

    if validation_set & test_set:
        raise ValueError(
            "Some engines appear in both validation and test."
        )

    combined_ids = train_set | validation_set | test_set

    if combined_ids != all_engine_ids:
        missing_ids = all_engine_ids - combined_ids
        extra_ids = combined_ids - all_engine_ids

        raise ValueError(
            f"Split IDs are incomplete. "
            f"Missing: {sorted(missing_ids)}, "
            f"Extra: {sorted(extra_ids)}"
        )


def apply_engine_splits(
    dataframe: pd.DataFrame,
    splits: dict[str, list[int]],
) -> dict[str, pd.DataFrame]:
    """
    Create one dataframe for each engine-level split.
    """

    split_dataframes: dict[str, pd.DataFrame] = {}

    for split_name, engine_ids in splits.items():
        split_dataframe = dataframe.loc[
            dataframe["engine_id"].isin(engine_ids)
        ].copy()

        split_dataframes[split_name] = (
            split_dataframe
            .sort_values(["engine_id", "cycle"])
            .reset_index(drop=True)
        )

    return split_dataframes


def save_engine_splits(
    splits: dict[str, list[int]],
    subset: str = "FD001",
    output_directory: Path = SPLITS_DIR,
) -> Path:
    """
    Save engine IDs to a JSON file for reproducibility.
    """

    output_directory.mkdir(parents=True, exist_ok=True)

    output_path = (
        output_directory
        / f"{subset.lower()}_engine_splits.json"
    )

    payload = {
        "subset": subset.upper(),
        "random_state": RANDOM_STATE,
        "ratios": {
            "train": 0.70,
            "validation": 0.15,
            "test": 0.15,
        },
        "engine_ids": splits,
    }

    with output_path.open(
        mode="w",
        encoding="utf-8",
    ) as file:
        json.dump(payload, file, indent=4)

    return output_path