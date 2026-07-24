from pathlib import Path

import pandas as pd

from src.config import CMAPSS_COLUMNS, FAILURE_HORIZONS, RAW_DATA_DIR


VALID_SUBSETS = {"FD001", "FD002", "FD003", "FD004"}


def load_cmapss_subset(
    subset: str = "FD001",
    raw_data_dir: Path = RAW_DATA_DIR,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load one NASA C-MAPSS subset.

    Parameters
    ----------
    subset:
        One of FD001, FD002, FD003, or FD004.
    raw_data_dir:
        Directory containing the original NASA text files.

    Returns
    -------
    train_df:
        Training trajectories ending at engine failure.
    test_df:
        Test trajectories stopping before failure.
    test_rul_df:
        Remaining useful life after the final observed test cycle.
    """

    subset = subset.upper()

    if subset not in VALID_SUBSETS:
        raise ValueError(
            f"Invalid subset: {subset}. "
            f"Expected one of {sorted(VALID_SUBSETS)}."
        )

    train_path = raw_data_dir / f"train_{subset}.txt"
    test_path = raw_data_dir / f"test_{subset}.txt"
    rul_path = raw_data_dir / f"RUL_{subset}.txt"

    required_paths = [train_path, test_path, rul_path]

    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    train_df = pd.read_csv(
        train_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )

    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
        names=CMAPSS_COLUMNS,
    )

    test_rul_df = pd.read_csv(
        rul_path,
        sep=r"\s+",
        header=None,
        names=["final_RUL"],
    )

    # IDs and cycles must be integers
    for dataframe in (train_df, test_df):
        dataframe["engine_id"] = dataframe["engine_id"].astype(int)
        dataframe["cycle"] = dataframe["cycle"].astype(int)

    validate_loaded_data(train_df, test_df, test_rul_df, subset)

    return train_df, test_df, test_rul_df


def validate_loaded_data(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_rul_df: pd.DataFrame,
    subset: str,
) -> None:
    """
    Perform basic structural checks after loading.
    """

    expected_column_count = 26

    if train_df.shape[1] != expected_column_count:
        raise ValueError(
            f"{subset} training data has {train_df.shape[1]} columns; "
            f"expected {expected_column_count}."
        )

    if test_df.shape[1] != expected_column_count:
        raise ValueError(
            f"{subset} test data has {test_df.shape[1]} columns; "
            f"expected {expected_column_count}."
        )

    number_of_test_engines = test_df["engine_id"].nunique()

    if len(test_rul_df) != number_of_test_engines:
        raise ValueError(
            "The number of test RUL values does not match "
            "the number of test engines."
        )

    if train_df[["engine_id", "cycle"]].duplicated().any():
        raise ValueError(
            "Duplicate (engine_id, cycle) pairs found in training data."
        )

    if test_df[["engine_id", "cycle"]].duplicated().any():
        raise ValueError(
            "Duplicate (engine_id, cycle) pairs found in test data."
        )


def add_training_targets(
    train_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create training RUL and 10/20/30-cycle classification targets.

    RUL(i, t) = final_cycle(i) - t
    """

    result = train_df.copy()

    result["final_cycle"] = result.groupby(
        "engine_id"
    )["cycle"].transform("max")

    result["RUL"] = result["final_cycle"] - result["cycle"]

    for horizon in FAILURE_HORIZONS:
        result[f"failure_within_{horizon}"] = (
            result["RUL"] <= horizon
        ).astype(int)

    return result