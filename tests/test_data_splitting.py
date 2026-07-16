from src.data_loading import (
    add_training_targets,
    load_cmapss_subset,
)
from src.data_splitting import (
    apply_engine_splits,
    split_engine_ids,
)


def get_fd001_splits():
    train_df, _, _ = load_cmapss_subset("FD001")
    labeled_df = add_training_targets(train_df)

    splits = split_engine_ids(labeled_df)

    return labeled_df, splits


def test_split_engine_counts() -> None:
    _, splits = get_fd001_splits()

    assert len(splits["train"]) == 70
    assert len(splits["validation"]) == 15
    assert len(splits["test"]) == 15


def test_split_engine_ids_are_disjoint() -> None:
    _, splits = get_fd001_splits()

    train_ids = set(splits["train"])
    validation_ids = set(splits["validation"])
    test_ids = set(splits["test"])

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(test_ids)
    assert validation_ids.isdisjoint(test_ids)


def test_all_engines_are_assigned() -> None:
    dataframe, splits = get_fd001_splits()

    expected_ids = set(dataframe["engine_id"].unique())

    assigned_ids = (
        set(splits["train"])
        | set(splits["validation"])
        | set(splits["test"])
    )

    assert assigned_ids == expected_ids


def test_all_rows_are_preserved() -> None:
    dataframe, splits = get_fd001_splits()

    split_dataframes = apply_engine_splits(
        dataframe,
        splits,
    )

    total_split_rows = sum(
        len(split_df)
        for split_df in split_dataframes.values()
    )

    assert total_split_rows == len(dataframe)


def test_split_is_reproducible() -> None:
    dataframe, first_splits = get_fd001_splits()

    second_splits = split_engine_ids(dataframe)

    assert first_splits == second_splits