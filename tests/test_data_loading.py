from src.data_loading import add_training_targets, load_cmapss_subset


def test_fd001_shapes() -> None:
    train_df, test_df, test_rul_df = load_cmapss_subset("FD001")

    assert train_df.shape == (20631, 26)
    assert test_df.shape == (13096, 26)
    assert test_rul_df.shape == (100, 1)


def test_fd001_engine_counts() -> None:
    train_df, test_df, _ = load_cmapss_subset("FD001")

    assert train_df["engine_id"].nunique() == 100
    assert test_df["engine_id"].nunique() == 100


def test_training_rul_ends_at_zero() -> None:
    train_df, _, _ = load_cmapss_subset("FD001")

    labeled_df = add_training_targets(train_df)
    final_rows = labeled_df.groupby("engine_id").tail(1)

    assert (final_rows["RUL"] == 0).all()


def test_failure_horizon_order() -> None:
    train_df, _, _ = load_cmapss_subset("FD001")

    labeled_df = add_training_targets(train_df)

    assert (
        labeled_df["failure_within_10"]
        <= labeled_df["failure_within_20"]
    ).all()

    assert (
        labeled_df["failure_within_20"]
        <= labeled_df["failure_within_30"]
    ).all()
