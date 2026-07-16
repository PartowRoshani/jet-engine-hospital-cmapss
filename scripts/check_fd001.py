from src.data_loading import add_training_targets, load_cmapss_subset


def main() -> None:
    train_df, test_df, test_rul_df = load_cmapss_subset("FD001")

    train_labeled = add_training_targets(train_df)

    print("=" * 60)
    print("FD001 LOADING SUMMARY")
    print("=" * 60)

    print(f"Training shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    print(f"Test RUL shape: {test_rul_df.shape}")

    print(f"Training engines: {train_df['engine_id'].nunique()}")
    print(f"Test engines: {test_df['engine_id'].nunique()}")

    print("\nColumn names:")
    print(train_df.columns.tolist())

    print("\nFirst five labeled training rows:")
    print(
        train_labeled[
            [
                "engine_id",
                "cycle",
                "final_cycle",
                "RUL",
                "failure_within_10",
                "failure_within_20",
                "failure_within_30",
            ]
        ].head()
    )

    print("\nLast five cycles of engine 1:")
    print(
        train_labeled.loc[
            train_labeled["engine_id"] == 1,
            [
                "engine_id",
                "cycle",
                "RUL",
                "failure_within_10",
                "failure_within_20",
                "failure_within_30",
            ],
        ].tail()
    )

    print("\nMissing values:")
    print(train_df.isna().sum().sum())

    print("\nDuplicate engine-cycle keys:")
    print(train_df[["engine_id", "cycle"]].duplicated().sum())

    print("\nFD001 data loading completed successfully.")


if __name__ == "__main__":
    main()