from src.data_loading import (
    add_training_targets,
    load_cmapss_subset,
)
from src.data_splitting import (
    apply_engine_splits,
    save_engine_splits,
    split_engine_ids,
)


def main() -> None:
    train_df, _, _ = load_cmapss_subset("FD001")

    labeled_train_df = add_training_targets(train_df)

    engine_splits = split_engine_ids(labeled_train_df)

    split_dataframes = apply_engine_splits(
        labeled_train_df,
        engine_splits,
    )

    output_path = save_engine_splits(
        engine_splits,
        subset="FD001",
    )

    print("=" * 70)
    print("FD001 ENGINE-LEVEL SPLIT SUMMARY")
    print("=" * 70)

    for split_name in ["train", "validation", "test"]:
        split_df = split_dataframes[split_name]

        engine_count = split_df["engine_id"].nunique()
        row_count = len(split_df)

        sequence_lengths = (
            split_df.groupby("engine_id")["cycle"].max()
        )

        print(f"\n{split_name.upper()}")
        print("-" * 40)
        print(f"Engines: {engine_count}")
        print(f"Rows: {row_count}")
        print(
            "Sequence length "
            f"(min / median / max): "
            f"{sequence_lengths.min()} / "
            f"{sequence_lengths.median():.1f} / "
            f"{sequence_lengths.max()}"
        )

        print("Class balance:")

        for horizon in (10, 20, 30):
            column_name = f"failure_within_{horizon}"

            positive_rate = (
                split_df[column_name].mean() * 100
            )

            print(
                f"  Failure within {horizon}: "
                f"{positive_rate:.2f}% positive"
            )

    train_ids = set(engine_splits["train"])
    validation_ids = set(engine_splits["validation"])
    test_ids = set(engine_splits["test"])

    print("\nOverlap checks:")
    print(
        "Train ∩ Validation:",
        len(train_ids & validation_ids),
    )
    print(
        "Train ∩ Test:",
        len(train_ids & test_ids),
    )
    print(
        "Validation ∩ Test:",
        len(validation_ids & test_ids),
    )

    print(f"\nSaved split file: {output_path}")
    print("\nEngine-level splitting completed successfully.")


if __name__ == "__main__":
    main()