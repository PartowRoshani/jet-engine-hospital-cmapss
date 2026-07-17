import numpy as np
import pandas as pd
import pytest

from src.data_loading import (
    add_training_targets,
    load_cmapss_subset,
)
from src.data_splitting import (
    apply_engine_splits,
    split_engine_ids,
)
from src.feature_engineering import (
    build_time_series_features,
    get_model_feature_columns,
)


@pytest.fixture(scope="module")
def engineered_fd001():
    train_df, _, _ = load_cmapss_subset("FD001")
    labeled_df = add_training_targets(train_df)

    engine_splits = split_engine_ids(labeled_df)

    split_dataframes = apply_engine_splits(
        labeled_df,
        engine_splits,
    )

    train_split_df = split_dataframes["train"]

    sensor_columns = [
        column
        for column in train_split_df.columns
        if column.startswith("sensor_")
    ]

    candidate_sensors = [
        column
        for column in sensor_columns
        if train_split_df[column].nunique() > 1
    ]

    engineered_df = build_time_series_features(
        train_split_df,
        sensor_columns=candidate_sensors,
    )

    return train_split_df, engineered_df, candidate_sensors


def test_row_count_is_preserved(engineered_fd001) -> None:
    original_df, engineered_df, _ = engineered_fd001

    assert len(engineered_df) == len(original_df)
    assert engineered_df.shape == (14507, 151)


def test_expected_number_of_temporal_features(
    engineered_fd001,
) -> None:
    _, engineered_df, candidate_sensors = engineered_fd001

    generated_columns = [
        column
        for column in engineered_df.columns
        if (
            "_delta_" in column
            or "_mean_" in column
            or "_std_" in column
        )
    ]

    assert len(candidate_sensors) == 15
    assert len(generated_columns) == 120


def test_generated_features_are_finite(
    engineered_fd001,
) -> None:
    _, engineered_df, _ = engineered_fd001

    generated_columns = [
        column
        for column in engineered_df.columns
        if (
            "_delta_" in column
            or "_mean_" in column
            or "_std_" in column
        )
    ]

    generated_values = engineered_df[
        generated_columns
    ].to_numpy()

    assert not np.isnan(generated_values).any()
    assert np.isfinite(generated_values).all()


def test_first_engine_cycles_have_zero_deltas(
    engineered_fd001,
) -> None:
    _, engineered_df, _ = engineered_fd001

    delta_columns = [
        column
        for column in engineered_df.columns
        if "_delta_" in column
    ]

    first_rows = (
        engineered_df
        .groupby("engine_id")
        .head(1)
    )

    assert np.allclose(
        first_rows[delta_columns].to_numpy(),
        0.0,
    )


def test_future_observations_do_not_change_past_features() -> None:
    original = pd.DataFrame(
        {
            "engine_id": [1, 1, 1, 1, 1, 1],
            "cycle": [1, 2, 3, 4, 5, 6],
            "sensor_2": [10, 11, 12, 13, 14, 15],
        }
    )

    modified_future = original.copy()
    modified_future.loc[
        modified_future["cycle"] == 6,
        "sensor_2",
    ] = 1000

    original_features = build_time_series_features(
        original,
        sensor_columns=["sensor_2"],
        rolling_windows=(3,),
        change_periods=(1,),
    )

    modified_features = build_time_series_features(
        modified_future,
        sensor_columns=["sensor_2"],
        rolling_windows=(3,),
        change_periods=(1,),
    )

    temporal_columns = [
        "sensor_2_delta_1",
        "sensor_2_mean_3",
        "sensor_2_std_3",
    ]

    pd.testing.assert_frame_equal(
        original_features.loc[
            original_features["cycle"] <= 5,
            temporal_columns,
        ].reset_index(drop=True),
        modified_features.loc[
            modified_features["cycle"] <= 5,
            temporal_columns,
        ].reset_index(drop=True),
    )