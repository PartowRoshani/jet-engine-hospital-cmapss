from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

project_root_string = str(
    PROJECT_ROOT
)

if project_root_string not in sys.path:
    sys.path.insert(
        0,
        project_root_string,
    )


from src.artifact_loader import (
    FD001ArtifactBundle,
    load_fd001_artifacts,
)
from src.inference import (
    get_latest_fd001_status,
    run_fd001_inference,
)


st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)


ACTION_ORDER = [
    "CONTINUE",
    "INSPECT",
    "STOP",
]


ACTION_ICONS = {
    "CONTINUE": "🟢",
    "INSPECT": "🟠",
    "STOP": "🔴",
}


ACTION_LABELS = {
    "CONTINUE": "Continue operation",
    "INSPECT": "Schedule inspection",
    "STOP": "Stop and maintain",
}


@st.cache_resource
def load_artifact_bundle() -> FD001ArtifactBundle:
    """
    Load and verify the locked FD001 artifact bundle.
    """
    return load_fd001_artifacts(
        project_root=PROJECT_ROOT,
        verify_checksums=True,
    )


@st.cache_data
def load_demo_inference() -> pd.DataFrame:
    """
    Load the saved realistic FD001 inference history.
    """
    demo_path = (
        PROJECT_ROOT
        / "reports"
        / "tables"
        / "fd001_validation_snapshot_inference.csv"
    )

    if not demo_path.exists():
        raise FileNotFoundError(
            "Demo inference file not found: "
            f"{demo_path}"
        )

    return pd.read_csv(
        demo_path
    )


def format_probability(
    value: float,
) -> str:
    """
    Format probability as a percentage.
    """
    return f"{100.0 * float(value):.1f}%"


def format_number(
    value: float,
    digits: int = 1,
) -> str:
    """
    Format numeric values safely.
    """
    if pd.isna(value):
        return "N/A"

    return f"{float(value):.{digits}f}"


def style_action_table(
    data: pd.DataFrame,
):
    """
    Highlight maintenance actions in a Streamlit table.
    """
    def highlight_action(
        value: object,
    ) -> str:
        if value == "STOP":
            return (
                "background-color: #f8d7da; "
                "color: #721c24; "
                "font-weight: bold;"
            )

        if value == "INSPECT":
            return (
                "background-color: #fff3cd; "
                "color: #856404; "
                "font-weight: bold;"
            )

        if value == "CONTINUE":
            return (
                "background-color: #d4edda; "
                "color: #155724; "
                "font-weight: bold;"
            )

        return ""

    return data.style.map(
        highlight_action,
        subset=["Action"],
    )


def build_latest_engine_table(
    inference_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create one current-status row per engine.
    """
    latest_df = get_latest_fd001_status(
        inference_df
    ).copy()

    # Compatibility with older saved demo files.
    if "Action level" not in latest_df.columns:
        if "Action" not in latest_df.columns:
            raise ValueError(
                "The inference data does not contain "
                "'Action' or 'Action level'."
            )

        latest_df["Action level"] = (
            latest_df["Action"]
            .map(
                {
                    "CONTINUE": 0,
                    "INSPECT": 1,
                    "STOP": 2,
                }
            )
        )

    if latest_df["Action level"].isna().any():
        invalid_actions = (
            latest_df.loc[
                latest_df[
                    "Action level"
                ].isna(),
                "Action",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unknown maintenance actions found: "
            f"{invalid_actions}"
        )

    # Sort before removing the Action level column.
    latest_df = (
        latest_df
        .sort_values(
            by=[
                "Action level",
                "probability_10",
            ],
            ascending=[
                False,
                False,
            ],
        )
        .reset_index(drop=True)
    )

    selected_columns = [
        "engine_id",
        "cycle",
    ]

    if "RUL" in latest_df.columns:
        selected_columns.append(
            "RUL"
        )

    selected_columns.extend(
        [
            "RUL prediction",
            "RUL lower",
            "RUL upper",
            "probability_10",
            "probability_20",
            "probability_30",
            "anomaly_percentile",
            "Action",
            "Confidence",
            "Trigger",
            "Next review cycles",
            "Signal disagreement",
        ]
    )

    return latest_df[
        selected_columns
    ].copy()


def validate_uploaded_columns(
    uploaded_df: pd.DataFrame,
    bundle: FD001ArtifactBundle,
) -> list[str]:
    """
    Return any columns required by the saved models
    that are missing from the uploaded DataFrame.
    """
    required_columns = {
        "engine_id",
        "cycle",
    }

    required_columns.update(
        bundle.feature_schema[
            "regression"
        ]["columns"]
    )

    required_columns.update(
        bundle.feature_schema[
            "classification"
        ]["columns"]
    )

    required_columns.update(
        bundle.feature_schema[
            "anomaly_detection"
        ]["columns"]
    )

    return sorted(
        required_columns.difference(
            uploaded_df.columns
        )
    )


def show_overview_metrics(
    latest_df: pd.DataFrame,
) -> None:
    """
    Display fleet-level maintenance summary.
    """
    total_engines = int(
        latest_df["engine_id"].nunique()
    )

    action_counts = (
        latest_df["Action"]
        .value_counts()
        .reindex(
            ACTION_ORDER,
            fill_value=0,
        )
    )

    metric_columns = st.columns(
        4
    )

    metric_columns[0].metric(
        "Engines monitored",
        total_engines,
    )

    metric_columns[1].metric(
        "Continue",
        int(
            action_counts["CONTINUE"]
        ),
    )

    metric_columns[2].metric(
        "Inspect",
        int(
            action_counts["INSPECT"]
        ),
    )

    metric_columns[3].metric(
        "Stop",
        int(
            action_counts["STOP"]
        ),
    )


def show_selected_engine(
    inference_df: pd.DataFrame,
    engine_id: int | float,
    anomaly_threshold: float,
) -> None:
    """
    Display detailed history and current decision for one engine.
    """
    engine_df = (
        inference_df.loc[
            inference_df[
                "engine_id"
            ] == engine_id
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
        .copy()
    )

    if engine_df.empty:
        st.warning(
            "No history is available for the selected engine."
        )
        return

    latest_row = engine_df.iloc[-1]

    action = str(
        latest_row["Action"]
    )

    action_icon = ACTION_ICONS.get(
        action,
        "⚪",
    )

    st.subheader(
        f"{action_icon} Engine {engine_id}: "
        f"{ACTION_LABELS.get(action, action)}"
    )

    st.caption(
        f"Current trigger: {latest_row['Trigger']}"
    )

    metric_columns = st.columns(
        5
    )

    metric_columns[0].metric(
        "Current cycle",
        int(
            latest_row["cycle"]
        ),
    )

    metric_columns[1].metric(
        "Predicted RUL",
        format_number(
            latest_row[
                "RUL prediction"
            ]
        ),
        help="Estimated remaining useful life in cycles.",
    )

    metric_columns[2].metric(
        "95% RUL interval",
        (
            f"{format_number(latest_row['RUL lower'])}"
            " – "
            f"{format_number(latest_row['RUL upper'])}"
        ),
    )

    metric_columns[3].metric(
        "Failure ≤10 cycles",
        format_probability(
            latest_row[
                "probability_10"
            ]
        ),
    )

    metric_columns[4].metric(
        "Anomaly percentile",
        format_probability(
            latest_row[
                "anomaly_percentile"
            ]
        ),
    )

    status_columns = st.columns(
        3
    )

    status_columns[0].info(
        "Confidence\n\n"
        f"**{latest_row['Confidence']}**"
    )

    status_columns[1].info(
        "Next review\n\n"
        f"**{int(latest_row['Next review cycles'])} cycles**"
    )

    disagreement_text = (
        "Yes"
        if bool(
            latest_row[
                "Signal disagreement"
            ]
        )
        else "No"
    )

    status_columns[2].info(
        "Signal disagreement\n\n"
        f"**{disagreement_text}**"
    )

    chart_tab_1, chart_tab_2, chart_tab_3, history_tab = (
        st.tabs(
            [
                "RUL trajectory",
                "Failure probabilities",
                "Anomaly history",
                "Decision history",
            ]
        )
    )

    with chart_tab_1:
        rul_columns = [
            "cycle",
            "RUL prediction",
            "RUL lower",
            "RUL upper",
        ]

        if "RUL" in engine_df.columns:
            rul_columns.insert(
                1,
                "RUL",
            )

        rul_chart_df = (
            engine_df[
                rul_columns
            ]
            .set_index(
                "cycle"
            )
        )

        st.line_chart(
            rul_chart_df
        )

        st.caption(
            "The interval shows the locked 95% "
            "conformal uncertainty range."
        )

    with chart_tab_2:
        probability_chart_df = (
            engine_df[
                [
                    "cycle",
                    "probability_10",
                    "probability_20",
                    "probability_30",
                ]
            ]
            .set_index(
                "cycle"
            )
        )

        st.line_chart(
            probability_chart_df
        )

        st.caption(
            "Probabilities are calibrated and corrected so that "
            "P10 ≤ P20 ≤ P30."
        )

    with chart_tab_3:
        anomaly_chart_df = (
            engine_df[
                [
                    "cycle",
                    "anomaly_percentile",
                ]
            ]
            .copy()
        )

        anomaly_chart_df[
            "locked_threshold"
        ] = anomaly_threshold

        st.line_chart(
            anomaly_chart_df.set_index(
                "cycle"
            )
        )

        st.caption(
            "Anomaly is advisory evidence and does not "
            "independently trigger STOP."
        )

    with history_tab:
        history_columns = [
            "cycle",
            "RUL prediction",
            "probability_10",
            "probability_20",
            "probability_30",
            "anomaly_percentile",
            "Action",
            "Trigger",
            "Hysteresis applied",
        ]

        history_columns = [
            column
            for column in history_columns
            if column in engine_df.columns
        ]

        st.dataframe(
            style_action_table(
                engine_df[
                    history_columns
                ].tail(
                    30
                )
            ),
            use_container_width=True,
            hide_index=True,
        )


def main() -> None:
    st.title(
        "✈️ Jet Engine Hospital"
    )

    st.write(
        "Predictive-maintenance dashboard for NASA C-MAPSS FD001."
    )

    st.caption(
        "RUL prediction, calibrated failure risks, anomaly "
        "detection, uncertainty, and maintenance decisions."
    )

    try:
        bundle = load_artifact_bundle()

    except Exception as error:
        st.error(
            "The FD001 artifact bundle could not be loaded."
        )

        st.exception(
            error
        )

        st.stop()

    with st.sidebar:
        st.header(
            "Dashboard controls"
        )

        data_source = st.radio(
            "Data source",
            options=[
                "Demo snapshot",
                "Upload engineered CSV",
            ],
            index=0,
        )

        st.divider()

        st.subheader(
            "Locked configuration"
        )

        horizons = bundle.classification_config[
            "horizons"
        ]

        st.write(
            "Risk thresholds:"
        )

        st.code(
            "\n".join(
                [
                    (
                        "10-cycle: "
                        f"{horizons['10']['threshold']}"
                    ),
                    (
                        "20-cycle: "
                        f"{horizons['20']['threshold']}"
                    ),
                    (
                        "30-cycle: "
                        f"{horizons['30']['threshold']}"
                    ),
                ]
            )
        )

        st.write(
            "Persistence: **2 of 3 cycles**"
        )

        st.write(
            "Policy: **supervised_only**"
        )

        st.write(
            "Hysteresis: **enabled**"
        )

        st.write(
            "Artifact version: "
            f"**{bundle.manifest['artifact_version']}**"
        )

    if data_source == "Demo snapshot":
        try:
            inference_df = load_demo_inference()

        except Exception as error:
            st.error(
                "The demonstration data could not be loaded."
            )

            st.exception(
                error
            )

            st.stop()

        st.info(
            "Demo mode uses validation engines stopped at "
            "different stages of their useful life."
        )

    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload engineered feature CSV",
            type=[
                "csv",
            ],
        )

        st.sidebar.caption(
            "The file must include engine_id, cycle, "
            "and all 138 engineered model features."
        )

        if uploaded_file is None:
            st.warning(
                "Upload an engineered FD001 CSV to run inference."
            )

            with st.expander(
                "Required feature schema"
            ):
                st.write(
                    "Classification features:"
                )

                st.code(
                    "\n".join(
                        bundle.feature_schema[
                            "classification"
                        ]["columns"]
                    )
                )

            st.stop()

        try:
            uploaded_df = pd.read_csv(
                uploaded_file
            )

        except Exception as error:
            st.error(
                "The uploaded CSV could not be read."
            )

            st.exception(
                error
            )

            st.stop()

        missing_columns = validate_uploaded_columns(
            uploaded_df=uploaded_df,
            bundle=bundle,
        )

        if missing_columns:
            st.error(
                "The uploaded file is missing "
                f"{len(missing_columns)} required columns."
            )

            st.code(
                "\n".join(
                    missing_columns
                )
            )

            st.stop()

        try:
            with st.spinner(
                "Running the FD001 inference pipeline..."
            ):
                inference_df = run_fd001_inference(
                    bundle=bundle,
                    feature_df=uploaded_df,
                    model_scope="evaluation",
                )

        except Exception as error:
            st.error(
                "Inference failed."
            )

            st.exception(
                error
            )

            st.stop()

        st.success(
            "Inference completed successfully."
        )

    latest_df = build_latest_engine_table(
        inference_df
    )

    show_overview_metrics(
        latest_df
    )

    st.divider()

    available_actions = [
        action
        for action in ACTION_ORDER
        if action in latest_df[
            "Action"
        ].unique()
    ]

    selected_actions = st.multiselect(
        "Filter by maintenance action",
        options=available_actions,
        default=available_actions,
    )

    filtered_latest_df = latest_df.loc[
        latest_df[
            "Action"
        ].isin(
            selected_actions
        )
    ].copy()

    table_columns = [
        "engine_id",
        "cycle",
    ]

    if "RUL" in filtered_latest_df.columns:
        table_columns.append(
            "RUL"
        )

    table_columns.extend(
        [
            "RUL prediction",
            "RUL lower",
            "RUL upper",
            "probability_10",
            "probability_20",
            "probability_30",
            "anomaly_percentile",
            "Action",
            "Confidence",
            "Trigger",
            "Next review cycles",
        ]
    )

    st.subheader(
        "Fleet maintenance status"
    )

    st.dataframe(
        style_action_table(
            filtered_latest_df[
                table_columns
            ]
        ),
        use_container_width=True,
        hide_index=True,
    )

    latest_csv = filtered_latest_df.to_csv(
        index=False
    ).encode(
        "utf-8"
    )

    st.download_button(
        label="Download current fleet status",
        data=latest_csv,
        file_name=(
            "fd001_current_fleet_status.csv"
        ),
        mime="text/csv",
    )

    st.divider()

    engine_options = (
        filtered_latest_df[
            "engine_id"
        ]
        .tolist()
    )

    if not engine_options:
        st.warning(
            "No engines match the selected action filter."
        )

        st.stop()

    selected_engine = st.selectbox(
        "Select an engine for detailed analysis",
        options=engine_options,
    )

    anomaly_threshold = float(
        bundle.anomaly_config[
            "percentile_threshold"
        ]
    )

    show_selected_engine(
        inference_df=inference_df,
        engine_id=selected_engine,
        anomaly_threshold=anomaly_threshold,
    )

    st.divider()

    with st.expander(
        "Model scope and safety limitations"
    ):
        st.markdown(
            """
            - This dashboard uses the locked FD001 evaluation
              inference pipeline.
            - The maintenance policy was selected on validation
              engines and evaluated once on internal-test engines.
            - Anomaly and RUL uncertainty are advisory evidence.
            - STOP is triggered by persistent calibrated
              10-cycle failure risk.
            - INSPECT is triggered by persistent calibrated
              20-cycle or 30-cycle risk.
            - Hysteresis prevents an action from becoming less
              severe without a maintenance reset.
            - The system is a decision-support prototype and
              does not replace certified maintenance procedures.
            """
        )


if __name__ == "__main__":
    main()