from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st


# ------------------------------------------------------------------
# Application paths
# ------------------------------------------------------------------

APP_DIR = Path(
    __file__
).resolve().parent


PROJECT_ROOT = (
    APP_DIR.parent
)


for import_path in [
    APP_DIR,
    PROJECT_ROOT,
]:

    import_path_string = str(
        import_path
    )

    if (
        import_path_string
        not in sys.path
    ):

        sys.path.insert(
            0,
            import_path_string,
        )


# ------------------------------------------------------------------
# Existing FD001 and FD003 application pipelines
# ------------------------------------------------------------------

from app import (
    create_trajectory_figure,
    read_engine_file,
    run_fd001_pipeline,
    run_fd003_pipeline,
)


# ------------------------------------------------------------------
# Standalone FD004 deployment pipeline
# ------------------------------------------------------------------

try:

    from fd004_inference import (
        create_fd004_terminal_summary,
        load_fd004_artifact,
        predict_fd004_trajectory,
    )

except ImportError:

    from src.fd004_inference import (
        create_fd004_terminal_summary,
        load_fd004_artifact,
        predict_fd004_trajectory,
    )


# ------------------------------------------------------------------
# Streamlit page configuration
# ------------------------------------------------------------------

st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
)


# ------------------------------------------------------------------
# FD004 artifact loading
# ------------------------------------------------------------------

FD004_ARTIFACT_CANDIDATES = [
    (
        APP_DIR
        / "artifacts"
        / "fd004_artifact.joblib"
    ),
    (
        PROJECT_ROOT
        / "artifacts"
        / "fd004_artifact.joblib"
    ),
    (
        APP_DIR
        / "fd004_artifact.joblib"
    ),
]


@st.cache_resource
def load_streamlit_fd004_artifact():
    """
    Locate and load the frozen FD004 deployment artifact.
    """
    for artifact_candidate in (
        FD004_ARTIFACT_CANDIDATES
    ):

        if artifact_candidate.exists():

            artifact = load_fd004_artifact(
                artifact_candidate
            )

            return (
                artifact,
                artifact_candidate,
            )

    raise FileNotFoundError(
        "The FD004 artifact could not be found. "
        "Checked paths:\n"
        + "\n".join(
            str(path)
            for path in (
                FD004_ARTIFACT_CANDIDATES
            )
        )
    )


# ------------------------------------------------------------------
# FD004 interface conversion
# ------------------------------------------------------------------

FD004_STATE_TO_ACTION = {
    "Normal": "CONTINUE",
    "Watch": "INSPECT",
    "Warning": "INSPECT",
    "Critical": "STOP",
}


FD004_STATE_TO_NEXT_REVIEW = {
    "Normal": 10,
    "Watch": 5,
    "Warning": 3,
    "Critical": 0,
}


def run_fd004_pipeline(
    raw_frame: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Run complete raw-to-policy FD004 inference.

    The returned tables use aliases compatible with the existing
    FD001/FD003 Streamlit plotting interface while retaining all
    native FD004 outputs.
    """
    (
        fd004_artifact,
        _,
    ) = load_streamlit_fd004_artifact()


    native_timeline = (
        predict_fd004_trajectory(
            raw_trajectory=(
                raw_frame
            ),
            artifact_or_path=(
                fd004_artifact
            ),
            require_contiguous_cycles=True,
            return_feature_frame=False,
        )
    )


    results = (
        native_timeline
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .reset_index(drop=True)
        .copy()
    )


    # --------------------------------------------------------------
    # Common dashboard aliases
    # --------------------------------------------------------------

    results[
        "RUL prediction"
    ] = results[
        "locked_RUL_prediction"
    ]


    results[
        "RUL lower"
    ] = results[
        "RUL_lower"
    ]


    results[
        "RUL upper"
    ] = results[
        "RUL_upper"
    ]


    results[
        "Policy state"
    ] = results[
        "policy_state"
    ]


    results[
        "Action"
    ] = (
        results[
            "policy_state"
        ]
        .map(
            FD004_STATE_TO_ACTION
        )
    )


    if results[
        "Action"
    ].isna().any():

        unknown_states = (
            results.loc[
                results[
                    "Action"
                ].isna(),
                "policy_state",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unknown FD004 policy states: "
            f"{unknown_states}"
        )


    results[
        "Trigger"
    ] = results[
        "policy_reason"
    ]


    results[
        "Next review cycles"
    ] = (
        results[
            "policy_state"
        ]
        .map(
            FD004_STATE_TO_NEXT_REVIEW
        )
        .astype(int)
    )


    supervised_alert_active = (
        results[
            [
                "alert_10",
                "alert_20",
                "alert_30",
            ]
        ]
        .max(axis=1)
        .astype(bool)
    )


    results[
        "Signal disagreement"
    ] = (
        supervised_alert_active
        !=
        results[
            "anomaly_alert"
        ]
        .astype(bool)
    )


    results[
        "Confidence"
    ] = np.select(
        condlist=[
            results[
                "policy_state"
            ].eq(
                "Critical"
            ),
            results[
                "policy_state"
            ].eq(
                "Warning"
            ),
            (
                results[
                    "policy_state"
                ].eq(
                    "Watch"
                )
                &
                results[
                    "alert_30"
                ].eq(1)
            ),
            (
                results[
                    "policy_state"
                ].eq(
                    "Watch"
                )
                &
                results[
                    "anomaly_alert"
                ].eq(1)
            ),
        ],
        choicelist=[
            results[
                "probability_10"
            ],
            results[
                "probability_20"
            ],
            results[
                "probability_30"
            ],
            np.nan,
        ],
        default=(
            1.0
            - results[
                "probability_30"
            ]
        ),
    )


    # --------------------------------------------------------------
    # One current-status row per engine
    # --------------------------------------------------------------

    native_terminal = (
        create_fd004_terminal_summary(
            results
        )
    )


    latest_rows = (
        results
        .sort_values(
            [
                "engine_id",
                "cycle",
            ]
        )
        .groupby(
            "engine_id",
            sort=True,
        )
        .tail(1)
        .reset_index(drop=True)
    )


    assert (
        len(
            latest_rows
        )
        == len(
            native_terminal
        )
    )


    dashboard_columns = [
        "engine_id",
        "cycle",
        "operating_regime",
        "operating_regime_distance",
        "RUL prediction",
        "RUL lower",
        "RUL upper",
        "probability_10",
        "probability_20",
        "probability_30",
        "anomaly_severity",
        "anomaly_alert",
        "Policy state",
        "Action",
        "Confidence",
        "Trigger",
        "Next review cycles",
        "Signal disagreement",
    ]


    dashboard_df = (
        latest_rows[
            dashboard_columns
        ]
        .copy()
    )


    return (
        results,
        dashboard_df,
        results,
    )


# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------

st.title(
    "✈️ Jet Engine Hospital"
)


st.caption(
    "Unified predictive-maintenance application for "
    "NASA C-MAPSS FD001, FD003, and FD004."
)


st.markdown(
    """
Upload a complete chronological engine trajectory and receive:

- Remaining Useful Life prediction;
- conformal uncertainty interval;
- 10-, 20-, and 30-cycle failure probabilities;
- anomaly evidence;
- a `CONTINUE`, `INSPECT`, or `STOP` recommendation.

**Dataset stages**

- **FD001 — Foundation:** one operating condition and one fault mode;
- **FD003 — Multi-fault:** one operating condition and two fault modes;
- **FD004 — Multi-condition and multi-fault:** six frozen operating regimes and two fault modes.

The uploaded history for every engine must begin at cycle 1 and contain
the 26 standard NASA C-MAPSS columns.

> Educational demonstration only. Not an aviation-certified maintenance system.
"""
)


# ------------------------------------------------------------------
# Dataset and upload controls
# ------------------------------------------------------------------

dataset_name = st.radio(
    "Dataset / project stage",
    options=[
        "FD001 — Foundation",
        "FD003 — Multi-fault",
        "FD004 — Multi-condition and multi-fault",
    ],
    horizontal=True,
)


uploaded_file = st.file_uploader(
    "Engine trajectory file",
    type=[
        "csv",
        "txt",
    ],
)


run_button = st.button(
    "Run inference",
    type="primary",
    use_container_width=True,
)


# ------------------------------------------------------------------
# Inference execution
# ------------------------------------------------------------------

if run_button:

    if uploaded_file is None:

        st.error(
            "Upload a CSV or C-MAPSS text file first."
        )

        st.stop()


    suffix = (
        Path(
            uploaded_file.name
        ).suffix
        or ".csv"
    )


    temporary_path = None


    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:

            temporary_file.write(
                uploaded_file.getbuffer()
            )

            temporary_path = (
                temporary_file.name
            )


        with st.spinner(
            "Running inference..."
        ):

            raw_frame = read_engine_file(
                temporary_path
            )


            if dataset_name.startswith(
                "FD001"
            ):

                (
                    timeline_df,
                    dashboard_df,
                    plot_frame,
                ) = run_fd001_pipeline(
                    raw_frame
                )

                short_dataset_name = (
                    "FD001"
                )


            elif dataset_name.startswith(
                "FD003"
            ):

                (
                    timeline_df,
                    dashboard_df,
                    plot_frame,
                ) = run_fd003_pipeline(
                    raw_frame
                )

                short_dataset_name = (
                    "FD003"
                )


            elif dataset_name.startswith(
                "FD004"
            ):

                (
                    timeline_df,
                    dashboard_df,
                    plot_frame,
                ) = run_fd004_pipeline(
                    raw_frame
                )

                short_dataset_name = (
                    "FD004"
                )


            else:

                raise ValueError(
                    "Unsupported dataset selection."
                )


        numeric_columns = (
            dashboard_df
            .select_dtypes(
                include=[
                    np.number,
                ]
            )
            .columns
        )


        dashboard_df = (
            dashboard_df.copy()
        )


        dashboard_df[
            numeric_columns
        ] = (
            dashboard_df[
                numeric_columns
            ]
            .round(4)
        )


        action_counts = (
            dashboard_df[
                "Action"
            ]
            .value_counts()
            .to_dict()
        )


        st.success(
            "Inference completed successfully."
        )


        # ----------------------------------------------------------
        # Fleet summary
        # ----------------------------------------------------------

        metric_columns = st.columns(
            5
        )


        metric_columns[0].metric(
            "Dataset",
            short_dataset_name,
        )


        metric_columns[1].metric(
            "Engines",
            int(
                dashboard_df[
                    "engine_id"
                ]
                .nunique()
            ),
        )


        metric_columns[2].metric(
            "CONTINUE",
            int(
                action_counts.get(
                    "CONTINUE",
                    0,
                )
            ),
        )


        metric_columns[3].metric(
            "INSPECT",
            int(
                action_counts.get(
                    "INSPECT",
                    0,
                )
            ),
        )


        metric_columns[4].metric(
            "STOP",
            int(
                action_counts.get(
                    "STOP",
                    0,
                )
            ),
        )


        # ----------------------------------------------------------
        # Latest status table
        # ----------------------------------------------------------

        st.subheader(
            "Latest maintenance status per engine"
        )


        st.dataframe(
            dashboard_df,
            use_container_width=True,
            hide_index=True,
        )


        # ----------------------------------------------------------
        # Selected-engine visualizations
        # ----------------------------------------------------------

        engine_options = (
            dashboard_df[
                "engine_id"
            ]
            .tolist()
        )


        selected_engine = st.selectbox(
            "Select an engine for trajectory analysis",
            options=engine_options,
        )


        selected_plot_frame = (
            plot_frame.loc[
                plot_frame[
                    "engine_id"
                ].eq(
                    selected_engine
                )
            ]
            .sort_values(
                "cycle"
            )
            .reset_index(drop=True)
        )


        rul_tab, risk_tab, decision_tab = (
            st.tabs(
                [
                    "RUL trajectory",
                    "Failure probabilities",
                    "Decision history",
                ]
            )
        )


        with rul_tab:

            trajectory_figure = (
                create_trajectory_figure(
                    selected_plot_frame,
                    dataset_name=(
                        short_dataset_name
                    ),
                )
            )


            st.pyplot(
                trajectory_figure,
                use_container_width=True,
            )


            plt.close(
                trajectory_figure
            )


        with risk_tab:

            probability_columns = [
                "cycle",
                "probability_10",
                "probability_20",
                "probability_30",
            ]


            available_probability_columns = [
                column_name
                for column_name in (
                    probability_columns
                )
                if column_name
                in selected_plot_frame.columns
            ]


            if (
                len(
                    available_probability_columns
                )
                == 4
            ):

                st.line_chart(
                    selected_plot_frame[
                        available_probability_columns
                    ]
                    .set_index(
                        "cycle"
                    )
                )

            else:

                st.info(
                    "Failure-probability history is not "
                    "available for this pipeline."
                )


            if (
                short_dataset_name
                == "FD004"
            ):

                anomaly_chart = (
                    selected_plot_frame[
                        [
                            "cycle",
                            "anomaly_severity",
                        ]
                    ]
                    .set_index(
                        "cycle"
                    )
                )


                st.caption(
                    "FD004 anomaly severity: larger values "
                    "indicate more abnormal behavior."
                )


                st.line_chart(
                    anomaly_chart
                )


        with decision_tab:

            decision_columns = [
                "cycle",
                "Action",
                "Trigger",
                "Next review cycles",
            ]


            if (
                short_dataset_name
                == "FD004"
            ):

                decision_columns.extend(
                    [
                        "Policy state",
                        "anomaly_alert",
                        "operating_regime",
                    ]
                )


            decision_columns = [
                column_name
                for column_name in (
                    decision_columns
                )
                if column_name
                in selected_plot_frame.columns
            ]


            st.dataframe(
                selected_plot_frame[
                    decision_columns
                ]
                .tail(50),
                use_container_width=True,
                hide_index=True,
            )


        # ----------------------------------------------------------
        # Download full timeline
        # ----------------------------------------------------------

        csv_bytes = (
            timeline_df
            .to_csv(
                index=False
            )
            .encode(
                "utf-8"
            )
        )


        st.download_button(
            label=(
                "Download complete inference timeline"
            ),
            data=csv_bytes,
            file_name=(
                f"{short_dataset_name.lower()}_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )


        # ----------------------------------------------------------
        # FD004 policy explanation
        # ----------------------------------------------------------

        if (
            short_dataset_name
            == "FD004"
        ):

            with st.expander(
                "FD004 operational policy"
            ):

                st.markdown(
                    """
- **Normal → CONTINUE**
- **Watch → INSPECT**
- **Warning → INSPECT**
- **Critical → STOP**
- A 10-cycle supervised alert creates the Critical state.
- A 20-cycle supervised alert creates the Warning state.
- A 30-cycle alert or anomaly evidence creates the Watch state.
- Anomaly evidence alone cannot create STOP.
- Escalation is immediate.
- De-escalation requires two consecutive lower-state cycles.
- RUL and conformal uncertainty are advisory and do not directly escalate the policy.
"""
                )


    except Exception as error:

        st.exception(
            error
        )


    finally:

        if (
            temporary_path
            and os.path.exists(
                temporary_path
            )
        ):

            os.remove(
                temporary_path
            )
