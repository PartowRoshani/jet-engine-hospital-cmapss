from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from app import (
    create_trajectory_figure,
    read_engine_file,
    run_fd001_pipeline,
    run_fd003_pipeline,
)


st.set_page_config(
    page_title="Jet Engine Hospital",
    page_icon="✈️",
    layout="wide",
)

st.title("✈️ Jet Engine Hospital")
st.caption(
    "Unified predictive-maintenance application for NASA C-MAPSS FD001 and FD003."
)

st.markdown(
    """
Upload a complete chronological engine trajectory and receive:

- Remaining Useful Life prediction;
- 95% conformal uncertainty interval;
- 10-, 20-, and 30-cycle failure probabilities;
- anomaly evidence;
- a `CONTINUE`, `INSPECT`, or `STOP` recommendation.

The uploaded history for every engine must begin at cycle 1.

> Educational demonstration only. Not an aviation-certified maintenance system.
"""
)

dataset_name = st.radio(
    "Dataset / project stage",
    options=[
        "FD001 — Foundation",
        "FD003 — Multi-fault",
    ],
    horizontal=True,
)

uploaded_file = st.file_uploader(
    "Engine trajectory file",
    type=["csv", "txt"],
)

run_button = st.button(
    "Run inference",
    type="primary",
    use_container_width=True,
)

if run_button:
    if uploaded_file is None:
        st.error("Upload a CSV or C-MAPSS text file first.")
        st.stop()

    suffix = Path(uploaded_file.name).suffix or ".csv"
    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix,
        ) as temporary_file:
            temporary_file.write(uploaded_file.getbuffer())
            temporary_path = temporary_file.name

        with st.spinner("Running inference..."):
            raw_frame = read_engine_file(temporary_path)

            if dataset_name.startswith("FD001"):
                timeline_df, dashboard_df, plot_frame = (
                    run_fd001_pipeline(raw_frame)
                )
                short_dataset_name = "FD001"
            else:
                timeline_df, dashboard_df, plot_frame = (
                    run_fd003_pipeline(raw_frame)
                )
                short_dataset_name = "FD003"

        numeric_columns = dashboard_df.select_dtypes(
            include=[np.number]
        ).columns

        dashboard_df = dashboard_df.copy()
        dashboard_df[numeric_columns] = (
            dashboard_df[numeric_columns].round(4)
        )

        action_counts = (
            dashboard_df["Action"]
            .value_counts()
            .to_dict()
        )

        st.success("Inference completed successfully.")

        metric_columns = st.columns(5)
        metric_columns[0].metric(
            "Dataset",
            short_dataset_name,
        )
        metric_columns[1].metric(
            "Engines",
            int(dashboard_df["engine_id"].nunique()),
        )
        metric_columns[2].metric(
            "CONTINUE",
            int(action_counts.get("CONTINUE", 0)),
        )
        metric_columns[3].metric(
            "INSPECT",
            int(action_counts.get("INSPECT", 0)),
        )
        metric_columns[4].metric(
            "STOP",
            int(action_counts.get("STOP", 0)),
        )

        st.subheader("Latest maintenance status per engine")
        st.dataframe(
            dashboard_df,
            use_container_width=True,
            hide_index=True,
        )

        st.subheader("RUL trajectory")
        trajectory_figure = create_trajectory_figure(
            plot_frame,
            dataset_name=short_dataset_name,
        )
        st.pyplot(
            trajectory_figure,
            use_container_width=True,
        )
        plt.close(trajectory_figure)

        csv_bytes = timeline_df.to_csv(
            index=False,
        ).encode("utf-8")

        st.download_button(
            label="Download complete inference timeline",
            data=csv_bytes,
            file_name=(
                f"{short_dataset_name.lower()}_predictions.csv"
            ),
            mime="text/csv",
            use_container_width=True,
        )

    except Exception as error:
        st.exception(error)

    finally:
        if temporary_path and os.path.exists(temporary_path):
            os.remove(temporary_path)
