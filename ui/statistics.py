import streamlit as st
import polars as pl
import numpy as np

@st.fragment()
def show_graphs():
    """
    Displays a Streamlit UI section for showing statistics and graphs of selected Dataframes.
    Allows users to select a file, columns and charts to visualize.
    """
    if st.session_state.is_file_uploaded and st.session_state.dfs and st.session_state.nulls_handled:
        st.subheader("3. Show statistics and graphs")
        file_names = list(st.session_state.dfs.keys())
        selected_file = st.selectbox("Select a file", file_names, key="stats_file")
        df: pl.DataFrame = st.session_state.dfs[selected_file]

        if df.is_empty():
            st.error("Cannot show statistics on an empty DataFrame after preprocessing.")
            return

        # Select columns for statistics
        columns = df.columns
        selected_columns = st.multiselect("Select columns for statistics", columns, default=columns, key="stats_columns")

        # Display statistics
        if selected_columns:
            st.write("Statistics:")
            stats_df = df.select(selected_columns).describe()
            st.dataframe(stats_df.to_pandas())

        # Select chart type
        chart_type = st.selectbox("Select chart type", ["Histogram", "Scatter Plot"], key="chart_type")

        # Select column for chart
        selected_chart_column = st.selectbox("Select column for chart", columns, key="chart_column")

        # Display chart
        if selected_chart_column:
            if chart_type == "Histogram":
                hist_data = df.select(selected_chart_column).to_numpy().flatten()
                st.bar_chart(hist_data)
            elif chart_type == "Scatter Plot":
                scatter_data = df.select(selected_chart_column).to_numpy().flatten()
                st.scatter_chart(scatter_data)