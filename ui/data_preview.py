import streamlit as st
import polars as pl

def data_preview_and_null_handling():
    """
    Displays a data preview and provides interactive options for handling null values in a Polars DataFrame.
    
    Shows a preview, statistics, and data types of the DataFrame stored in session state. If null values are present, allows the user to choose a null handling strategy (drop rows, fill with mean, median, or mode) and select columns to process. Updates the DataFrame in session state after processing and manages UI feedback accordingly.
    """
    if st.session_state.is_file_uploaded and st.session_state.df is not None:
        st.subheader("2. Data Preview and Preprocessing")
        df: pl.DataFrame = st.session_state.df

        with st.expander("View Data", expanded=True):
            if not df.is_empty():
                st.write("Data Preview:")
                st.dataframe(df.head(10))
                st.write("Data Statistics:")
                st.dataframe(df.describe())
                st.write("Data Types:")
                dtypes_str = "\n".join([f"- **{col}**: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
                st.markdown(dtypes_str)

        null_counts = df.null_count()
        total_nulls = int(null_counts.select(pl.all().sum()).item())

        if not total_nulls.is_empty():
            st.warning(f"The DataFrame contains {total_nulls} null values.")
            st.write("Null counts per column:")
            null_counts_dict = null_counts.row(0, named=True)
            st.dataframe(
                pl.DataFrame(
                    {
                        "Column": null_counts_dict.keys(),
                        "Null Count": null_counts_dict.values()
                    }
                )
            )

            st.subheader("Handle Null Values")
            null_handling_method = st.radio(
                "Choose a method to handle null values:",
                ('Drop Rows with Nulls', 'Fill with Mean', 'Fill with Median', 'Fill with Mode'),
                key="null_handling_method"
            )

            columns_with_nulls = [col for col in df.columns if df[col].null_count() > 0]
            if not columns_with_nulls:
                st.info("No columns with null values found.")
                st.session_state.nulls_handled = True
            else:
                cols_to_process = st.multiselect(
                    "Select columns to apply the method (select none for all applicable columns):",
                    columns_with_nulls,
                    default=columns_with_nulls if null_handling_method == 'Drop Rows with Nulls' else [],
                    key="cols_to_process"
                )
                if not cols_to_process and null_handling_method != 'Drop Rows with Nulls':
                    st.info("Select columns to apply imputation or select 'Drop Rows with Nulls' to process all rows with nulls.")
                    process_nulls_button_disabled = True
                else:
                    process_nulls_button_disabled = False

                if st.button("Apply Null Handling", disabled=process_nulls_button_disabled):
                    processed_df = df.clone()
                    if null_handling_method == 'Drop Rows with Nulls':
                        if cols_to_process:
                            rows_before = processed_df.height
                            processed_df = processed_df.drop_nulls(subset=cols_to_process)
                            rows_after = processed_df.height
                            st.success(f"Dropped {rows_before - rows_after} rows with nulls in selected columns.")
                        else:
                            rows_before = processed_df.height
                            processed_df = processed_df.drop_nulls()
                            rows_after = processed_df.height
                            st.success(f"Dropped {rows_before - rows_after} rows with any null values.")
                    elif null_handling_method in ['Fill with Mean', 'Fill with Median']:
                        numeric_cols = [col for col in cols_to_process if processed_df[col] in [pl.Float64, pl.Int64, pl.UInt64]]
                        non_numeric_cols = [col for col in cols_to_process if processed_df[col] not in [pl.Float64, pl.Int64, pl.UInt64]]
                        if numeric_cols:
                            strategy = 'mean' if null_handling_method == 'Fill with Mean' else 'median'
                            processed_df = processed_df.fill_null(strategy=strategy, subset=numeric_cols)
                            st.success(f"Filled nulls in numeric columns {numeric_cols} with {strategy}.")
                        if non_numeric_cols:
                            st.warning(f"Skipped non-numeric columns for {null_handling_method}: {non_numeric_cols}")
                    elif null_handling_method == 'Fill with Mode':
                        for col in cols_to_process:
                            try:
                                mode_value_series = processed_df[col].mode()
                                if mode_value_series.height > 0:
                                    fill_value = mode_value_series[0].item()
                                    processed_df = processed_df.fill_null(fill_value, subset=[col])
                                    st.success(f"Filled nulls in column '{col}' with mode: {fill_value}")
                                else:
                                    st.warning(f"Could not calculate mode for column '{col}' (possibly no data). Skipping.")
                            except Exception as e:
                                st.warning(f"Error calculating/filling mode for column '{col}': {e}")
                    st.session_state.df = processed_df
                    st.session_state.nulls_handled = True
                    st.rerun()
        else:
            st.success("The DataFrame does not contain null values. Proceed to model training.")
            st.session_state.nulls_handled = True
