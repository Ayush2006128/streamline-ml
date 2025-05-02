import streamlit as st
from logic.dataframes.file_opener import open_file
from logic.dataframes.edit_dataframe import edit_dataframe

st.title("Upload and Edit DataFrames")
st.info("Upload, View, and Edit DataFrames")
if uploaded_files := st.file_uploader("Upload files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True):
    st.toast("Files uploaded successfully!")
    st.balloons()
    with st.expander("View DataFrames", expanded=False):
        for file in uploaded_files:
            # Get the file format from the file name
            file_format = file.name.split(".")[-1]
                
            # Open the file using the open_file function
            df = open_file(file, file_format)
                
            # Display the DataFrame
            st.subheader(f"Data from {file.name}:")
            st.dataframe(df)

    with st.expander("Edit DataFrame", expanded=False):
        selected_file = st.selectbox("Select a file to edit", [file.name for file in uploaded_files])
        selected_df = open_file(selected_file, selected_file.split(".")[-1])
        
        column_to_edit = st.selectbox("Select a column to edit", selected_df.columns)
        operation = st.selectbox("Select an operation", ["add", "subtract", "multiply", "divide", "fill nan"])
        value = st.number_input("Enter a value for the operation", value=0.0)
        new_column_name = st.text_input("Enter a new column name (leave blank to overwrite)", "")
        
        if st.button("Apply Changes"):
            edited_df = edit_dataframe(selected_df, column_to_edit, operation, value, new_column_name or None)
            st.success("Changes applied successfully!")
            st.dataframe(edited_df)
            st.download_button("Download Edited DataFrame", edited_df.to_csv(), file_name="edited_dataframe.csv", mime="text/csv")