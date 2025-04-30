import streamlit as st
from logic.dataframes.file_opener import open_file

st.title("Upload files")
st.info("Upload and View DataFrames")
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