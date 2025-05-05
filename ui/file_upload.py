import streamlit as st
import io
from logic.dataframes.file_opener import open_file

def file_upload_section():
    """
    Displays a file upload interface in Streamlit and processes the first uploaded file.
    
    Allows users to upload multiple files in CSV, Parquet, JSON, or XLSX formats. If files are uploaded and no DataFrame is currently loaded in the session state, processes the first file by loading it into a DataFrame and updating relevant session state variables. Displays success or error messages based on the outcome.
    """
    st.subheader("1. Upload your files")
    uploaded_files = st.file_uploader("Upload your files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True)

    if uploaded_files and st.session_state.df is None:
        for f in uploaded_files:
            if f is not None:
                file = io.BytesIO(f.read())
                file_format = f.name.split(".")[-1]
                break
        try:
            st.session_state.df = open_file(file, file_format)
            st.session_state.is_file_uploaded = True
            st.session_state.nulls_handled = False
            st.session_state.model_trained = False
            st.session_state.model = None
            st.session_state.trained_model_bytes = None
            st.toast(f"File {file.name} uploaded successfully!", icon=":material/thumb_up:")
        except Exception as e:
            st.error(f"Error opening file {file.name}: {e}")
            st.session_state.df = None
            st.session_state.is_file_uploaded = False
