import streamlit as st
import io
from logic.dataframes.file_opener import open_file

@st.fragment()
def file_upload_section():
    """
    Displays a file upload interface in Streamlit and processes uploaded files.
    
    Allows users to upload multiple files in CSV, Parquet, JSON, or XLSX formats. Processes each file by loading it into a DataFrame and stores them in session state as a dictionary with filenames as keys. Displays success or error messages based on the outcome.
    """
    st.subheader("1. Upload your files")
    uploaded_files = st.file_uploader("Upload your files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True)

    if uploaded_files:
        st.session_state.dfs = {}
        for f in uploaded_files:
            if f is not None:
                file = io.BytesIO(f.read())
                file_format = f.name.split(".")[-1]
                try:
                    st.session_state.dfs[f.name] = open_file(file, file_format)
                    st.toast(f"File {f.name} uploaded successfully!", icon=":material/thumb_up:")
                except Exception as e:
                    st.error(f"Error opening file {f.name}: {e}")
        st.session_state.is_file_uploaded = bool(st.session_state.dfs)
        st.session_state.nulls_handled = False
        st.session_state.model_trained = False
        st.session_state.model = None
        st.session_state.trained_model_bytes = None
