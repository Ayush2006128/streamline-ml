import streamlit as st
from logic.dataframes.file_opener import open_file

def file_upload_section():
    st.subheader("1. Upload your files")
    uploaded_files = st.file_uploader("Upload your files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True)

    if uploaded_files and st.session_state.df is None:
        file = uploaded_files[0]
        file_format = file.name.split(".")[-1]
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
