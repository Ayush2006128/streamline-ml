import streamlit as st
from ui.file_upload import file_upload_section
from ui.data_preview import data_preview_and_null_handling
from ui.model_training import model_training_section
from ui.download import download_model_section
from ui.statistics import show_graphs

st.set_page_config(page_title="streamlineML", page_icon="favicon.png", layout="wide")
st.title("Welcome to streamlineML")

# --- Session State Initialization ---
if "is_file_uploaded" not in st.session_state:
    st.session_state.is_file_uploaded = False
if "dfs" not in st.session_state:
    st.session_state.dfs = {}
if "model" not in st.session_state:
    st.session_state.model = None
if "nulls_handled" not in st.session_state:
    st.session_state.nulls_handled = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "trained_model_bytes" not in st.session_state:
    st.session_state.trained_model_bytes = None
if "current_step" not in st.session_state:
    st.session_state.current_step = 0

# --- File Upload ---
if st.session_state.current_step == 0:
    file_upload_section()
    if st.session_state.is_file_uploaded:
        if st.button("Next", key="next_to_preview"):
            st.session_state.current_step = 1
            st.rerun()
    st.stop()

# --- Data Preview and Null Handling ---
if st.session_state.current_step == 1:
    data_preview_and_null_handling()
    if st.session_state.nulls_handled:
        if st.button("Next", key="next_to_stats"):
            st.session_state.current_step = 2
            st.rerun()
    st.stop()

# --- Show Statistics and Graphs ---
if st.session_state.current_step == 2:
    show_graphs()
    if st.button("Next", key="next_to_train"):
        st.session_state.current_step = 3
        st.rerun()
    st.stop()

# --- Model Training ---
if st.session_state.current_step == 3:
    model_training_section()
    if st.session_state.model_trained:
        if st.button("Next", key="next_to_download"):
            st.session_state.current_step = 4
            st.rerun()
    st.stop()

# --- Download Model ---
if st.session_state.current_step == 4:
    download_model_section()