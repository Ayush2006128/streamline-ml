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
if "df" not in st.session_state:
    st.session_state.dfs = []
if "model" not in st.session_state:
    st.session_state.model = None
if "nulls_handled" not in st.session_state:
    st.session_state.nulls_handled = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "trained_model_bytes" not in st.session_state:
    st.session_state.trained_model_bytes = None

# --- File Upload ---
file_upload_section()

# --- Data Preview and Null Handling ---
data_preview_and_null_handling()

# --- Show Statistics and Graphs ---
show_graphs()

# --- Model Training ---
model_training_section()

# --- Download Model ---
download_model_section()