import streamlit as st

def download_model_section():
    if st.session_state.model_trained and st.session_state.trained_model_bytes is not None:
        st.subheader("4. Download Trained Model")
        st.download_button(
            label="Download Model (.keras)",
            data=st.session_state.trained_model_bytes,
            file_name="my_trained_model.keras",
            mime="application/octet-stream"
        )
