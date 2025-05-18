import streamlit as st

def download_model_section():
    """
    Displays a download section in the Streamlit app for retrieving the trained model.
    
    Renders a subheader and a download button if a trained model is available in the session state, allowing users to download the model as a .keras file.
    """
    if st.session_state.model_trained and st.session_state.trained_model_bytes is not None:
        st.subheader("4. Download Trained Model")
        st.download_button(
            label="Download Model (.keras)",
            data=st.session_state.trained_model_bytes,
            file_name="my_trained_model.keras",
            mime="application/octet-stream"
        )
    st.balloons()
    st.markdown(
        """
        ### Note:
        - The model is saved in the Keras format. You can load it using `keras.models.load_model()`.
        """
    )
    st.markdown("""
       ### Thank you for using streamlineML!
       - Do you want to train another model?
       - If yes, please go back to the first step and upload a new dataset."""
    )

    if st.button("Start Over"):
        st.session_state.is_file_uploaded = False
        st.session_state.dfs = {}
        st.session_state.model = None
        st.session_state.nulls_handled = False
        st.session_state.model_trained = False
        st.session_state.trained_model_bytes = None
        st.session_state.current_step = 0
        st.rerun()