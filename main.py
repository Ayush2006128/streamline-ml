import streamlit as st

st.set_page_config(page_title="streamlineML", page_icon="favicon.ong", layout="wide")
st.title("Welcome to streamlineML")

file_upload_page = st.Page("screens/file_upload.py", title="Upload files", icon=":material/upload_file:")

model_builder_page = st.Page("screens/model_builder_ui.py", title="Model Builder", icon=":material:robot:")

router = st.navigation([file_upload_page, model_builder_page], position="hidden")

if st.button("Go to File Upload", help="Upload files to edit and view them"):
    router.url_path = file_upload_page.url_path

if st.button("Go to Model Builder", help="Build and train models"):
    router.url_path = model_builder_page.url_path

if __name__ == "__main__":
    router.run()
