import streamlit as st

st.set_page_config(page_title="streamlineML", page_icon="📊", layout="wide")
st.title("Welcome to streamlineML")

file_upload_page = st.Page("screens/file_upload.py", title="Upload files", icon=":material/upload_file:")

if st.button("Get Started"):
    st.switch_page(file_upload_page)