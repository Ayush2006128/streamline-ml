import streamlit as st

from logic.dataframes.file_opener import open_file

st.set_page_config(page_title="streamlineML", page_icon="favicon.ong", layout="wide")
st.title("Welcome to streamlineML")

if "is_file_uploaded" not in st.session_state:
    st.session_state.is_file_uploaded = False
if "model" not in st.session_state:
    st.session_state.model = None
if "df" not in st.session_state:
    st.session_state.df = None

df_list = []

def upload_files():
    if uploaded_files := st.file_uploader("Upload your files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True):
        for file in uploaded_files:
            file_format = file.name.split(".")[-1]
            if file_format not in ["csv", "parquet", "json", "xlsx"]:
                st.error(f"Unsupported file format: {file_format}")
                continue
            df = open_file(file, file_format)
            df_list.append(df)
            st.session_state.df = df
            st.session_state.is_file_uploaded = True
            st.toast(f"File {file.name} uploaded successfully!", icon=":material/thumb_up:")

def edit_file():
        # TODO: Add more editing options here
        pass  # Placeholder for file editing logic

def train_model():
    # TODO: Add model training logic here
    pass  # Placeholder for model training logic


if st.session_state.is_file_uploaded == False and df_list == []:
    st.subheader("Upload your files")
    upload_files()

if st.session_state.is_file_uploaded == True and df_list != []:
    st.subheader("Edit your file")
    edit_file()

if st.session_state.model == None and st.session_state.df != None:
    st.subheader("Train your model")
    train_model()
