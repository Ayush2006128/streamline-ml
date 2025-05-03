import streamlit as st
import polars as pl
from logic.dataframes.file_opener import open_file
from logic.models import model_builder

st.set_page_config(page_title="streamlineML", page_icon="favicon.png", layout="wide")
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

def edit_file(df: pl.DataFrame):
        with st.expander("View Data", expanded=True):
            if not df.is_empty():
                st.write("Data Preview:")
                st.dataframe(df.head(10))
                st.write("Data Statistics:")
                st.write(df.describe())
                st.write("Data Types:")
                st.write(df.dtypes)
        # Check if the DataFrame has null values
        if df.is_empty():
            st.warning("The DataFrame is empty.")
        else:
            if df.null_count().sum() > 0:
                st.warning("The DataFrame contains null values.")
            else:
                st.success("The DataFrame does not contain null values.")

def train_model(df: pl.DataFrame):
    target_column = st.selectbox("Select target column", df.columns)
    feature_columns = st.multiselect("Select feature columns", df.columns, default=df.columns)
    num_layers = st.number_input("Number of layers", min_value=1, max_value=10, value=5)
    activation = st.selectbox("Activation function", ["relu", "sigmoid", "tanh"])
    task = st.selectbox("Task type", ["classification", "regression"])
    if st.button("Train Model"):
        if target_column not in df.columns:
            st.error(f"Target column {target_column} not found in the DataFrame.")
            return
        if len(feature_columns) == 0:
            st.error("Please select at least one feature column.")
            return
        if task == "classification" and df[target_column].n_unique() > 2:
            st.error("Classification task requires a binary target column.")
            return
        if task == "regression" and df[target_column].n_unique() <= 2:
            st.error("Regression task requires a continuous target column.")
            return
        # Here you would call your model training function
        model = model_builder.ModelBuilder(
            input_shape=df[feature_columns].shape[1],
            num_layers=num_layers,
            num_classes=df[target_column].n_unique() if task == "classification" else None,
            activation=activation,
            task=task
        )
        model = model.build_model()

        model.compile_model(
            model,
            learning_rate=0.001
        )
        epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10)
        batch_size = st.number_input("Batch size", min_value=1, max_value=100, value=32)
        model.fit(
            model,
            df[feature_columns].to_numpy(),
            df[target_column].to_numpy(),
            epochs=epochs,
            batch_size=batch_size
        )
        st.session_state.model = model
        st.success("Model trained successfully!")
        st.balloons()


if st.session_state.is_file_uploaded == False and df_list == []:
    st.subheader("Upload your files")
    upload_files()

if st.session_state.is_file_uploaded == True and df_list != []:
    st.subheader("Edit your file")
    edit_file(df=st.session_state.df)

if st.session_state.model == None and not st.session_state.df.is_empty():
    st.subheader("Train your model")
    train_model(df=st.session_state.df)
