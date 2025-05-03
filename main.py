import streamlit as st
import polars as pl
import numpy as np # For dummy data and array conversion
import tensorflow as tf
import io
import tempfile # To save the model temporarily
import os # To remove the temporary file
from logic.dataframes import open_file
from logic.models import model_builder

# --- Custom Keras Callback for Streamlit Progress ---
class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """A Keras callback to update a Streamlit progress bar."""
    def __init__(self, progress_bar, num_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        # Update text at the beginning of each epoch
        self.progress_bar.progress(
            epoch / self.num_epochs,
            text=f"Epoch {epoch + 1}/{self.num_epochs}"
        )

    def on_epoch_end(self, epoch, logs=None):
        # Ensure progress is updated at the end of the epoch, especially for the last one
        # Get relevant metrics, handling potential None if they aren't computed yet
        loss_val = logs.get('loss')
        val_loss_val = logs.get('val_loss')
        text = f"Epoch {epoch + 1}/{self.num_epochs}"
        if loss_val is not None:
             text += f" - Loss: {loss_val:.4f}"
        if val_loss_val is not None:
             text += f" - Val Loss: {val_loss_val:.4f}"

        self.progress_bar.progress(
            (epoch + 1) / self.num_epochs,
            text=text
        )


# --- Streamlit App ---
st.set_page_config(page_title="streamlineML", page_icon="favicon.png", layout="wide")
st.title("Welcome to streamlineML")

# --- Session State Initialization ---
if "is_file_uploaded" not in st.session_state:
    st.session_state.is_file_uploaded = False
if "df" not in st.session_state:
    st.session_state.df = None
if "model" not in st.session_state:
    st.session_state.model = None
if "nulls_handled" not in st.session_state:
    st.session_state.nulls_handled = False
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False
if "trained_model_bytes" not in st.session_state:
    st.session_state.trained_model_bytes = None


# --- File Upload ---
st.subheader("1. Upload your files")
uploaded_files = st.file_uploader("Upload your files", type=["csv", "parquet", "json", "xlsx"], accept_multiple_files=True)

if uploaded_files and st.session_state.df is None:
    # Process only the first file for simplicity in this example,
    # or loop if you want to combine/process multiple files.
    file = uploaded_files[0]
    file_format = file.name.split(".")[-1]
    try:
        st.session_state.df = open_file(file, file_format)
        st.session_state.is_file_uploaded = True
        st.session_state.nulls_handled = False # Reset null handling status on new upload
        st.session_state.model_trained = False # Reset training status on new upload
        st.session_state.model = None # Reset model
        st.session_state.trained_model_bytes = None # Reset downloadable model
        st.toast(f"File {file.name} uploaded successfully!", icon=":material/thumb_up:")
    except Exception as e:
        st.error(f"Error opening file {file.name}: {e}")
        st.session_state.df = None # Ensure df is None if opening failed
        st.session_state.is_file_uploaded = False


# --- Data Preview and Null Handling ---
if st.session_state.is_file_uploaded and st.session_state.df is not None:
    st.subheader("2. Data Preview and Preprocessing")
    df = st.session_state.df

    with st.expander("View Data", expanded=True):
        if not df.is_empty():
            st.write("Data Preview:")
            # Streamlit can display Polars DataFrames directly
            st.dataframe(df.head(10))
            st.write("Data Statistics:")
            st.dataframe(df.describe())
            st.write("Data Types:")
            # Display dtypes nicely
            dtypes_str = "\n".join([f"- **{col}**: {dtype}" for col, dtype in zip(df.columns, df.dtypes)])
            st.markdown(dtypes_str)


    # Null Value Handling
    null_counts = df.null_count()
    total_nulls = null_counts.sum().item() # Use .item() to get the scalar value

    if total_nulls > 0:
        st.warning(f"The DataFrame contains {total_nulls} null values.")
        st.write("Null counts per column:")
        # Display null counts without converting to pandas
        null_counts_dict = null_counts.row(0, named=True) # Get the first row as a dictionary
        st.dataframe(list(null_counts_dict.items())) # Display as a list of tuples (Column, Null Count)


        st.subheader("Handle Null Values")
        null_handling_method = st.radio(
            "Choose a method to handle null values:",
            ('Drop Rows with Nulls', 'Fill with Mean', 'Fill with Median', 'Fill with Mode'),
            key="null_handling_method"
        )

        columns_with_nulls = [col for col in df.columns if df[col].null_count().item() > 0]
        if not columns_with_nulls: # Should not happen if total_nulls > 0, but for safety
             st.info("No columns with null values found.")
             st.session_state.nulls_handled = True # Mark as handled if no nulls
        else:
            cols_to_process = st.multiselect(
                "Select columns to apply the method (select none for all applicable columns):",
                columns_with_nulls,
                default=columns_with_nulls if null_handling_method == 'Drop Rows with Nulls' else [], # Default select all for dropping
                key="cols_to_process"
            )
            if not cols_to_process and null_handling_method != 'Drop Rows with Nulls':
                 st.info("Select columns to apply imputation or select 'Drop Rows with Nulls' to process all rows with nulls.")
                 process_nulls_button_disabled = True
            else:
                 process_nulls_button_disabled = False

            if st.button("Apply Null Handling", disabled=process_nulls_button_disabled):
                processed_df = df.clone() # Create a clone to avoid modifying the original state directly

                if null_handling_method == 'Drop Rows with Nulls':
                    if cols_to_process:
                        # Drop rows where *any* of the selected columns have nulls
                        rows_before = processed_df.height
                        processed_df = processed_df.drop_nulls(subset=cols_to_process)
                        rows_after = processed_df.height
                        st.success(f"Dropped {rows_before - rows_after} rows with nulls in selected columns.")
                    else:
                        # Drop rows where *any* column has a null (applies to all columns implicitly)
                        rows_before = processed_df.height
                        processed_df = processed_df.drop_nulls()
                        rows_after = processed_df.height
                        st.success(f"Dropped {rows_before - rows_after} rows with any null values.")

                elif null_handling_method in ['Fill with Mean', 'Fill with Median']:
                     numeric_cols = [col for col in cols_to_process if processed_df[col].is_numeric()]
                     non_numeric_cols = [col for col in cols_to_process if not processed_df[col].is_numeric()]

                     if numeric_cols:
                         strategy = 'mean' if null_handling_method == 'Fill with Mean' else 'median'
                         processed_df = processed_df.fill_null(strategy=strategy, subset=numeric_cols)
                         st.success(f"Filled nulls in numeric columns {numeric_cols} with {strategy}.")
                     if non_numeric_cols:
                          st.warning(f"Skipped non-numeric columns for {null_handling_method}: {non_numeric_cols}")

                elif null_handling_method == 'Fill with Mode':
                    for col in cols_to_process:
                        try:
                            # Calculate mode using Polars
                            mode_value_series = processed_df[col].mode()
                            if mode_value_series.height > 0:
                                # Use the first mode if multiple exist
                                fill_value = mode_value_series[0].item() # Get scalar item from the series
                                processed_df = processed_df.fill_null(fill_value, subset=[col])
                                st.success(f"Filled nulls in column '{col}' with mode: {fill_value}")
                            else:
                                st.warning(f"Could not calculate mode for column '{col}' (possibly no data). Skipping.")
                        except Exception as e:
                            st.warning(f"Error calculating/filling mode for column '{col}': {e}")


                st.session_state.df = processed_df # Update the DataFrame in session state
                st.session_state.nulls_handled = True
                st.rerun() # Rerun to show the updated DataFrame and null counts
    else:
         st.success("The DataFrame does not contain null values. Proceed to model training.")
         st.session_state.nulls_handled = True # Mark as handled if no nulls initially


# --- Model Training ---
# Only show training if file uploaded and nulls are handled
if st.session_state.is_file_uploaded and st.session_state.df is not None and st.session_state.nulls_handled:
    st.subheader("3. Train your model")
    df = st.session_state.df # Get the potentially processed dataframe

    if df.is_empty():
         st.error("Cannot train on an empty DataFrame after preprocessing.")
    else:
        target_column = st.selectbox("Select target column", df.columns, key="target_column")
        # Exclude target column from feature selection default
        feature_columns = st.multiselect(
            "Select feature columns",
            df.columns,
            default=[col for col in df.columns if col != target_column],
            key="feature_columns"
        )

        task = st.selectbox("Task type", ["classification", "regression"], key="task_type")

        # Parameters for the CNN ModelBuilder (assuming it's a CNN builder)
        st.write("CNN Model Parameters:")
        # Prompt for input shape tuple for the CNN builder
        st.info("Assuming image-like data for CNN. Please provide the shape per sample (height, width, channels).")
        col1, col2, col3 = st.columns(3)
        with col1:
            input_h = st.number_input("Input Height", min_value=1, value=32, key="input_h")
        with col2:
            input_w = st.number_input("Input Width", min_value=1, value=32, key="input_w")
        with col3:
            input_c = st.number_input("Input Channels", min_value=1, value=3, key="input_c")
        model_input_shape = (input_h, input_w, input_c)


        num_layers = st.number_input("Number of Conv/Pool blocks", min_value=1, max_value=10, value=3, key="num_layers_model")
        activation = st.selectbox("Activation function", ["relu", "sigmoid", "tanh"], key="activation_model")
        dense_units = st.number_input("Dense layer units", min_value=1, value=64, key="dense_units")
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.0, step=0.05, key="dropout_rate")
        add_batch_norm = st.checkbox("Add Batch Normalization", value=False, key="add_batch_norm")


        st.write("Training Parameters:")
        epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10, key="epochs_train")
        batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=32, key="batch_size_train")
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=0.1, value=0.001, format="%.6f", key="lr_train")
        optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], key="optimizer_train") # Simplified optimizer choice

        # Add option for validation split or data
        validation_split = st.slider("Validation Split", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="val_split")


        if st.button("Train Model", key="train_button"):
            if target_column not in df.columns:
                st.error(f"Target column '{target_column}' not found in the DataFrame.")
                st.session_state.model_trained = False
                return
            if not feature_columns: # Check if list is empty
                st.error("Please select at least one feature column.")
                st.session_state.model_trained = False
                return

            # Basic checks based on task
            target_series = df.get_column(target_column) # Use get_column
            if task == "classification":
                n_unique = target_series.n_unique()
                if n_unique <= 1:
                     st.error("Classification task requires at least two unique values in the target column.")
                     st.session_state.model_trained = False
                     return
                # Assuming binary classification for num_classes=1 or multi-class otherwise
                num_classes_model = n_unique if n_unique > 2 else 1
                if n_unique > 2 and num_classes_model == 1:
                     st.warning(f"Target column has {n_unique} unique values, but num_classes is set to 1. Assuming binary classification or issue with unique count.")
                     # Adjust num_classes_model if the unique count implies multi-class
                     if n_unique > 2: num_classes_model = n_unique


            elif task == "regression":
                num_classes_model = None # num_classes not needed for regression
                # Optional: Check if target column is numeric for regression
                if not target_series.is_numeric():
                    st.error("Regression task requires a numeric target column.")
                    st.session_state.model_trained = False
                    return
                if target_series.n_unique() <= 2:
                     st.warning("Regression task typically requires more than two unique values in the target column.")


            # --- Prepare Data for Keras ---
            # Convert Polars DataFrame to NumPy arrays
            # Need to select only feature columns and target column
            try:
                # Use .select().to_numpy()
                X = df.select(feature_columns).to_numpy().astype(np.float32)
                y = df.get_column(target_column).to_numpy() # Get target series and convert

                # Reshape y for regression if it's a single column vector [n,] to [n, 1]
                if task == 'regression' and y.ndim == 1:
                     y = y.reshape(-1, 1)

                # For classification with sparse_categorical_crossentropy, target should be integers
                if task == 'classification':
                    # Ensure target is integer type
                    if not np.issubdtype(y.dtype, np.integer):
                         st.warning(f"Target column '{target_column}' data type ({y.dtype}) is not integer. Attempting conversion.")
                         try:
                              # Attempt conversion, might fail if values are not convertible
                             y = y.astype(np.int32)
                         except ValueError:
                             st.error(f"Could not convert target column '{target_column}' to integers for classification.")
                             st.session_state.model_trained = False
                             return
                    # For binary classification (sigmoid output), y might need to be float32
                    # Binary crossentropy typically expects float targets (0.0 or 1.0)
                    if num_classes_model == 1:
                         y = y.astype(np.float32)


                # --- Reshape input data if necessary for CNN ---
                # The ModelBuilder expects input_shape=(height, width, channels)
                # X currently has shape (num_samples, num_features)
                # We need to reshape X to (num_samples, height, width, channels)
                expected_elements = model_input_shape[0] * model_input_shape[1] * model_input_shape[2]
                if X.shape[1] != expected_elements:
                     st.error(f"Number of features ({X.shape[1]}) does not match the total elements in the specified input shape {model_input_shape} ({expected_elements}). Please ensure your data can be reshaped to this shape.")
                     st.session_state.model_trained = False
                     return

                try:
                     X_reshaped = X.reshape(-1, *model_input_shape)
                except ValueError as e:
                     st.error(f"Could not reshape data to {model_input_shape}. Check your feature selection and input shape parameters. Error: {e}")
                     st.session_state.model_trained = False
                     return

                st.info(f"Input data shape for training: {X_reshaped.shape}")
                st.info(f"Target data shape for training: {y.shape}")

            except Exception as e:
                 st.error(f"Error preparing data for training: {e}")
                 st.session_state.model_trained = False
                 return

            # --- Build and Compile Model ---
            try:
                st.info("Building model...")
                builder = model_builder.ModelBuilder(
                    input_shape=model_input_shape, # Use the specified tuple input shape
                    num_layers=num_layers,
                    num_classes=num_classes_model,
                    activation=activation,
                    task=task,
                    dense_units=dense_units,
                    dropout_rate=dropout_rate,
                    add_batch_norm=add_batch_norm
                )
                model = builder.build_model()

                st.info("Compiling model...")
                compiled_model = builder.compile_model(
                    model,
                    learning_rate=learning_rate,
                    optimizer=optimizer
                    # loss and metrics will be defaulted by the builder based on task
                )

                st.write("Model Summary:")
                # Redirect summary to a Streamlit expandable
                summary_str = io.StringIO()
                compiled_model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
                with st.expander("View Model Summary"):
                    st.text(summary_str.getvalue())


            except Exception as e:
                st.error(f"Error building or compiling model: {e}")
                st.session_state.model_trained = False
                return

            # --- Train Model ---
            st.info("Starting training...")
            # Create progress bar
            progress_bar = st.progress(0, text="Training in progress...")

            try:
                # Use validation_split
                history = compiled_model.fit(
                    X_reshaped,
                    y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split, # Use validation split
                    callbacks=[StreamlitProgressCallback(progress_bar, epochs)], # Add the custom callback
                    verbose=0 # Suppress default Keras output
                )
                progress_bar.progress(1.0, text="Training complete!") # Ensure it reaches 100%

                st.session_state.model = compiled_model
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
                st.balloons()

                # --- Save model to bytes for download ---
                st.info("Preparing model for download...")
                # Use a temporary file path to save the model
                temp_dir = tempfile.mkdtemp()
                save_path = os.path.join(temp_dir, "my_streamlit_model.keras")
                try:
                    compiled_model.save(save_path, save_format='keras')

                    # Read the saved file into bytes
                    with open(save_path, 'rb') as f:
                        st.session_state.trained_model_bytes = f.read()

                    st.success("Model ready for download.")
                except Exception as e:
                     st.error(f"Error saving model for download: {e}")
                     st.session_state.trained_model_bytes = None # Ensure no bytes are saved on error
                finally:
                    # Clean up the temporary file and directory
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)


            except Exception as e:
                st.error(f"Error during model training: {e}")
                progress_bar.progress(0, text="Training failed.")
                st.session_state.model_trained = False


# --- Download Model ---
if st.session_state.model_trained and st.session_state.trained_model_bytes is not None:
    st.subheader("4. Download Trained Model")
    st.download_button(
        label="Download Model (.keras)",
        data=st.session_state.trained_model_bytes,
        file_name="my_trained_model.keras",
        mime="application/octet-stream" # Generic mime type for binary data
    )

# --- Display training history (Optional) ---
# You could add a section here to display history.history plots or dataframes
# after training is complete, using the history object returned by model.fit.
# history is not stored in session_state in this version, you'd need to add that
# if you want to display it after a rerun.