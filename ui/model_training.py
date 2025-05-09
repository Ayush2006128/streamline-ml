import streamlit as st
import numpy as np
import io
import tempfile
import os
import polars as pl
from logic.models import model_builder
from ui.progress_callback import StreamlitProgressCallback

def model_training_section():
    """
    Displays a Streamlit UI section for configuring and training a Dense NN model on uploaded data.
    
    This function guides the user through selecting the target and feature columns, specifying the task type (classification or regression), and configuring both Dense NN architecture and training parameters. It validates the selected columns and data compatibility, prepares the data for model input, builds and compiles the model using user-specified settings, and trains the model with real-time progress feedback. Upon successful training, the trained model is saved and made available for download. Errors during data preparation, model building, or training are reported to the user, and application state is updated accordingly.
    """
    if st.session_state.is_file_uploaded and st.session_state.dfs is not None and st.session_state.nulls_handled:
        st.subheader("3. Train your model")
        df: pl.DataFrame = st.session_state.dfs

        if df.is_empty():
            st.error("Cannot train on an empty DataFrame after preprocessing.")
            return

        target_column = st.selectbox("Select target column", df.columns, key="target_column")
        feature_columns = st.multiselect(
            "Select feature columns",
            df.columns,
            default=[col for col in df.columns if col != target_column],
            key="feature_columns"
        )
        task = st.selectbox("Task type", ["classification", "regression"], key="task_type")

        st.write("Model Parameters:")
        num_layers = st.number_input("Number of Hidden Layers", min_value=1, max_value=10, value=2, key="num_layers_model")
        activation = st.selectbox("Activation function", ["relu", "sigmoid", "tanh"], key="activation_model")
        dense_units = st.number_input("Units per Hidden Layer", min_value=1, value=64, key="dense_units")
        dropout_rate = st.slider("Dropout Rate", min_value=0.0, max_value=0.8, value=0.0, step=0.05, key="dropout_rate")
        add_batch_norm = st.checkbox("Add Batch Normalization", value=False, key="add_batch_norm")

        st.write("Training Parameters:")
        epochs = st.number_input("Number of epochs", min_value=1, max_value=100, value=10, key="epochs_train")
        batch_size = st.number_input("Batch size", min_value=1, max_value=1000, value=32, key="batch_size_train")
        learning_rate = st.number_input("Learning Rate", min_value=1e-6, max_value=0.1, value=0.001, format="%.6f", key="lr_train")
        optimizer = st.selectbox("Optimizer", ["adam", "sgd", "rmsprop"], key="optimizer_train")
        validation_split = st.slider("Validation Split", min_value=0.0, max_value=0.5, value=0.2, step=0.05, key="val_split")

        if st.button("Train Model", key="train_button"):
            if target_column not in df.columns:
                st.error(f"Target column '{target_column}' not found in the DataFrame.")
                st.session_state.model_trained = False
                return

            if not feature_columns:
                st.error("Please select at least one feature column.")
                st.session_state.model_trained = False
                return

            target_series = df.get_column(target_column)
            if task == "classification":
                n_unique = target_series.n_unique()
                if n_unique <= 1:
                    st.error("Classification task requires at least two unique values in the target column.")
                    st.session_state.model_trained = False
                    return
                num_classes_model = n_unique if n_unique > 2 else 1
                if n_unique > 2 and num_classes_model == 1:
                    if n_unique > 2: num_classes_model = n_unique
            elif task == "regression":
                num_classes_model = None
                if not target_series.dtype in [pl.Float32, pl.Float64, pl.Int32, pl.Int64]:
                    st.error("Regression task requires a numeric target column.")
                    st.session_state.model_trained = False
                    return
                if target_series.n_unique() <= 2:
                    st.warning("Regression task typically requires more than two unique values in the target column.")

            try:
                X = df.select(feature_columns).to_numpy().astype(np.float32)
                y = df.get_column(target_column).to_numpy()
                if task == 'regression' and y.ndim == 1:
                    y = y.reshape(-1, 1)
                if task == 'classification':
                    if not np.issubdtype(y.dtype, np.integer):
                        st.warning(f"Target column '{target_column}' data type ({y.dtype}) is not integer. Attempting conversion.")
                        try:
                            y = y.astype(np.int32)
                        except ValueError:
                            st.error(f"Could not convert target column '{target_column}' to integers for classification.")
                            st.session_state.model_trained = False
                            return
                    if num_classes_model == 1:
                        y = y.astype(np.float32)
                # For dense models, input shape is just (num_features,)
                model_input_shape = (X.shape[1],)
                st.info(f"Input data shape for training: {X.shape}")
                st.info(f"Target data shape for training: {y.shape}")
            except Exception as e:
                st.error(f"Error preparing data for training: {e}")
                st.session_state.model_trained = False
                return

            try:
                st.info("Building model...")
                builder = model_builder.ModelBuilder(
                    input_shape=model_input_shape,
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
                )
                st.write("Model Summary:")
                summary_str = io.StringIO()
                compiled_model.summary(print_fn=lambda x: summary_str.write(x + '\n'))
                with st.expander("View Model Summary"):
                    st.text(summary_str.getvalue())
            except Exception as e:
                st.error(f"Error building or compiling model: {e}")
                st.session_state.model_trained = False
                return

            st.info("Starting training...")
            progress_bar = st.progress(0, text="Training in progress...")
            try:
                history = compiled_model.fit(
                    X,
                    y,
                    epochs=epochs,
                    batch_size=batch_size,
                    validation_split=validation_split,
                    callbacks=[StreamlitProgressCallback(progress_bar, epochs)],
                    verbose=0
                )
                progress_bar.progress(1.0, text="Training complete!")
                st.session_state.model = compiled_model
                st.session_state.model_trained = True
                st.success("Model trained successfully!")
                st.balloons()
                st.info("Preparing model for download...")
                temp_dir = tempfile.mkdtemp()
                save_path = os.path.join(temp_dir, "my_streamlit_model.keras")
                try:
                    compiled_model.save(save_path, save_format='keras')
                    with open(save_path, 'rb') as f:
                        st.session_state.trained_model_bytes = f.read()
                    st.success("Model ready for download.")
                except Exception as e:
                    st.error(f"Error saving model for download: {e}")
                    st.session_state.trained_model_bytes = None
                finally:
                    if os.path.exists(save_path):
                        os.remove(save_path)
                    if os.path.exists(temp_dir):
                        os.rmdir(temp_dir)
            except Exception as e:
                st.error(f"Error during model training: {e}")
                progress_bar.progress(0, text="Training failed.")
                st.session_state.model_trained = False
