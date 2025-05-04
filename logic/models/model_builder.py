import tensorflow as tf
from typing import Optional, Literal, Tuple, List, Union
import numpy as np

class ModelBuilder:
    """
    A class to build, compile, and optionally train a TensorFlow Keras Convolutional Neural Network model.
    Supports both classification and regression tasks.
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_layers: int = 3, # Reduced default layers for a simpler start
        num_classes: Optional[int] = None,
        activation: str = 'relu',
        task: Literal['classification', 'regression'] = 'classification',
        filters: Union[int, List[int]] = 32, # Allow specifying filter counts per layer
        kernel_size: Union[int, Tuple[int, ...], List[Union[int, Tuple[int, ...]]]] = 3, # Allow specifying kernel sizes
        dense_units: int = 64, # Parameterize Dense layer units
        dropout_rate: Optional[float] = None, # Add optional dropout
        add_batch_norm: bool = False # Add optional batch normalization
    ):
        """
        Args:
            input_shape: The shape of the input data, excluding the batch size.
                         For Conv2D, this should be a tuple like (height, width, channels).
            num_layers: The number of convolutional blocks (Conv2D + MaxPooling2D).
            num_classes: The number of classes for the classification task. Required if task is 'classification'.
            activation: The activation function to use for hidden layers.
            task: The type of task, either 'classification' or 'regression'.
            filters: An integer or a list of integers specifying the number of filters
                     for each convolutional layer. If an integer, the number of filters
                     will increase with each layer. If a list, its length must match num_layers.
            kernel_size: An integer, a tuple of integers, or a list specifying the
                         kernel size for each convolutional layer. If an integer or tuple,
                         it will be used for all layers. If a list, its length must match num_layers.
            dense_units: The number of units in the dense layer before the output layer.
            dropout_rate: The dropout rate to apply after pooling and the dense layer.
                         Set to None to disable dropout.
            add_batch_norm: Whether to add Batch Normalization layers after each convolution.
        """
        if task == 'classification' and num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) for d in input_shape):
             raise ValueError("input_shape must be a tuple of integers.")
        if len(input_shape) != 3 and any(isinstance(layer, tf.keras.layers.Conv2D) for layer in self._build_conv_layers(input_shape, num_layers, filters, kernel_size, activation, add_batch_norm)):
             # Basic check: if building Conv2D, input shape should be 3D (H, W, C)
             # This is a simplification, as other layers might have different requirements.
             # A more robust check might involve building the model and catching errors.
             pass # Defer detailed shape validation to Keras build

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation
        self.task = task
        self.filters = filters
        self.kernel_size = kernel_size
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.add_batch_norm = add_batch_norm

        # Determine filter and kernel size lists based on input
        self._filter_list = self._get_param_list(filters, num_layers, is_filter=True)
        self._kernel_list = self._get_param_list(kernel_size, num_layers, is_filter=False)

        if len(self._filter_list) != num_layers:
             raise ValueError("Length of filters list must match num_layers")
        if len(self._kernel_list) != num_layers:
             raise ValueError("Length of kernel_size list must match num_layers")


    def _get_param_list(self, param, num_layers, is_filter=False):
        """Helper to convert single param to a list for each layer."""
        if isinstance(param, list):
            return param
        elif isinstance(param, (int, tuple)):
            if is_filter:
                # For filters, increase the count with depth if single integer provided
                return [param * (2**i) for i in range(num_layers)]
            else:
                # For kernel size, repeat the single value
                return [param] * num_layers
        else:
            raise ValueError(f"Unsupported type for parameter: {type(param)}")


    def _build_conv_layers(self, inputs, num_layers, filters_list, kernel_size_list, activation, add_batch_norm):
        """Builds the sequence of convolutional and pooling layers."""
        x = inputs
        for i in range(num_layers):
            x = tf.keras.layers.Conv2D(
                filters_list[i],
                kernel_size_list[i],
                activation=activation,
                padding='same', # Added padding to maintain spatial dimensions initially
                name=f'conv_{i+1}'
            )(x)
            if add_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            x = tf.keras.layers.MaxPooling2D((2, 2), name=f'max_pool_{i+1}')(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                 # Add dropout after pooling as a common practice
                 x = tf.keras.layers.Dropout(self.dropout_rate, name=f'dropout_conv_{i+1}')(x)
        return x


    def build_model(self) -> tf.keras.Model:
        """
        Builds the Keras model using the Functional API.

        Returns:
            A TensorFlow Keras Model.
        """
        inputs = tf.keras.Input(shape=self.input_shape, name='input_layer')

        # Convolutional Base
        x = self._build_conv_layers(inputs, self.num_layers, self._filter_list,
                                    self._kernel_list, self.activation, self.add_batch_norm)

        # Flatten and Dense layers
        x = tf.keras.layers.Flatten(name='flatten')(x)
        x = tf.keras.layers.Dense(self.dense_units, activation=self.activation, name='dense_1')(x)
        if self.dropout_rate is not None and self.dropout_rate > 0:
             x = tf.keras.layers.Dropout(self.dropout_rate, name='dropout_dense_1')(x)


        # Output layer
        if self.task == 'classification':
            output_activation = 'softmax' # Softmax for multi-class classification
            if self.num_classes == 1:
                 output_activation = 'sigmoid' # Sigmoid for binary classification
            outputs = tf.keras.layers.Dense(self.num_classes, activation=output_activation, name='output_layer')(x)
        elif self.task == 'regression':
            outputs = tf.keras.layers.Dense(1, activation='linear', name='output_layer')(x)
        else:
            # This case should ideally be caught in __init__, but kept for safety
            raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

        # Create the model
        model = tf.keras.Model(inputs=inputs, outputs=outputs)

        return model

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float = 0.001,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam', # Allow custom optimizer
        loss: Optional[Union[str, tf.keras.losses.Loss]] = None, # Allow custom loss
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None # Allow custom metrics
    ) -> tf.keras.Model:
        """
        Compiles the Keras model.

        Args:
            model: The Keras model to compile.
            learning_rate: The learning rate for the optimizer.
            optimizer: The optimizer to use (string name or optimizer instance).
            loss: The loss function to use (string name or loss instance).
                  If None, default loss based on task will be used.
            metrics: A list of metrics to evaluate during training.
                     If None, default metrics based on task will be used.

        Returns:
            The compiled Keras Model.
        """
        if isinstance(optimizer, str):
            optimizer_instance = tf.keras.optimizers.get(optimizer)
            # Set learning rate if it's an Adam optimizer and learning_rate is provided
            if isinstance(optimizer_instance, tf.keras.optimizers.Adam) and learning_rate is not None:
                 optimizer_instance.learning_rate = learning_rate
            # Note: For other optimizers, setting learning_rate might require
            # creating the optimizer instance explicitly with the learning_rate argument.
            # The current approach prioritizes ease of use with string names.
        elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer_instance = optimizer
            # Assume learning rate is already set in the provided optimizer instance
            if learning_rate is not None:
                 print("Warning: learning_rate argument is ignored when an optimizer instance is provided.")
        else:
            raise ValueError("optimizer must be a string name or a tf.keras.optimizers.Optimizer instance.")


        if loss is None:
            if self.task == 'classification':
                # Use categorical_crossentropy for one-hot labels, sparse for integer labels
                # Assuming sparse for simplicity here based on original code, user might need to adjust
                loss_function = 'sparse_categorical_crossentropy' if self.num_classes > 1 else 'binary_crossentropy'
            elif self.task == 'regression':
                loss_function = 'mean_squared_error'
            else:
                raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")
        else:
            loss_function = loss

        if metrics is None:
            if self.task == 'classification':
                metrics_list = ['accuracy']
            elif self.task == 'regression':
                metrics_list = ['mae', 'mse'] # Added mse for regression
            else:
                raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")
        else:
            metrics_list = metrics


        model.compile(
            optimizer=optimizer_instance,
            loss=loss_function,
            metrics=metrics_list
        )

        return model

    def fit(
        self,
        model: tf.keras.Model,
        x_train: tf.data.Dataset | tf.Tensor | np.ndarray, # Use more flexible type hints
        y_train: tf.data.Dataset | tf.Tensor | np.ndarray,
        epochs: int = 10,
        batch_size: Optional[int] = 32, # Batch size can be None for Datasets
        validation_data: Optional[Tuple[tf.Tensor | np.ndarray, tf.Tensor | np.ndarray] | tf.data.Dataset] = None, # Add validation data support
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None # Add callbacks support
    ):
        """
        Trains the Keras model.

        Args:
            model: The compiled Keras model to train.
            x_train: Training input data. Can be a TensorFlow Dataset, Tensor, or NumPy array.
            y_train: Training target data. Can be a TensorFlow Dataset, Tensor, or NumPy array.
            epochs: The number of epochs to train for.
            batch_size: The batch size for training. Ignored if x_train is a Dataset.
            validation_data: Validation data. Can be a tuple of (x_val, y_val) or a Dataset.
            callbacks: A list of Keras callbacks to apply during training.

        Returns:
            A History object containing training loss and metrics.
        """
        if isinstance(x_train, tf.data.Dataset):
             # If using a Dataset, batch_size is typically handled by the Dataset
             if batch_size is not None:
                  print("Warning: batch_size is ignored when x_train is a tf.data.Dataset.")
             history = model.fit(x_train, epochs=epochs, validation_data=validation_data, callbacks=callbacks)
        else:
             history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data, callbacks=callbacks)

        return history

# Example Usage:

if __name__ == '__main__':
    # --- Classification Example ---
    print("Building Classification Model...")
    # Assuming input images are 32x32 with 3 channels (e.g., CIFAR-10)
    classification_builder = ModelBuilder(
        input_shape=(32, 32, 3),
        num_classes=10,
        num_layers=4,
        activation='relu',
        task='classification',
        filters=[16, 32, 64, 128], # Specify filters per layer
        kernel_size=(3, 3),
        dense_units=128,
        dropout_rate=0.5,
        add_batch_norm=True
    )

    classification_model = classification_builder.build_model()
    classification_model.summary()

    compiled_classification_model = classification_builder.compile_model(
        classification_model,
        learning_rate=0.0005,
        optimizer='adam',
        metrics=['accuracy', 'precision', 'recall'] # Add more metrics
    )

    # Create dummy data for demonstration
    import numpy as np
    x_train_cls = np.random.rand(100, 32, 32, 3).astype(np.float32)
    y_train_cls = np.random.randint(0, 10, 100) # Integer labels for sparse_categorical_crossentropy

    x_val_cls = np.random.rand(20, 32, 32, 3).astype(np.float32)
    y_val_cls = np.random.randint(0, 10, 20)

    print("\nTraining Classification Model (with dummy data)...")
    history_cls = classification_builder.fit(
        compiled_classification_model,
        x_train_cls,
        y_train_cls,
        epochs=3,
        validation_data=(x_val_cls, y_val_cls)
    )
    print("Classification Training History:", history_cls.history)


    # --- Regression Example ---
    print("\nBuilding Regression Model...")
    # Assuming input data is still image-like for this CNN example
    regression_builder = ModelBuilder(
        input_shape=(32, 32, 3),
        num_layers=3,
        activation='relu',
        task='regression',
        filters=32, # Use increasing filter count with depth
        kernel_size=5, # Use a single kernel size for all layers
        dense_units=32,
        dropout_rate=0.3
    )

    regression_model = regression_builder.build_model()
    regression_model.summary()

    compiled_regression_model = regression_builder.compile_model(
        regression_model,
        learning_rate=0.001,
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.001), # Use a different optimizer instance
        loss='mse', # Explicitly specify loss
        metrics=['mae'] # Explicitly specify metrics
    )

    # Create dummy data for demonstration
    x_train_reg = np.random.rand(100, 32, 32, 3).astype(np.float32)
    y_train_reg = np.random.rand(100, 1).astype(np.float32) * 100 # Continuous target

    x_val_reg = np.random.rand(20, 32, 32, 3).astype(np.float32)
    y_val_reg = np.random.rand(20, 1).astype(np.float32) * 100

    print("\nTraining Regression Model (with dummy data)...")
    history_reg = regression_builder.fit(
        compiled_regression_model,
        x_train_reg,
        y_train_reg,
        epochs=3,
        validation_data=(x_val_reg, y_val_reg)
    )
    print("Regression Training History:", history_reg.history)