import tensorflow as tf
from typing import Optional, Literal, Tuple, List, Union
import numpy as np

class ModelBuilder:
    """
    A class to build, compile, and optionally train a TensorFlow Keras Dense Neural Network model.
    Supports both classification and regression tasks.
    """
    def __init__(
        self,
        input_shape: Tuple[int, ...],
        num_layers: int = 2,
        num_classes: Optional[int] = None,
        activation: str = 'relu',
        task: Literal['classification', 'regression'] = 'classification',
        dense_units: int = 64,
        dropout_rate: Optional[float] = None,
        add_batch_norm: bool = False
    ):
        """
        Initializes the ModelBuilder with configuration for constructing a Dense NN model.
        Args:
            input_shape: Shape of the input data as a tuple (features,).
            num_layers: Number of dense hidden layers.
            num_classes: Number of output classes for classification tasks; required if task is 'classification'.
            activation: Activation function to use in hidden layers.
            task: Specifies whether the model is for 'classification' or 'regression'.
            dense_units: Number of units in each dense hidden layer.
            dropout_rate: Optional dropout rate applied after dense layers.
            add_batch_norm: Whether to include batch normalization after each dense layer.
        Raises:
            ValueError: If required parameters are missing or have invalid formats.
        """
        if task == 'classification' and num_classes is None:
            raise ValueError("num_classes must be specified for classification task")
        if not isinstance(input_shape, tuple) or not all(isinstance(d, int) for d in input_shape):
            raise ValueError("input_shape must be a tuple of integers.")
        if len(input_shape) != 1:
            raise ValueError("input_shape must be a 1D tuple (features,) for Dense layers.")

        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation
        self.task = task
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate
        self.add_batch_norm = add_batch_norm

    def build_model(self) -> tf.keras.Model:
        """
        Constructs and returns a Keras Dense NN model based on the specified architecture and task.
        Returns:
            A TensorFlow Keras Model instance ready for compilation and training.
        """
        inputs = tf.keras.Input(shape=self.input_shape, name='input_layer')
        x = inputs
        for i in range(self.num_layers):
            x = tf.keras.layers.Dense(self.dense_units, activation=self.activation, name=f'dense_{i+1}')(x)
            if self.add_batch_norm:
                x = tf.keras.layers.BatchNormalization(name=f'batch_norm_{i+1}')(x)
            if self.dropout_rate is not None and self.dropout_rate > 0:
                x = tf.keras.layers.Dropout(self.dropout_rate, name=f'dropout_{i+1}')(x)

        # Output layer
        if self.task == 'classification':
            output_activation = 'softmax'
            if self.num_classes == 1:
                output_activation = 'sigmoid'
            outputs = tf.keras.layers.Dense(self.num_classes, activation=output_activation, name='output_layer')(x)
        elif self.task == 'regression':
            outputs = tf.keras.layers.Dense(1, activation='linear', name='output_layer')(x)
        else:
            raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        return model

    def compile_model(
        self,
        model: tf.keras.Model,
        learning_rate: float = 0.001,
        optimizer: Union[str, tf.keras.optimizers.Optimizer] = 'adam',
        loss: Optional[Union[str, tf.keras.losses.Loss]] = None,
        metrics: Optional[List[Union[str, tf.keras.metrics.Metric]]] = None
    ) -> tf.keras.Model:
        """
        Compiles a Keras model with specified optimizer, loss function, and metrics.
        Args:
            model: The Keras model to compile.
            learning_rate: Learning rate for the optimizer (used if applicable).
            optimizer: Optimizer to use, specified as a string or optimizer instance.
            loss: Loss function to use; defaults to task-appropriate loss if None.
            metrics: List of metrics to evaluate during training; defaults to task-appropriate metrics if None.
        Returns:
            The compiled Keras Model.
        """
        if isinstance(optimizer, str):
            if 'adam' in optimizer.lower():
                optimizer_instance = tf.keras.optimizers.Adam(learning_rate=learning_rate)
            elif 'sgd' in optimizer.lower():
                optimizer_instance = tf.keras.optimizers.SGD(learning_rate=learning_rate)
            elif 'rmsprop' in optimizer.lower():
                optimizer_instance = tf.keras.optimizers.RMSprop(learning_rate=learning_rate)
            else:
                optimizer_instance = tf.keras.optimizers.get(optimizer)
        elif isinstance(optimizer, tf.keras.optimizers.Optimizer):
            optimizer_instance = optimizer
            if learning_rate is not None:
                print("Warning: learning_rate argument is ignored when an optimizer instance is provided.")
        else:
            raise ValueError("optimizer must be a string name or a tf.keras.optimizers.Optimizer instance.")

        if loss is None:
            if self.task == 'classification':
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
                metrics_list = ['mae', 'mse']
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
        x_train: tf.data.Dataset | tf.Tensor | np.ndarray,
        y_train: tf.data.Dataset | tf.Tensor | np.ndarray,
        epochs: int = 10,
        batch_size: Optional[int] = 32,
        validation_data: Optional[Tuple[tf.Tensor | np.ndarray, tf.Tensor | np.ndarray] | tf.data.Dataset] = None,
        callbacks: Optional[List[tf.keras.callbacks.Callback]] = None
    ):
        """
        Trains the provided Keras model on the given data.
        Supports training with TensorFlow Datasets, Tensors, or NumPy arrays as input. Optionally accepts validation data and callbacks.
        Args:
            model: The compiled Keras model to train.
            x_train: Training input data as a TensorFlow Dataset, Tensor, or NumPy array.
            y_train: Training target data as a TensorFlow Dataset, Tensor, or NumPy array.
            epochs: Number of epochs to train.
            batch_size: Batch size for training; ignored if x_train is a Dataset.
            validation_data: Optional validation data as a tuple (x_val, y_val) or a Dataset.
            callbacks: Optional list of Keras callbacks to use during training.
        Returns:
            A Keras History object containing training loss and metrics.
        """
        if isinstance(x_train, tf.data.Dataset):
            if batch_size is not None:
                print("Warning: batch_size is ignored when x_train is a tf.data.Dataset.")
            history = model.fit(x_train, epochs=epochs, validation_data=validation_data, callbacks=callbacks)
        else:
            history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size,
                                validation_data=validation_data, callbacks=callbacks)
        return history