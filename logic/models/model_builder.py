import tensorflow as tf
from typing import Optional, Literal

class ModelBuilder:
    def __init__(self, input_shape: int, num_layers: Optional[int] = 5, num_classes: int = None, activation='relu', task: Literal['classification', 'regression'] = 'classification'):
        """
        Args:
            input_shape: The shape of the input data.
            num_classes: The number of classes for classification task.
            num_layers: The number of Conv2D layers in the model.
            activation: The activation function to use.
            task: The type of task, either 'classification' or 'regression'.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.activation = activation
        self.task = task

    def build_model(self):
        model = tf.keras.Sequential()
        # Input layer
        model.add(tf.keras.layers.Conv2D(32, (3, 3), activation=self.activation, input_shape=self.input_shape))
        model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        # Hidden Conv2D layers
        for i in range(1, self.num_layers):
            model.add(tf.keras.layers.Conv2D(32 * (i+1), (3, 3), activation=self.activation))
            model.add(tf.keras.layers.MaxPooling2D((2, 2)))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(64, activation=self.activation))
        # Output layer
        if self.task == 'classification':
            if self.num_classes is None:
                raise ValueError("num_classes must be specified for classification task")
            model.add(tf.keras.layers.Dense(self.num_classes, activation='softmax'))
        elif self.task == 'regression':
            model.add(tf.keras.layers.Dense(1, activation='linear'))
        else:
            raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")
        return model
    def compile_model(self, model, learning_rate=0.001):
        if self.task == 'classification':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='sparse_categorical_crossentropy',
                          metrics=['accuracy'])
        elif self.task == 'regression':
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                          loss='mean_squared_error',
                          metrics=['mae'])
        else:
            raise ValueError("Unsupported task type. Use 'classification' or 'regression'.")
        
        return model
    def fit(self, model, x_train, y_train, epochs=10, batch_size=32):
        history = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
        return history