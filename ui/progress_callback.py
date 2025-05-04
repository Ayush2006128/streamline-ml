import tensorflow as tf

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """A Keras callback to update a Streamlit progress bar."""
    def __init__(self, progress_bar, num_epochs):
        """
        Initializes the StreamlitProgressCallback with a progress bar and total epochs.
        
        Args:
            progress_bar: A Streamlit progress bar object to be updated during training.
            num_epochs: The total number of training epochs.
        """
        super().__init__()
        self.progress_bar = progress_bar
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        """
        Updates the Streamlit progress bar at the start of each training epoch.
        
        Args:
            epoch: The current epoch index (zero-based).
            logs: Optional dictionary of logs (unused).
        """
        self.progress_bar.progress(
            epoch / self.num_epochs,
            text=f"Epoch {epoch + 1}/{self.num_epochs}"
        )

    def on_epoch_end(self, epoch, logs=None):
        """
        Updates the Streamlit progress bar at the end of an epoch with current progress and loss metrics.
        
        At the end of each training epoch, this method sets the progress bar to the new completion fraction and displays the epoch number along with training and validation loss values if available.
        """
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
