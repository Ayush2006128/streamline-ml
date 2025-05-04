import tensorflow as tf

class StreamlitProgressCallback(tf.keras.callbacks.Callback):
    """A Keras callback to update a Streamlit progress bar."""
    def __init__(self, progress_bar, num_epochs):
        super().__init__()
        self.progress_bar = progress_bar
        self.num_epochs = num_epochs

    def on_epoch_begin(self, epoch, logs=None):
        self.progress_bar.progress(
            epoch / self.num_epochs,
            text=f"Epoch {epoch + 1}/{self.num_epochs}"
        )

    def on_epoch_end(self, epoch, logs=None):
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
