import keras

class MSEReconstructionLossLayer(keras.layers.Layer):
    """ Layer that calculates the mean squared error between input and output"""

    INNER_FACTOR = 100
    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSEReconstructionLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """Expects inputs=[y_true, y_predict] and returns y_predict."""
        y_true, y_predict = inputs

        # Square pixel differences, multiplied by constant inner factor for giving more weight to reconstruction
        reconstruction_loss = keras.backend.square((y_true-y_predict) * MSEReconstructionLossLayer.INNER_FACTOR)
        reconstruction_loss = keras.backend.mean(reconstruction_loss, axis=[0, 1, 2, 3]) # Get the mean value
        self.add_loss(reconstruction_loss, inputs=inputs)
        self.add_metric(reconstruction_loss, name="mse_loss")   # Add loss as metric too to appear in history
        return y_predict