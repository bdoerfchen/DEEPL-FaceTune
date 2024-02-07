import keras

class MSEReconstructionLossLayer(keras.layers.Layer):
    
    """ Layer that calculates the mean squared error between input and output. 
    It expects [y_true, y_predict] and returns y_predict.
    """
    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(MSEReconstructionLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        y_true, y_predict = inputs

        reconstruction_loss = keras.backend.square((y_true-y_predict)*100)
        reconstruction_loss = keras.backend.mean(reconstruction_loss, axis=[0, 1, 2, 3])
        self.add_loss(reconstruction_loss, inputs=inputs)
        self.add_metric(reconstruction_loss, name="mse_loss")
        return y_predict