import keras

class VAESamplingLayer(keras.layers.Layer):
    
    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(VAESamplingLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))
        return mean + keras.backend.exp(0.5 * log_variance) * epsilon