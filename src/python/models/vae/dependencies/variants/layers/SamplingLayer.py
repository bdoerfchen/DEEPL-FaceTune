import keras

# Author: Francois Chollet (https://keras.io/examples/generative/vae/)
# The content is pretty much copied from him
class VAESamplingLayer(keras.layers.Layer):
    """The sampling layer for VAEs. Uses the 'reparametrization trick' and introduces randomness"""
    
    # def __init__(self, *args, **kwargs):
    #     self.is_placeholder = True
    #     super(VAESamplingLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        """Args:
            inputs (list): Provide inputs as [mean, log_variance]
        Returns:
            Sampled latent values (mean + e^(log_variance/2)*eps)"""

        mean, log_variance = inputs

        batch = keras.backend.shape(mean)[0]
        dim = keras.backend.shape(mean)[1]
        epsilon = keras.backend.random_normal(shape=(batch, dim))       # Generate random values, following a gaussian normal distribution
        return mean + keras.backend.exp(0.5 * log_variance) * epsilon   # Return sampled values