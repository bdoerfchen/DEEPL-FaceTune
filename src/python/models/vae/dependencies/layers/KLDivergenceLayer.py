import keras

class KLDivergenceLossLayer(keras.layers.Layer):

    """Identity transform layer that adds KL divergence
    to the final model loss.
    """
    KL_LOSS_FACTOR = 1

    def __init__(self, *args, **kwargs):
        self.is_placeholder = True
        super(KLDivergenceLossLayer, self).__init__(*args, **kwargs)

    def call(self, inputs):
        mean, log_variance = inputs

        kl_batch = keras.backend.sum(-0.5 * (1 + log_variance - keras.backend.square(mean) - keras.backend.exp(log_variance)), axis=[1])
        kl_loss = keras.backend.mean(kl_batch * KLDivergenceLossLayer.KL_LOSS_FACTOR, axis=-1)
        self.add_loss(kl_loss, inputs=inputs)
        self.add_metric(kl_loss, name="kl_loss")
        return inputs