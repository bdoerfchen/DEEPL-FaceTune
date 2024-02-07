import keras
import numpy as np

from .VAEDescriptor import VAEDescriptor

from .layers.KLDivergenceLayer import KLDivergenceLossLayer


class ConvLayer():
    def __init__(self, filters: int, kernel: int, strides: int = 1, padding: str = "same", maxpool: int|None = 2) -> None:
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.maxpool = maxpool 
        pass

    def addEncoderLayers(self, previous_layer) -> keras.layers.Layer:
        conv = keras.layers.Conv2D(self.filters, self.kernel, activation="relu", strides=self.strides, padding=self.padding)(previous_layer)
        if self.maxpool != None:
            maxpool = keras.layers.MaxPool2D((self.maxpool, self.maxpool))(conv)
            return maxpool
        else:
            return conv
        
    def addDecoderLayers(self, previous_layer) -> keras.layers.Layer:
        if self.maxpool != None:
            previous_layer = keras.layers.UpSampling2D((self.maxpool, self.maxpool))(previous_layer)
        conv = keras.layers.Conv2DTranspose(self.filters, self.kernel, activation="relu", strides=self.strides, padding=self.padding)(previous_layer)
        return conv
    

class ConvVAEDescriptor(VAEDescriptor):
    
    def __init__(self, img_shape, units: list[ConvLayer], prelatent_factor, latent_size) -> None:
        self.img_shape = img_shape
        self.units = units
        self.latent_size = latent_size
        self.prelatent_size = latent_size * prelatent_factor

    def createModel(self) -> (keras.models.Model, keras.models.Model):
        img_shape = self.img_shape

        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=img_shape, name="encoder_input")

        # Encoder convoluted layers
        last_layer = encoder_input
        for unit in self.units:
            x = unit.addEncoderLayers(last_layer)
            last_layer = x

        # Flatten for (pre)latent layers
        beforeflattened_shape = keras.backend.int_shape(last_layer)[1:]
        conv_flatten = keras.layers.Flatten()(last_layer)

        # Latent layers
        encoder_prelatent = keras.layers.Dense(units=self.prelatent_size, name="prelantent", activation="relu")(conv_flatten)
        encoder_mean = keras.layers.Dense(units=self.latent_size, name="encoder_mean")(encoder_prelatent)
        encoder_log_variance = keras.layers.Dense(units=self.latent_size, name="encoder_log_variance")(encoder_prelatent)

        latent_space = KLDivergenceLossLayer()([encoder_mean, encoder_log_variance]) # Does nothing but computes the kldivergence and adds loss

        encoder = keras.models.Model(encoder_input, latent_space, name="encoder_model")
        encoder.summary()

        #   -----------
        #   [ Decoder ]
        decoder_input = keras.layers.Input(shape=(self.latent_size), name="decoder_input")
        decoder_postlatent = keras.layers.Dense(units=self.prelatent_size, name="decoder_postlatent", activation="relu")(decoder_input)

        # Restore convoluted layers
        decoder_conv_redense = keras.layers.Dense(units=np.prod(beforeflattened_shape), name="decoder_conv_redense", activation="relu")(decoder_postlatent)
        decoder_reshape = keras.layers.Reshape(target_shape=beforeflattened_shape)(decoder_conv_redense)

        # Decoder convoluted layers
        last_layer = decoder_reshape
        for unit in list(reversed(self.units)):
            x = unit.addDecoderLayers(last_layer)
            last_layer = x

        # Restore three channels and use sigmoid to limit to 0..1
        decoder_output = keras.layers.Conv2DTranspose(img_shape[-1], 3, activation="sigmoid", padding="same")(last_layer)

        decoder = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
        decoder.summary()
        
        return (encoder, decoder)




