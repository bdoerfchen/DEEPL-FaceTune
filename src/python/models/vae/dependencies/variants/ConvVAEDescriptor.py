import keras
import numpy as np

from .VAEDescriptor import VAEDescriptor

from .layers.KLDivergenceLayer import KLDivergenceLossLayer


class ConvLayer():
    """A class representing single convolutional layer"""

    def __init__(self, filters: int, kernel: int, strides: int = 1, padding: str = "same", maxpool: int|None = 2) -> None:
        """Create new ConvLayer instance
        
        Args:
            filters (int): Number of filters
            kernel (int): Size of squared kernel
            strides (int): Number of striding pixels
            padding (str): 'valid' or 'same' to pad
            maxpool (int): size of pooling after convolutions. Can be None to not use max pooling"""
        assert filters > 0, "Expecting at least one filter"
        assert kernel > 0, "Expecting positive kernel size"
        assert strides >= 1, "Expecting strides to be >= 1"
        assert padding in ["same", "valid"], "Invalid value for padding"
        assert maxpool == None or maxpool >= 1, "Invalid value for maxpool"

        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.maxpool = maxpool 
        pass

    def addEncoderLayers(self, previous_layer: keras.layers.Layer) -> keras.layers.Layer:
        """Create convolutional layer for the encoder
        
        Args:
            previous_layer (keras.layers.Layer): The layer to get the input from
        Returns:
            The last created keras.layers.Layer as a result"""
        
        conv = keras.layers.Conv2D(self.filters, self.kernel, activation="relu", strides=self.strides, padding=self.padding)(previous_layer)
        if self.maxpool != None:
            maxpool = keras.layers.MaxPool2D((self.maxpool, self.maxpool))(conv)
            return maxpool
        else:
            return conv
        
    def addDecoderLayers(self, previous_layer: keras.layers.Layer) -> keras.layers.Layer:
        """Create convolutional layer for the decoder
        
        Args:
            previous_layer (keras.layers.Layer): The layer to get the input from
        Returns:
            The last created keras.layers.Layer as a result"""
        
        if self.maxpool != None:
            previous_layer = keras.layers.UpSampling2D((self.maxpool, self.maxpool))(previous_layer)
        conv = keras.layers.Conv2DTranspose(self.filters, self.kernel, activation="relu", strides=self.strides, padding=self.padding)(previous_layer)
        return conv
    

class ConvVAEDescriptor(VAEDescriptor):
    """The descriptor for convolutional variational autoencoder."""
    
    def __init__(self, img_shape: tuple, units: list[ConvLayer], prelatent_factor: int, latent_size: int) -> None:
        """Create new instance of a ConvVAE
        
        Args:
            img_shape (tuple): The shape of the input data, seen as (height, width, channels)
            units (list): List of layers of the encoder. The decoder will be mirrored
            prelatent_factor (int): A factor determining the pre/post-latent layer size as latent_size * prelatent_factor
            latent_size (int): The size of the latent layers, where mean and log_variance each have this number of parameters"""
        assert len(units) > 0, "Expecting at least one layer"
        assert latent_size > 0, "Expecting at least one parameter in latent layer"

        self.img_shape = img_shape
        self.units = units
        self.latent_size = latent_size
        self.prelatent_size = latent_size * prelatent_factor

    def createModel(self) -> tuple[keras.models.Model, keras.models.Model]:
        # Doc comment is provided by base class

        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=self.img_shape, name="encoder_input")

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

        latent_space = KLDivergenceLossLayer()([encoder_mean, encoder_log_variance]) # Does nothing but computes the kullback leibler divergence and adds loss

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
        decoder_output = keras.layers.Conv2DTranspose(self.img_shape[-1], 3, activation="sigmoid", padding="same")(last_layer)

        decoder = keras.models.Model(decoder_input, decoder_output, name="decoder_model")
        decoder.summary()
        
        return (encoder, decoder)




