import keras
import numpy as np

from .VAEDescriptor import VAEDescriptor
from .layers.KLDivergenceLayer import KLDivergenceLossLayer

class DenseVAEDescriptor(VAEDescriptor):
    
    def __init__(self, img_shape, encoder_layer_units : list, latent_size: int = 20, resize_factor = 1, resize_interpolation_method = "bilinear") -> None:
        """ Creating a new Dense-Descriptor instance.

        Args:
            img_shape (tuple): shape of the input image (height, width, channels)
            encoder_layer_units (list): list of units per each layer. Will be mirrored
            latent_size (int): Number of parameters in latent layer
            resize_factor (float): Factor to resize input image. Should be 1/2^x for any x>=0 to avoid down- and upscaling problems
            resize_interpolation_method (str): The method for resizing. Can be one of the following: ["bilinear", "nearest", "bicubic", "area", "lanczos3", "lanczos5", "gaussian", "mitchellcubic"]
        """

        assert len(encoder_layer_units) > 0, "Expecting at least one dense layer"
        assert latent_size > 0, "Expecting at least one parameter in latent layer"
        assert resize_factor <= 1, "Descriptor will not upscale image"

        self.img_shape = img_shape
        self.encoder_layer_units = encoder_layer_units
        self.latent_size = latent_size
        self.resize_factor = resize_factor
        self.interpolation = resize_interpolation_method

    def createModel(self) -> tuple[keras.models.Model, keras.models.Model]:
        # Doc comment is provided by base class

        img_shape = self.img_shape
        encoder_layer_units = self.encoder_layer_units
        latent_size = self.latent_size
        resized_shape = np.multiply(img_shape, self.resize_factor).astype(np.int32)
        resized_shape[2] = img_shape[2]


        #   [ Encoder ]
        encoder_input = keras.layers.Input(shape=img_shape, name="encoder_input")
        encoder_input_downscaled = keras.layers.Resizing(resized_shape[0], resized_shape[1], interpolation=self.interpolation)(encoder_input)
        encoder_flatten = keras.layers.Flatten()(encoder_input_downscaled)

        last_layer = encoder_flatten
        for units in encoder_layer_units:
            x = keras.layers.Dense(units, activation="relu")(last_layer)
            last_layer = x

        encoder_mean = keras.layers.Dense(units=latent_size, name="encoder_mean")(last_layer)
        encoder_log_variance = keras.layers.Dense(units=latent_size, name="encoder_log_variance")(last_layer)

        latent_space = KLDivergenceLossLayer()([encoder_mean, encoder_log_variance]) # Does nothing but computes the kldivergence and adds loss

        encoder = keras.models.Model(encoder_input, latent_space, name="encoder_model")
        encoder.summary()

        #   -----------
        #   [ Decoder ]
        decoder_input = keras.layers.Input(shape=(latent_size), name="decoder_input")

        last_layer = decoder_input
        for units in list(reversed(encoder_layer_units)):
            x = keras.layers.Dense(units, activation="relu")(last_layer)
            last_layer = x

        decoder_output = keras.layers.Dense(keras.backend.prod(resized_shape), activation="sigmoid")(last_layer)
        decoder_reshape = keras.layers.Reshape(target_shape=resized_shape)(decoder_output)
        decoder_output_upscaled = keras.layers.Resizing(img_shape[0], img_shape[1], interpolation=self.interpolation)(decoder_reshape)

        decoder = keras.models.Model(decoder_input, decoder_output_upscaled, name="decoder_model")
        decoder.summary()
        
        return (encoder, decoder)
