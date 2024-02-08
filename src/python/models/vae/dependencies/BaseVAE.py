import os
#from typing import Self
import keras
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np

from .VAEDescriptor import VAEDescriptor

from .layers.KLDivergenceLayer import KLDivergenceLossLayer
from .layers.MSEReconstructionLossLayer import MSEReconstructionLossLayer
from .layers.SamplingLayer import VAESamplingLayer


# https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/


class BaseVAE(keras.models.Model):
    """The base class for all VAE models

    They need to be initialized with an descriptor, describing their internal structure and thus determining the VAE variant
    """

    def __init__(self, descriptor: VAEDescriptor, name: str = "VAE") -> None:
        """Create a new VAE instance

        Args:
            descriptor (VAEDescriptor): An object describing the structure of the model, determines the kind of VAE
            name (str): The name of the model
        """

        encoder, decoder = descriptor.createModel()

        #   -------
        #   [ VAE ]
        vae_input = encoder.input
        vae_latent = encoder(vae_input)
        vae_latent_sampled = VAESamplingLayer(name="vae_latent_sampled")(vae_latent)
        vae_decoder_output = decoder(vae_latent_sampled)
        vae_mse_output = MSEReconstructionLossLayer(name="mse_vae_output")([vae_input, vae_decoder_output])

        super().__init__(vae_input, vae_mse_output, name=name)
        return

    def save_to_directory(self, directory, file = "vae") -> None:
        self.save(os.path.join(directory, file + ".keras"), save_format="keras")
        return

    def load_from_directory(descriptor, directory: str, file = "vae"):
        assert os.path.exists(directory)
        vae = BaseVAE(descriptor)
        vae.load_weights(os.path.join(directory, file + ".keras"))
        return vae