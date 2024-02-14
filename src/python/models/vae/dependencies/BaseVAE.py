import os
from typing import Self
import keras
from keras.utils import CustomObjectScope
import tensorflow as tf
import numpy as np
import jsonpickle

from variants.VAEDescriptor import VAEDescriptor

from variants.layers.KLDivergenceLayer import KLDivergenceLossLayer
from variants.layers.MSEReconstructionLossLayer import MSEReconstructionLossLayer
from variants.layers.SamplingLayer import VAESamplingLayer


# Some influences by https://blog.paperspace.com/how-to-build-variational-autoencoder-keras/


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
        self.descriptor = descriptor                    # Save descriptor
        encoder, decoder = descriptor.createModel()     # Use descriptor to create encoder and decoder model

        # Construct the internal structure with the encoder and decoder models
        # It is: input -> encoder -> latent space -> sampled latent space -> decoder -> (add mse) -> output
        vae_input = encoder.input
        vae_latent = encoder(vae_input)
        vae_latent_sampled = VAESamplingLayer(name="vae_latent_sampled")(vae_latent)
        vae_decoder_output = decoder(vae_latent_sampled)
        vae_mse_output = MSEReconstructionLossLayer(name="mse_vae_output")([vae_input, vae_decoder_output])

        super().__init__(vae_input, vae_mse_output, name=name)
        return

    def save_to_directory(self, directory, file = "vae") -> None:
        filepath = os.path.join(directory, file + ".keras")
        keras.saving.save_model(self, filepath, save_format="keras")
        return

    def load_from_directory(directory: str, file = "vae") -> Self:
        assert os.path.exists(directory)
        filepath = os.path.join(directory, file + ".keras")
        with CustomObjectScope({
            'BaseVAE': BaseVAE,
            'KLDivergenceLossLayer': KLDivergenceLossLayer,
            'MSEReconstructionLossLayer': MSEReconstructionLossLayer,
            'VAESamplingLayer': VAESamplingLayer
        }):
            vae = keras.saving.load_model(filepath)
            return vae
    
    def get_config(self):
        super_config = super().get_config()
        d_json = jsonpickle.encode(self.descriptor)
        return {
            'descriptor': d_json,
            **super_config
        }
    
    @classmethod
    def from_config(cls, config):
        descriptor_json = config.pop('descriptor')
        descriptor = jsonpickle.decode(descriptor_json)
        return cls(descriptor, **config)