## Author: Benjamin Bissendorf

import os
import tensorflow as tf
import keras
import numpy as np
from PIL import Image

from python.models.AEModel import AEModel
from python.models.vae.dependencies.BaseVAE import BaseVAE


class VariationalAE(AEModel):

    MODEL_FILENAME = "vae"
    MODEL_DIRPATH = "./src/python/models/vae"

    def getName(self) -> str:
        return "VAE"
        
    def load(self) -> None:        
        self.model = BaseVAE.load_from_directory(VariationalAE.MODEL_DIRPATH, file="vae")

    def encode(self, img) -> list[float]:
        # Rescale image to fit expected size
        img_r = np.array(Image.fromarray(img).resize((256, 256)))

        # Normalize image
        img_i = np.multiply(img_r, 1/255)
        img_i = np.reshape(img_i, (1, 256, 256, 3))

        # Encode and sample
        latent = self.model.encoder(img_i) # Encode
        sampled_latent = self.model.layers[2](latent) # Sample

        # Reshape and return
        latent_reshape = np.reshape(sampled_latent, sampled_latent.shape[1:])
        return latent_reshape
    
    def decode(self, latent: list[float]) -> list:
        # reshape
        latent_i = np.reshape(latent, (1, len(latent)))

        result = self.model.decoder(latent_i)

        # process output
        result_img = np.multiply(result, 255) # Denormalize
        result_img = result_img.astype(np.uint8) # To discrete values (uint8)
        result_img = np.reshape(result_img, result_img.shape[1:]) # Reshaping
        return result_img
    
    def isModelAvailable() -> bool:
        filepath = os.path.join(VariationalAE.MODEL_DIRPATH, VariationalAE.MODEL_FILENAME + ".keras")
        return os.path.exists(filepath)