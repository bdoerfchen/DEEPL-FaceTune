import os
import tensorflow as tf
import keras
import numpy as np

from python.models.AEModel import AEModel
from python.models.vae.dependencies.BaseVAE import BaseVAE
from python.models.vae.dependencies.DenseVAEDescriptor import DenseVAEDescriptor


class VariationalAE(AEModel):

    MODEL_FILENAME = "vae"
    MODEL_DIRPATH = "./src/python/models/vae"

    def getName(self) -> str:
        return "VAE"
    
    def load(self) -> None:        
        d = DenseVAEDescriptor((256, 256, 3), [300, 100], 16, 1/4)
        self.model = BaseVAE.load_from_directory(d, VariationalAE.MODEL_DIRPATH)

    def expects_shape() -> tuple:
        return (256, 256, 3)

    def encode(self, img) -> list[float]:
        assert img.shape == (256, 256, 3)

        #Bild normalisieren
        img_i = np.multiply(img, 1/255)
        img_i = np.reshape(img, (1, 256, 256, 3))

        r = np.array(np.multiply(self.model(img_i), 255)).astype(np.uint8)

        # extract layers and pass-forward
        l_input = self.model.layers[0]
        l_encode = self.model.layers[1]
        l_sample = self.model.layers[2]
        latent = l_sample(l_encode(l_input(img_i)))
        latent_reshape = np.reshape(latent, latent.shape[1:])
        return latent_reshape
    
    def decode(self, latent: list[float]) -> []:
        # reshape
        latent_i = np.reshape(latent, (1, len(latent)))

        # extract layers and pass-forward
        l_decode = self.model.layers[3]
        l_out = self.model.layers[4]
        result = l_out(l_decode(latent_i))

        # process output
        result_img = np.multiply(result, 255)
        result_img = result_img.astype(np.uint8)
        result_img = np.reshape(result_img, result_img.shape[1:])
        return result_img