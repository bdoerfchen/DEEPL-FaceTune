## Author: Niklas Seeliger

from python.models.AEModel import AEModel
from python.models.AEGAN.lazarou.generative_model import AutoEncodingGenerativeAdversarialNetwork

import numpy as np
import cv2 as cv
# import matplotlib.pyplot as plt

class AEGANModel(AEModel):

    def load(self) -> None:
        
        encoder_path = "src/python/models/AEGAN/gcurrent.encoder.h5"
        decoder_path = "src/python/models/AEGAN/gcurrent.generator.h5"
        
        latent_dims = 64

        trained_aegan = AutoEncodingGenerativeAdversarialNetwork(
            image_shape=(64,64,3), 
            latent_dim=latent_dims, 
            parameter_json_path='src/python/models/AEGAN/params_64.json', 
            data_generating_function=lambda x: np.zeros((64,64,3), dtype=int),
            noise_generating_function=lambda x: np.random.normal(0, 1, (x, latent_dims)).astype(np.float32))
        
        trained_aegan.encoder.load_weights(encoder_path)
        trained_aegan.generator.load_weights(decoder_path)
        
        self.aegan = trained_aegan

    def getName(self) -> str:
        """Return the name of the model"""
        return "AEGAN"
    
    def encode(self, img) -> list[float]:
        """Encode an image and return the latent encoding vector"""
        
        # resize image to 64x64
        im = cv.resize(img, (64, 64))
        
        # normalize to -1 to 1
        im = np.array(im) / 255.0
        im = im * 2 - 1
        
        # predict
        encoded = self.aegan.encode(np.array([im]))[0]
        
        return encoded
    
    def decode(self, latent: list[float]) -> []:
        """Decode the latent vector into an image"""
        
        # Generate output using the neural network model
        
        input = np.array(latent).reshape(1, -1)
        output = self.aegan.generator.predict(input)
        # Rescale output to RGB values
        output = np.clip((output/2+.5) * 255, 0, 255).astype(np.uint8)
        # Reshape output to image dimensions
        im = output.reshape(64, 64, 3)
        img = cv.resize(im, (256, 256))
        
        return img
