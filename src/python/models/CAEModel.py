from python.models.AEModel import AEModel
import keras
import cv2 as cv
import numpy as np
class CAEModel(AEModel):
    model = None


    def __init__(self, img_size):
        self.img_size = img_size

    def load(self) -> None:
        model_path = "src/python/models/CAE/BestQualityCAE.tf"
        self.model = keras.models.load_model(model_path)

    def getName(self) -> str:
        """Return the name of the model"""
        return "Convolutional Autoencoder"

    def encode(self, img) -> list[float]:
        """Encode an image and return the latent encoding vector"""

        # Resize Image
        im = cv.resize(img, (256, 384))
        # Normalize
        im = np.array(im) / 255.0

        # Wrap Image in Array
        wrapper = np.array([im])

        # Encode Image
        encoded = self.model.encoder.predict(wrapper)[0]
        return encoded

    def decode(self, latent: list[float]) -> []:


        wrapper = np.array([latent])

        # Decode Image
        decoded = self.model.decoder.predict(wrapper)[0]

        # Scale Back to 0 to 255
        img = np.array(decoded) * 255.0

        # Convert Pixel Data to Bytes
        img = np.clip(img,0,255).astype(np.uint8)
        img = cv.resize(img, self.img_size)
        return img