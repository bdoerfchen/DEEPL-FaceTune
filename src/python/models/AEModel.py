## Author: Benjamin Bissendorf

class AEModel:
    """Base class for autoencoder models to inherit from. Will be used to encode and decode images"""

    def load(self) -> None:
        """Initial method to load the model from disk and initialize"""
        pass

    def getName(self) -> str:
        """Return the name of the model"""
        return ""
    
    def encode(self, img) -> list[float]:
        """Encode an image and return the latent encoding vector"""
        return []
    
    def decode(self, latent: list[float]) -> []:
        """Decode the latent vector into an image"""
        return []
