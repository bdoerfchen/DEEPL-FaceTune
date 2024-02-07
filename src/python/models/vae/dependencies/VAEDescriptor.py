import keras

class VAEDescriptor:
    """
    An abstract class for describing the structure of a VAE
    """
    def __init__(self) -> None:
        pass

    def createModel(self) -> (keras.models.Model, keras.models.Model):
        """Creates and returns the encoder and decoder.

        Info:
            The created encoder model's output is expected to be (mean, log_variance)

        Args:
            None
        
        Returns:
            encoder (Model): The encoder model, whose output is the mean and log_variance latent space
            decoder (Model): The decoder model, which expects an already sampled latent space
        """
        return (None,None)
    
    def getObjectContext(self) -> None:
        pass