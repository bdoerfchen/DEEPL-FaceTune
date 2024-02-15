## Authors: Benjamin Bissendorf and Niklas Seeliger

from .models.AEModel import AEModel
from .ftutilities import imageToB64

class ModelResult:
    """A class representing the results of a model inference"""

    def __init__(self, model : AEModel, image) -> None:
        self.name = model.getName()
        self.encoding = model.encode(image)
        self.result = model.decode(self.encoding)
        self.result_b64 = imageToB64(self.result)