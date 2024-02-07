from .AEModel import AEModel
from .ftutilities import imageToB64

class ModelResult:

    def __init__(self, model : AEModel, image) -> None:
        self.name = model.getName()
        self.encoding = model.encode(image)
        self.result = model.decode(self.encoding)
        self.result_b64 = imageToB64(self.result)