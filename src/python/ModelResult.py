from AEModel import AEModel

class ModelResult:

    def __init__(self, model : AEModel, image) -> None:
        self.name = model.getName()
        self.encoding = model.encode(image)
        self.result = model.decode(self.encoding)