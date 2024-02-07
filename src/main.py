import eel 
from python.models.AEModel import AEModel
from python.models.IdentityAE import IdentityAE 
from python.models.AEGANModel import AEGANModel
from python.models.CAEModel import CAEModel
from python.ModelResult import ModelResult
from python.ftutilities import *

models : list[AEModel] = [
    AEGANModel(), CAEModel(), IdentityAE(), IdentityAE()
]
def main():
    # Init eel
    eel.init("./src/web")          

    # Load and init models
    for model in models:
        model.load()

    # Start the index.html file 
    eel.start("index.html", mode=None)

@eel.expose
def decodeImage(baseImage) -> list:
    """A function to en- and decode and image with all available models"""
    img = b64ToImage(baseImage)

    result : list[ModelResult] = []

    for model in models:
        result.append(
            ModelResult(model, img).result_b64
        )
    return result

@eel.expose
def decodeLatentEncoding(name, encoding):
    assert isinstance(encoding, list)

    model = [m for m in models if m.getName() == name]
    assert len(model) == 1
    model = model[0]

    resultImage = model.decode(encoding)
    b64string = imageToB64(resultImage)
    return b64string


if __name__ == "__main__":
    main()