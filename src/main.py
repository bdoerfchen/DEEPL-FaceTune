import eel 
from python.models.AEModel import AEModel
from python.models.IdentityAE import IdentityAE 
from python.models.AEGANModel import AEGANModel
from python.models.CAEModel import CAEModel
from python.models.vae.VariationalAE import VariationalAE
from python.ModelResult import ModelResult
from python.ftutilities import *

models : list[AEModel] = [
    AEGANModel(), CAEModel((256, 256)), IdentityAE(), IdentityAE()
]
def main():
    # Init eel
    eel.init("./src/web")          

    # Load and init models
    for model in models:
        model.load()

    # Start the index.html file 
    print("Starting server")
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
def decodeLatentEncoding(index, encoding):
    model = models[index]

    encoding = np.array(encoding).astype(np.float32)
    #encoding = list(map(lambda x: float(x), encoding))
    resultImage = model.decode(encoding)
    b64string = imageToB64(resultImage)
    return b64string

@eel.expose
def getEncoding(index, imgB64):
    model = models[index]
    img = b64ToImage(imgB64)
    latent = np.array(model.encode(img)).tolist()
    return latent


if __name__ == "__main__":
    main()