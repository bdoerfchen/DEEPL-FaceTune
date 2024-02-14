import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence tensorflow

print("Started FaceTune")

import eel 
from datetime import datetime

import tensorflow as tf
from python.models.AEModel import AEModel
from python.models.IdentityAE import IdentityAE 
from python.models.AEGANModel import AEGANModel
from python.models.CAEModel import CAEModel
from python.models.vae.VariationalAE import VariationalAE
from python.ModelResult import ModelResult
from python.ftutilities import *
tf.keras.utils.disable_interactive_logging()  # Silence keras

print("Imported modules")

models : list[AEModel] = [
    AEGANModel(), 
    CAEModel((256, 256)), 
    VariationalAE() if VariationalAE.isModelAvailable() else IdentityAE(), # Only load VAE if the model file is available
    IdentityAE()
]
def main():
    # Init eel
    eel.init("./src/web")          
    print("Initialized webserver")

    # Load and init models
    print("Initializing models")
    for model in models:
        model.load()
        print("\tInitialized", model.getName())
    print("All models initialized")

    # Start the index.html file 
    print("Starting server on :8080")
    eel.start("index.html", mode=None, host='0.0.0.0', port=8080)



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

@eel.expose
def logConnection():
    print("\t", datetime.now().isoformat() + ": New connection established")

if __name__ == "__main__":
    main()

print("Stopping application")