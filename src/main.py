import eel 
from python.AEModel import AEModel 
from python.ModelResult import ModelResult

models : list[AEModel] = []
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
    result : list[ModelResult] = []

    for model in models:
        result.append(
            ModelResult(model, baseImage)
        )
    return result

@eel.expose
def decodeLatentEncoding(modelIndex, encoding):
    assert int(modelIndex) == modelIndex and modelIndex >= 0
    assert isinstance(encoding, list)

    resultImage = models[modelIndex].decode(encoding)
    #image to datastring
    b64string = ""
    return b64string


if __name__ == "__main__":
    main()