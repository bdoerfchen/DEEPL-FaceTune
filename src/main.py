import eel 
from python.AEModel import AEModel 
from python.ModelResult import ModelResult

eel.init("./src/web")  
    

models : list[AEModel] = []

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

    return models[modelIndex].decode(encoding)


    



# Start the index.html file 
eel.start("index.html", mode=None)