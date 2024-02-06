from PIL import Image
import base64
from io import BytesIO

def imageToB64(img) -> str:
    buffered = BytesIO()
    pilImg = Image.fromarray(img)  
    pilImg.save(buffered, format="png")
    img_str = "data:image/png;base64," + str(base64.b64encode(buffered.getvalue()))