from PIL import Image
import numpy as np
import base64
from io import BytesIO

def b64ToImage(b64: str):
    b64_clean = b64.replace("data:image/png;base64,", "")
    image_bytes = BytesIO(base64.b64decode(b64_clean))
    image_pil = Image.open(image_bytes)
    image_pil = image_pil.convert('RGB')
    image_np = np.array(image_pil)
    return image_np

def imageToB64(img) -> str:
    buffered = BytesIO()
    pilImg = Image.fromarray(img)
    pilImg.save(buffered, format="png")
    img_str = "data:image/png;base64," + str(base64.b64encode(buffered.getvalue()))[2:]
    return img_str