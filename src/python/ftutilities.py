## Authors: Benjamin Bissendorf and Niklas Seeliger

from PIL import Image
import numpy as np
import base64
from io import BytesIO

def b64ToImage(b64: str):
    """Converts a base64 data url image to a np rgb image array"""

    # Remove heading for png / jpg
    b64_clean = b64.replace("data:image/png;base64,", "").replace("data:image/jpeg;base64,", "") #Better: regex
    image_bytes = BytesIO(base64.b64decode(b64_clean)) # Load into memory
    image_pil = Image.open(image_bytes) # Open image
    image_pil = image_pil.convert('RGB') # Convert to rgb
    image_np = np.array(image_pil) # Load as array
    return image_np

def imageToB64(img) -> str:
    """Calculate the base64 string from an image"""

    buffered = BytesIO() # Init memory buffer
    pilImg = Image.fromarray(img) # Load image from array
    pilImg.save(buffered, format="png") # Save loaded image to buffer in png format
    img_str = "data:image/png;base64," + str(base64.b64encode(buffered.getvalue()))[2:-1] # Concat to base64 data url
    return img_str