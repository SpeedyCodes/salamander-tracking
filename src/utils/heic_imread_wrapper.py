import cv2 as cv
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener


register_heif_opener()

def wrapped_imread(*args) -> np.ndarray:
    """
    wrapper for imread to also support HEIC images
    """
    # try to read the image with cv.imread
    img = cv.imread(*args)
    if img is not None: # if the image was read successfully
        return img

    # if the image was not read successfully, try to read it with PIL
    PIL_img = Image.open(args[0])
    open_cv_image = np.array(PIL_img)

    # apply any flags that were passed to the function
    # expand this if more flags are needed
    imread_mode = args[1] if len(args) > 1 else cv.IMREAD_COLOR
    if imread_mode == cv.IMREAD_COLOR:
        open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2BGR)
    elif imread_mode == cv.IMREAD_GRAYSCALE:
        open_cv_image = cv.cvtColor(open_cv_image, cv.COLOR_RGB2GRAY)
    else:
        raise NotImplementedError("this imread_mode is not supported yet")

    if len(args) > 2:
        raise NotImplementedError("this function does not support more than 2 arguments yet")

    return open_cv_image
