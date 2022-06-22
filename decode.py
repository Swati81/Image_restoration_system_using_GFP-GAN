import pybase64
import numpy as np
import cv2

def Decode(image):
    imgdata = pybase64.b64decode(image)
    img = np.asarray(bytearray(imgdata), dtype="uint8")
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img