from enum import Enum
import os

IMG_FOLDER = "Pictures"

class ImageTypes(Enum):
    RAW = "raw"
    THRESH = "threshold"
    CONTOUR = "contour"
    EDGE = "edge"
    PERSPECTIVE = "perspective"
    SEGMENT = "segment"

class Models(Enum):
    THC100D = "thc100d"
    OVI = "ovi"

class TestTypes(Enum):
    LABEL = "label"
    INDICATOR = "indicator"

def LABEL_PATH(model): return os.path.join(IMG_FOLDER, model, TestTypes.LABEL.value)
def INDICATOR_PATH(model): return os.path.join(IMG_FOLDER, model, TestTypes.INDICATOR.value)