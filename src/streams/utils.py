import os

from .camera import CameraStream
from .camerapi import CameraPiStream
from .esp32 import Esp32
from .image import ImageStream


def create_stream(identifier):
    if identifier.isnumeric():
        return CameraStream(int(identifier))
    elif identifier.startswith("pi"):
        return CameraPiStream(int(identifier[2:]))
    elif identifier.startswith("esp"):
        return Esp32(identifier[3:])
    elif os.path.isfile(identifier):
        return ImageStream(identifier)
    else:
        raise ValueError("Unknown stream type: {}".format(identifier))
