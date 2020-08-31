import os

from .camera import CameraStream
from .camerapi import CameraPiStream
from .image import ImageStream


def create_stream(identifier):
    if identifier.isnumeric():
        return CameraStream(int(identifier))
    elif identifier.startswith("pi"):
        return CameraPiStream(int(identifier[2:]))
    elif os.path.isfile(identifier):
        return ImageStream(identifier)
    else:
        raise ValueError("Unknown stream type: {}".format(identifier))
