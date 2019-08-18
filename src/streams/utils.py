import os

from .camera import CameraStream
from .image import ImageStream


def create_stream(identifier):
    if identifier.isnumeric():
        return CameraStream(int(identifier))
    elif os.path.isfile(identifier):
        return ImageStream(identifier)
    else:
        raise ValueError("Unknown stream type: {}".format(identifier))
