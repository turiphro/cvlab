from .stream import Stream
from images.image import Image
from images.image_type import ImageType

import cv2


class ImageStream(Stream):
    def __init__(self, identifier: int):
        super().__init__(identifier)

        self.image = Image(cv2.imread(identifier), opencv=True)

    def get(self, latest=False):
        return self.image

    def stop(self):
        pass
