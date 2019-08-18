import cv2

from .Image import Image
from .ImageType import ImageType

# Various functions for loading and saving images and streams.


def read(filename: str, image_type: ImageType = ImageType.OPENCV) -> Image:
    if image_type == ImageType.OPENCV:
        return cv2.imread(filename)
    else:
        raise ValueError("Unsupported format for reading: {}".format(image_type))


