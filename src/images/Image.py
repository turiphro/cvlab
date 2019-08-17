from enum import Enum
import numpy as np
import PIL
import cv2

from . import convert
from .ImageType import ImageType


# Image abstraction supporting multiple image formats.
#
# This class is useful as an interface between different classes;
# within a class you can use the underlying data format directly.
class Image:
    orig_type = ImageType.UNSET

    img = {
        ImageType.NUMPY: None,
        ImageType.OPENCV: None,
        ImageType.PILLOW: None,
    }

    def __init__(self, img, copy=True, opencv=False):
        """
        Create an Image instance. Image instances are immutable, will be copied when
        instantiated, and will be cached for any format that's being requested.

        img: image data (np.ndarray, PIL.ImageFile.ImageFile)
        copy: set to False for performance improvements if you can guarantee the original data won't be altered
        opencv: whether the image is an opencv image (e.g., numpy but in BGR format)
        """
        self.orig_type = convert.get_type(img, opencv=opencv)

        if self.orig_type == ImageType.OPENCV:
            self.img[ImageType.OPENCV] = img.copy() if copy else img
        elif self.orig_type == ImageType.NUMPY:
            self.img[ImageType.NUMPY] = img.copy() if copy else img
        elif self.orig_type == ImageType.PILLOW:
            self.img[ImageType.PILLOW] = img.copy() if copy else img
        else:
            raise ValueError("Cannot handle this image data format: {}".format(
                self.orig_type))

    def get(self, target_type: ImageType):
        """Get the image data as target_type:ImageType"""
        if target_type not in self.img:
            raise ValueError("Unsupported image type: {}".format(target_type))

        if self.img[target_type] is None:
            # Assuming we have a conversion from self.orig_type to target_type
            img_conv = convert.convert(
                self.img[self.orig_type], self.orig_type, target_type)
            # Cache result
            self.img[target_type] = img_conv

        return self.img[target_type]

    def asnumpy(self):
        return self.get(ImageType.NUMPY)

    def asopencv(self):
        return self.get(ImageType.OPENCV)

    def aspil(self):
        return self.get(ImageType.PILLOW)

