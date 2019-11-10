import numpy as np
import cv2
from PIL.Image import Image as PILImage

from .image_type import ImageType

# Various conversion functions


def get_type(img, opencv=False):
    img_type = ImageType.UNSET

    if isinstance(img, np.ndarray) and opencv:
        img_type = ImageType.OPENCV
    elif isinstance(img, np.ndarray):
        img_type = ImageType.NUMPY
    elif isinstance(img, PILImage):
        img_type = ImageType.PILLOW
    else:
        raise ValueError("Unknown type: {}".format(type(img)))

    return img_type


# Conversion functions between pairs of image types
CONVERSIONS = {
    (ImageType.PILLOW, ImageType.NUMPY): np.array,
    (ImageType.NUMPY, ImageType.OPENCV):
        (lambda img: cv2.cvtColor(img, cv2.COLOR_RGB2BGR)),
    (ImageType.OPENCV, ImageType.NUMPY):
        (lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
}


def convert(img, source_type, target_type, debug=False):
    """Convert directly from source_type to target_type."""
    direct_conv = (source_type, target_type)

    # try direct conversion
    if direct_conv in CONVERSIONS:
        # direct conversion available
        if debug:
            print("CONVERTING DIRECTLY {} -> {}".format(source_type, target_type))
        return CONVERSIONS[direct_conv](img)

    # try with one hop
    possible_types = set(ImageType) - set([ImageType.UNSET, source_type, target_type])

    for intermediate_type in possible_types:
        conv1 = (source_type, intermediate_type)
        conv2 = (intermediate_type, target_type)

        if conv1 in CONVERSIONS and conv2 in CONVERSIONS:
            if debug:
                print("CONVERTING VIA {} -> {} -> {}".format(
                    source_type, intermediate_type, target_type))
            intermediate_img = CONVERSIONS[conv1](img)
            target_img       = CONVERSIONS[conv2](intermediate_img)
            return target_img

    # if no path was found, throw an exception
    raise ValueError("Cannot convert from {} to {} with one or two steps".format(
        source_type, target_type))
