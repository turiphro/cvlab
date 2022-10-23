from cloud.aws import AWSInference, InferenceType
from images.image import Image
from images.image_type import ImageType
from ..inference import Inference

from typing import Sequence, Dict
import os
import cv2

CASCADE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(cv2.__file__)), "data")


class CloudText(Inference):
    """Text detection, using a Cloud provider"""
    ARGUMENTS = {
        'cloud': str
    }
    KEYSTROKES = {}


    def __init__(self, cloud=None):
        cloud = cloud or "aws"

        if cloud == "aws":
            self.PROVIDER = AWSInference()
        else:
            raise ValueError(f"Unknown cloud provider: {cloud}")

        self.PROVIDER.load(InferenceType.TEXT_EXTRACT)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.PROVIDER.process(image)
                visualised = self.PROVIDER.visualise(image, metadata)

                outputs[str(i)] = visualised

        return outputs
