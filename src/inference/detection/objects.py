from dnn.mxnet import MxnetLoader, ModelType
from cloud.aws import AWSInference, InferenceType
from images.image import Image
from ..inference import Inference

from typing import Sequence, Dict


class Objects(Inference):
    """Object detection, using neural nets"""
    ARGUMENTS = {
        'model': str
    }

    def __init__(self, model=None):
        self.LOADER = MxnetLoader()
        self.LOADER.load(model or "yolo", ModelType.DETECTION)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.LOADER.process(image)
                visualised = self.LOADER.visualise(image, metadata)

                outputs[str(i)] = visualised

        return outputs


class CloudObjects(Inference):
    """Object detection, using a Cloud provider"""
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

        self.PROVIDER.load(InferenceType.DETECTION)
        if self.PROVIDER.KEYSTROKES:
            self.KEYSTROKES.update(self.PROVIDER.KEYSTROKES)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.PROVIDER.process(image)
                visualised = self.PROVIDER.visualise(image, metadata)

                outputs[str(i)] = visualised

        return outputs

    def handle_command(self, key):
        self.PROVIDER.handle_command(key)


# TODO add class for AWS[Rekognition]Objects