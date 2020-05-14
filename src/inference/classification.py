from dnn.mxnet import MxnetLoader, ModelType
from images.image import Image
from inference.inference import Inference

from typing import Sequence, Dict


class Classification(Inference):
    """Classification, using neural nets"""
    ARGUMENTS = {
        'model': str
    }

    def __init__(self, model=None):
        self.LOADER = MxnetLoader()
        self.LOADER.load(model or "resnet", ModelType.CLASSIFICATION)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.LOADER.process(image)
                visualised = self.LOADER.visualise(image, metadata)

                outputs[str(i)] = visualised

        return outputs
