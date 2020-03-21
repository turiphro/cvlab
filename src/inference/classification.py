from dnn.mxnet import MxnetLoader, ModelType
from images.image import Image
from inference.inference import Inference

from typing import Sequence, Dict


class Classification(Inference):
    """Classification, using neural nets"""

    def __init__(self, model_name="resnet"):
        self.LOADER = MxnetLoader()
        self.LOADER.load(model_name, ModelType.CLASSIFICATION)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.LOADER.process(image)
                visualised = self.LOADER.visualise(image, metadata)

                outputs[str(i)] = visualised

        return outputs
