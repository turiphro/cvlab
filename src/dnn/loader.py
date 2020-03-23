from enum import Enum
from typing import Dict, Any

from images.image import Image


class ModelLoader:
    """
    Interface for loaders for (deep) pre-trained neural network models

    Loaders should hide the specifics of the neural network framework;
    however, since pretrained models vary widely, the specific return
    values might differ for different frameworks and models, and need
    to be handled inside the consumer classes.
    """
    def load(self, model_name) -> None:
        """
        Load a neural network model, given a symbolic name

        Models will be cached where possible.
        """
        raise NotImplementedError()

    def process(self, image: Image) -> Dict[str, Any]:
        """
        Run inference on an Image, returning model-specific results

        The results may be used in client-specific logic, or fed into
        .annotate() for image annotation.
        """
        raise NotImplementedError()

    def visualise(self, image: Image, metadata: Dict[str, Any]) -> Image:
        """
        Annotate the image given the result of process()

        The annotation type may differ depending on the model. Examples:
        - object detection -> bounding boxes
        - segmentation -> pixel-wise categories
        - captioning -> text on image
        """
        raise NotImplementedError()


class ModelType(Enum):
    CLASSIFICATION  = 0
    DETECTION       = 1
    SEGMENTATION    = 2

