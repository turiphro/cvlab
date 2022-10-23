from enum import Enum
from typing import Dict, Any

from images.image import Image


class InferenceType(Enum):
    CLASSIFICATION  = 0
    DETECTION       = 1
    FACE_DETECTION  = 2
    TEXT_EXTRACT    = 3


class CloudProvider:
    """
    Interface for cloud providers

    Instances should hide the specifics of the cloud provider or
    specific cloud service.
    """

    def load(self, inference_type: InferenceType) -> None:
        """
        Load a cloud provider client
        """
        raise NotImplementedError()

    def process(self, image: Image) -> Dict[str, Any]:
        """
        Run inference on an Image, returning model-specific results

        The results may be used in client-specific logic, or fed into
        .visualise() for image annotation.
        """
        raise NotImplementedError()

    def visualise(self, image: Image, metadata: Dict[str, Any]) -> Image:
        """
        Annotate the image given the result of process()

        The annotation type may differ depending on the cloud service. Examples:
        - object detection -> bounding boxes
        - text detection -> text on image
        """
        raise NotImplementedError()