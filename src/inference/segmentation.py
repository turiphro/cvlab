from dnn.mxnet import MxnetLoader, ModelType
from images.image import Image
from inference.inference import Inference

from typing import Sequence, Dict


class Segmentation(Inference):
    """Segmentation, using neural nets"""
    KEYSTROKES = {
        '-': "Decrease segmentation mask visibility (more original image)",
        '+': "Increase segmentation mask visibility",
        'l': "Show/hide class labels"
    }
    ARGUMENTS = {
        'model': str
    }

    def __init__(self, model=None):
        self.blend = 0.75
        self.show_labels = True
        self.LOADER = MxnetLoader()
        self.LOADER.load(model or "deeplab", ModelType.SEGMENTATION)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                metadata = self.LOADER.process(image)
                visualised = self.LOADER.visualise(image, metadata, blend=self.blend, show_labels=self.show_labels)

                outputs[str(i)] = visualised

        return outputs

    def handle_command(self, key):
        if key == '-':
            self.blend = max(0.0, self.blend - 0.25)
        elif key == '+':
            self.blend = min(1.0, self.blend + 0.25)
        elif key == 'l':
            self.show_labels = not self.show_labels
            print("Showing" if self.show_labels else "Hiding", "labels")
