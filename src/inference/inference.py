from images.image import Image

from typing import Sequence, Dict


class Inference():
    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        """The process method takes an ordered list of images, and produces a dict with named output images."""
        raise NotImplementedError()
