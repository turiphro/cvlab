from images.image import Image
from images.image_type import ImageType
from .inference import Inference

from typing import Sequence, Dict
import cv2


class Nothing(Inference):
    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        return {str(i): img for (i, img) in enumerate(images)}


class Edges(Inference):
    thr1 = 60
    thr2 = 120

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            img = image.get(ImageType.OPENCV)
            img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_bil = cv2.bilateralFilter(img_bw, 7, 50, 50)
            img_out = cv2.Canny(img_bil, self.thr1, self.thr2)

            outputs[str(i)] = Image(img_out, opencv=True)

        return outputs
