from images.image import Image
from images.image_type import ImageType
from ..inference import Inference

from typing import Sequence, Dict
import os
import cv2

CASCADE_FOLDER = os.path.join(os.path.dirname(os.path.abspath(cv2.__file__)), "data")


class Faces(Inference):
    """Classic face detection, using Haar Cascades"""

    FACE_CASCADE = cv2.CascadeClassifier(CASCADE_FOLDER + os.sep + 'haarcascade_frontalface_default.xml')
    BOX_COLOUR = (255, 0, 0)

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        outputs = {}
        for (i, image) in enumerate(images):
            if image is not None:
                img = image.get(ImageType.OPENCV)
                img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # detect
                faces = Faces.FACE_CASCADE.detectMultiScale(img_gray, 1.3, 5)

                # draw
                for x, y, w, h in faces:
                    img = cv2.rectangle(img, (x, y), (x+w, y+h), Faces.BOX_COLOUR, 2)

                outputs[str(i)] = Image(img, opencv=True)

        return outputs
