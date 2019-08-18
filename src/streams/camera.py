from .stream import Stream
from images.image import Image

import cv2


class CameraStream(Stream):
    def __init__(self, identifier: int):
        super().__init__(identifier)

        self.camera_stream = cv2.VideoCapture(identifier)

    def get(self, latest=False):
        ret, frame = self.camera_stream.read()

        return Image(frame, opencv=True) if ret else False

    def stop(self):
        self.camera_stream.release()
