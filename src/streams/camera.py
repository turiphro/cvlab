import threading
from queue import Queue, Empty

from .stream import Stream
from images.image import Image

import cv2


class CameraStream(Stream):
    """
    Stream of images from a camera.

    The images are read asynchronously, and only the latest image will be returned.
    """
    def __init__(self, identifier: int):
        super().__init__(identifier)

        self.camera_stream = cv2.VideoCapture(identifier)
        self.image_queue = Queue()

        self.thread = threading.Thread(target=self._image_reader)
        self.thread.daemon = True
        self.thread.start()

    def _image_reader(self):
        """Asynchronous reading of images from the camera"""
        while True:
            ret, frame = self.camera_stream.read()
            if not ret:
                break

            # Empty queue
            if not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except Empty:
                    pass

            # Add new image
            self.image_queue.put(frame)

    def get(self, latest=False):
        # Wait for next image
        frame = self.image_queue.get()

        return Image(frame, opencv=True)

    def stop(self):
        self.camera_stream.release()
