# This module requires the pycamera module

import threading
from queue import Queue, Empty

from .stream import Stream
from images.image import Image

import cv2
try:
    from picamera import PiCamera
    from picamera.array import PiRGBArray
except ModuleNotFoundError:
    pass # silently ignore (as it's always loaded by utils)


class CameraPiStream(Stream):
    """
    Stream of images from a camera.

    The images are read asynchronously, and only the latest image will be returned.
    """
    def __init__(self, identifier: int):
        super().__init__(identifier)

        self.camera_stream = PiCamera()
        self.camera_stream.framerate = 32
        self.image_queue = Queue()

        self.thread = threading.Thread(target=self._image_reader)
        self.thread.daemon = True
        self.thread.start()

    def _image_reader(self):
        """Asynchronous reading of images from the camera"""
        raw_capture = PiRGBArray(self.camera_stream)
        for frame in self.camera_stream.capture_continuous(
                raw_capture, format='rgb', use_video_port=True):
            img = frame.array
            img = cv2.flip(img, -1) # flip both axes

            # Empty queue
            if not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except Empty:
                    pass

            # Add new image
            self.image_queue.put(img)
            raw_capture.truncate(0)

    def get(self, latest=False):
        # Wait for next image
        frame = self.image_queue.get()

        return Image(frame, opencv=False)

    def stop(self):
        pass
