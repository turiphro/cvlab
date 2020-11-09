import threading
import time
import cv2
import numpy as np
import requests
from queue import Queue, Empty

from .stream import Stream
from images.image import Image


class Esp32(Stream):
    """
    Network-based stream from an ESP32-CAM, flashed with the CameraWebServer Arduino sketch

    The images are read asynchronously from the network, and only the latest image will be returned.
    """

    # ESP32-CAM server endpoints:
    # /         serves web page
    # /capture  sends JPG capture
    # /stream   sends continuous JPG part stream
    # /status   status info (json): framesize, quality, vflip/hmirror, face_detect
    # /control  change settings (GET param: ?var=framesize&val=10)
    #           framesize: 0 - 10; capture takes ~40ms - 300-500ms

    def __init__(self, identifier: str):
        super().__init__(identifier)

        self.camera_ip = identifier
        # TODO (sep function, also call during recovery): check status, set settings, save IP
        # resolution: /control?var=framesize&val=7
        self.image_queue = Queue()

        self.thread = threading.Thread(target=self._image_reader)
        self.thread.daemon = True
        self.thread.start()

    def _image_reader(self):
        """Asynchronous reading of images from the camera"""
        while True:
            # TODO check if we can read from $IP/stream (continuous)
            url = "http://{}/capture".format(self.camera_ip)
            response = None
            try:
                response = requests.get(url, timeout=2)
            except requests.exceptions.ConnectTimeout \
            or requests.exceptions.ConnectionError as ex:
                print("[!!] timeout reading from stream", url)
                print(ex)
                time.sleep(5)
                continue

            if response.status_code != 200:
                print("[!!] can't read from stream", url)
                print(response.content)
                time.sleep(5)
                continue

            img_np = np.array(bytearray(response.content), dtype=np.uint8)
            img = cv2.imdecode(img_np, -1)

            # Empty queue
            if not self.image_queue.empty():
                try:
                    self.image_queue.get_nowait()
                except Empty:
                    pass

            # Add new image
            self.image_queue.put(img)

            # don't overheat the ESP32
            time.sleep(0.1)

    def get(self, latest=False):
        # Wait for next image
        img = self.image_queue.get()

        return Image(img, opencv=True)

    def stop(self):
        pass

