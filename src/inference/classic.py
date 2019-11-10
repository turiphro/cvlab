from .inference import Inference
from images.image import Image
from images.image_type import ImageType

import cv2
import numpy as np
from typing import Sequence, Dict


class OpticalFlow(Inference):
    KEYSTROKES = {'r': "Reset tracking"}
    FEATURE_PARAMS = {
        "maxCorners": 100,
        "qualityLevel": 0.3,
        "minDistance": 7,
        "blockSize": 7
    }
    LK_PARAMS = {
        "winSize": (15, 15),
        "maxLevel": 2,
        "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }
    COLOURS = np.random.randint(0, 255, (100, 3))

    def __init__(self):
        self.reset_tracking()

    def reset_tracking(self):
        self.img_prev = {}      # previous image to compare with
        self.points_prev = {}   # features to track (since reset, per image)
        self.mask = {}          # mask to keep drawing on

    def process(self, images: Sequence[Image], count=[0]) -> Dict[str, Image]:
        # Based on:
        # https://docs.opencv.org/3.4/d4/dee/tutorial_optical_flow.html
        outputs = {}

        for (i, image) in enumerate(images):
            key = str(i)
            if image is not None:

                img = image.get(ImageType.OPENCV)
                img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # on first image since reset, init
                if key not in self.img_prev:
                    self.img_prev[key] = img_grey
                    self.mask[key] = np.zeros_like(img)
                    self.points_prev[key] = cv2.goodFeaturesToTrack(
                        img_grey, mask=None, **OpticalFlow.FEATURE_PARAMS)

                # track on all future frames
                else:
                    points, status, err = cv2.calcOpticalFlowPyrLK(
                        self.img_prev[key], img_grey, self.points_prev[key], None, **OpticalFlow.LK_PARAMS)

                    if points is not None:

                        points_good_curr = points[status == 1]
                        points_good_prev = self.points_prev[key][status == 1]

                        # draw
                        for j, (curr, prev) in enumerate(zip(points_good_curr, points_good_prev)):
                            a, b = curr.ravel()
                            c, d = prev.ravel()
                            self.mask[key] = cv2.line(
                                self.mask[key], (a, b), (c, d), OpticalFlow.COLOURS[j % 100].tolist(), 2)
                            img = cv2.circle(img, (a, b), 5, OpticalFlow.COLOURS[j % 100].tolist(), -1)
                        img = cv2.add(img, self.mask[key])

                        # update for next frame
                        self.points_prev[key] = points_good_curr.reshape(-1, 1, 2)

                    self.img_prev[key] = img_grey

                outputs[key] = Image(img, opencv=True)

        return outputs

    def handle_keystroke(self, key):
        if key == 'r':
            print("Resetting tracking")
            self.reset_tracking()
