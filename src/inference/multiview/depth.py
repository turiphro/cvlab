from collections import defaultdict

from ..inference import Inference
from . import utils as utils
from images.image import Image
from images.image_type import ImageType

import time
import cv2
import numpy as np
from pprint import pprint
from typing import Sequence, Dict


class StereoVision(Inference):
    KEYSTROKES = {
        'c': "Calibrate cameras",
        ' ': "Start taking snapshots (during calibration)"
    }
    STAGES = {"CALIBRATE_WAIT": 1, "CALIBRATING": 2, "RUNNING": 3}
    CHESSBOARD_SIZE = (10, 7)   # number of corners inside the chessboard pattern
    SQUARE_SIZE = 2.47          # real world size of chessboard square size (in cm)
    SUBPIX_PARAMS = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self):
        self.reset_calibration()

    def reset_calibration(self):
        self.stage = StereoVision.STAGES["CALIBRATE_WAIT"]
        # collecting calibration images
        self.last_snapshot = 0
        self.snapshot_count = 0
        self.snapshots = defaultdict(lambda: [])
        self.corners = defaultdict(lambda: [])
        # final calibration parameters
        self.camparams = {}

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        # Inspired by:
        # - https://docs.opencv.org/master/d9/db7/tutorial_py_table_of_contents_calib3d.html
        # - https://albertarmea.com/post/opencv-stereo-camera/
        if len(images) != 2:
            raise Exception("Stereo vision requires 2 images, not " + str(len(images)))

        if self.stage in [StereoVision.STAGES["CALIBRATE_WAIT"], StereoVision.STAGES["CALIBRATING"]]:
            return self.process_calibration(images)

        else:
            return self.process_stereo(images)

        return {}

    def process_calibration(self, images: Sequence[Image]):
        """
        Calibrate 2+ images: find intrinsic + extrinsic matrices and R/T between the cameras
        """
        outputs = {str(i): image for (i, image) in enumerate(images)}

        # step 1: press space to start
        if self.stage == StereoVision.STAGES["CALIBRATE_WAIT"]:
            print("[calibration] Grab your chessboard pattern, and press space to start taking snapshots")

        # step 2: take snapshots every 3s
        elif self.snapshot_count < 8:

            # step 2a: waiting for next snapshot
            if time.time() - self.last_snapshot < 3.000:
                # waiting for next snapshot moment
                outputs = {str(i): image for (i, image) in enumerate(images)}

            # step 2b: take snapshot (if delay passed and the pattern was found for all images)
            else:
                all_corners  = {}
                for i, image in enumerate(images):
                    key = str(i)
                    img = image.get(ImageType.OPENCV)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    #ret, corners = cv2.findChessboardCorners(
                    #    img_gray, StereoVision.CHESSBOARD_SIZE, None)
                    ret, corners = cv2.findChessboardCornersSB(
                        img_gray, StereoVision.CHESSBOARD_SIZE, None)
                    if ret:
                        corners_subpix = cv2.cornerSubPix(
                            img_gray, corners, (11, 11), (-1, -1), StereoVision.SUBPIX_PARAMS)
                        # TODO calibrate and save
                        all_corners[key] = corners

                        img = cv2.drawChessboardCorners(
                            img, StereoVision.CHESSBOARD_SIZE, corners_subpix, ret)
                        img = cv2.rectangle(img, (1, 1), (img.shape[1]-2, img.shape[0]-2), (0, 255, 0), 2)
                    else:
                        img = cv2.rectangle(img, (1, 1), (img.shape[1]-2, img.shape[0]-2), (0, 0, 255), 2)

                    outputs[key] = Image(img, opencv=True)

                if len(all_corners) == len(images):
                    # found corners in all images; saving
                    self.snapshot_count += 1
                    self.last_snapshot = time.time()

                    for key, corners in all_corners.items():
                        # TODO tmp
                        self.snapshot_count += 3
                        for _ in range(4):
                            self.corners[key].append(corners)
                            self.snapshots[key].append(img)

                    print("[calibration] Snapshot {}! Waiting 3s for the next snapshot..".format(
                        self.snapshot_count))

        # step 3: calculate calibration parameters
        elif self.snapshot_count >= 8:
            print("[calibration] Calculating calibration parameters")

            # calibrate individual cameras
            SIZEX, SIZEY = StereoVision.CHESSBOARD_SIZE
            world_points_frame = np.zeros((SIZEX*SIZEY, 3), np.float32)
            world_points_frame[:, :2] = np.mgrid[0:SIZEX, 0:SIZEY].T.reshape(-1, 2) * StereoVision.SQUARE_SIZE
            world_points = [world_points_frame] * 8

            for key in self.corners:
                img_points = self.corners[key]
                img_size = self.snapshots[key][0].shape[1::-1]

                camparams = utils.calibrate_camera(
                    world_points, img_points, img_size)
                self.camparams[key] = camparams

                pprint(camparams)

                self.stage = StereoVision.STAGES["RUNNING"]

            # calibrate camera pairs (just 1 pair for now)
            # TODO

        return outputs

    def process_stereo(self, images: Sequence[Image]):
        outputs = {}

        # return undistorted images
        for i, image in enumerate(images):
            key = str(i)
            img = image.get(ImageType.OPENCV)
            camparams = self.camparams[key]

            img_undistort = cv2.undistort(
                img, camparams["intrinsic"], camparams["distortion"],
                None, camparams["intrinsic_crop"])
            #x, y, w, h = camparams["intrinsic_roi"]
            #img_undistort = img_undistort[y:y+h, x:x+w]
            outputs[key] = Image(img_undistort, opencv=True)

        # return depth image(s)
        # TODO

        return outputs

    def handle_keystroke(self, key):
        if key == ' ' and self.stage == StereoVision.STAGES["CALIBRATE_WAIT"]:
            self.stage = StereoVision.STAGES["CALIBRATING"]
        elif key == 'c':
            self.reset_calibration()


class Stitching(Inference):
    """
    Stitching together camera streams with overlapping areas

    Expecting 2 camera streams (or images) by default; the second view
    will be the reference point (left unwarped image)
    """

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        # Based on:
        # https://www.pyimagesearch.com/2016/01/11/opencv-panorama-stitching/
        outputs = {}
        if len(images) != 2:
            raise Exception("Stitching requires 2 images, not " + str(len(images)))

        img1 = images[0].get(ImageType.OPENCV)
        img2 = images[1].get(ImageType.OPENCV)
        stitched, vis = self.stitch(img1, img2)

        if stitched is not None:
            outputs["stitched"] = Image(stitched, opencv=True)
        if vis is not None:
            outputs["vis"] = Image(stitched, opencv=True)
        return outputs

    def stitch(self, img1, img2, ratio=0.75, thresh=4.0):
        # get features
        points1, features1 = self.detect_and_describe(img1)
        points2, features2 = self.detect_and_describe(img2)
        if len(points1) == 0 or len(points2) == 0:
            return None, None

        # match features
        match = self.match_keypoints(
            points1, points2, features1, features2, ratio, thresh)
        if match is None:
            return None, None
        matches, H, status = match

        # stitch
        # TODO: properly find the bounding box and project both inside
        stitched = cv2.warpPerspective(
            img1, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
        stitched[0:img1.shape[0], 0:img2.shape[1]] = img2
        annotated = self.draw_matches(
            img1, img2, points1, points2, matches, status)

        return stitched, annotated

    def detect_and_describe(self, img):
        descriptor = cv2.xfeatures2d.SIFT_create()
        points, features = descriptor.detectAndCompute(img, None)
        points = np.float32([point.pt for point in points])
        return points, features

    def match_keypoints(self, points1, points2, features1, features2, ratio, thresh):
        """
        Match descriptors from two images

        Note: this only accounts for rotations between cameras, not
        for translations.
        """
        matcher = cv2.DescriptorMatcher_create("BruteForce")
        matches_raw = matcher.knnMatch(features1, features2, 2)

        matches = []
        for m in matches_raw:
            if len(m) == 2 and m[0].distance < m[1].distance * ratio:
                matches.append((m[0].trainIdx, m[0].queryIdx))

        if len(matches) < 4:
            print("Less than 4 matches - can't stitch!")
            return None

        match_points1 = np.float32([points1[i] for (_, i) in matches])
        match_points2 = np.float32([points2[i] for (i, _) in matches])
        # findHomography estimates the rotation between cameras (not T)
        H, status = cv2.findHomography(
            match_points1, match_points2, cv2.RANSAC, thresh)

        return matches, H, status

    def draw_matches(self, img1, img2, points1, points2, matches, status):
        """Draw adjacently and connect the point matches"""
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        vis = np.zeros((max(h1, h2), w1 + w2, 3), dtype="uint8")
        vis[0:h1, 0:w1] = img1
        vis[0:h2, w1:] = img2

        for ((trainIdx, queryIdx), s) in zip(matches, status):
            if s == 1:
                cv2.line(vis,
                         (int(points1[queryIdx][0]), int(points1[queryIdx][1])),
                         (int(points2[trainIdx][0]) + w1, int(points2[trainIdx][1])),
                         (0, 255, 0),
                         1)

        return vis
