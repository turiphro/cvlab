from collections import defaultdict

from ..inference import Inference
from . import utils as utils
from images.image import Image
from images.image_type import ImageType

import time
import os
import cv2
import numpy as np
import pickle
from pprint import pprint
from typing import Sequence, Dict


class StereoVision(Inference):
    """
    Depth maps from camera pairs.

    Be sure to:
    - list the cameras from left to right (from the cameras' perspective)
    - when re-running with cached calibration parameters, have the same order as when calibrated
      (device indices may change after a reboot)
    """

    KEYSTROKES = {
        'c': "Calibrate cameras",
        ' ': "Start taking snapshots (during calibration)",
        'a': "Switch algorithm",
    }
    CACHE = os.path.expanduser("~/.cvlab/")
    STAGES = {"CALIBRATE_WAIT": 1, "CALIBRATING": 2, "RUNNING": 3}
    CHESSBOARD_SIZE = (10, 7)   # number of corners inside the chessboard pattern
    SQUARE_SIZE = 2.47          # real world size of chessboard square size (in cm)
    SUBPIX_PARAMS = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    def __init__(self):
        self.sgbm = True
        self.reset_calibration()
        loaded = self.load_params()
        if loaded:
            self.stage = StereoVision.STAGES["RUNNING"]
            print("Loading calibration data from cache. Touch 'c' to re-calibrate.")

    def reset_calibration(self):
        self.stage = StereoVision.STAGES["CALIBRATE_WAIT"]
        # collecting calibration images
        self.last_snapshot = 0
        self.snapshot_count = 0
        self.snapshots = defaultdict(lambda: [])
        self.corners = defaultdict(lambda: [])
        # final calibration parameters
        self.camparams = {}
        self.pairparams = {}

        self.stereo_matcher = None

    def process(self, images: Sequence[Image]) -> Dict[str, Image]:
        # Inspired by:
        # - https://docs.opencv.org/master/d9/db7/tutorial_py_table_of_contents_calib3d.html
        # - https://albertarmea.com/post/opencv-stereo-camera/
        if len(images) < 2:
            raise Exception("Stereo vision requires at least 2 images, but " + str(len(images)) + " were given")

        if self.stage in [StereoVision.STAGES["CALIBRATE_WAIT"], StereoVision.STAGES["CALIBRATING"]]:
            return self.process_calibration(images)

        else:
            return self.process_stereo(images)

        return {}

    def process_calibration(self, images: Sequence[Image]):
        """
        Calibrate 2+ images: find intrinsic + extrinsic matrices and R/T between the camera pairs
        """
        outputs = {str(i): image for (i, image) in enumerate(images)}

        # step 1: press space to start
        if self.stage == StereoVision.STAGES["CALIBRATE_WAIT"]:
            print("[calibration] Grab your chessboard pattern, and press space to start taking snapshots.")
            print("              The chessboard needs to be fully visible in *all* images.")

        # step 2: take snapshots every 3s
        elif self.snapshot_count < 8:

            # step 2a: waiting for next snapshot
            if time.time() - self.last_snapshot < 3.000:
                # waiting for next snapshot moment
                outputs = {str(i): image for (i, image) in enumerate(images)}

            # step 2b: take snapshot (if delay passed and the pattern was found for all images)
            else:
                all_corners  = {}
                all_snapshots = {}
                for i, image in enumerate(images):
                    key = str(i)
                    img = image.get(ImageType.OPENCV)
                    print(i, img.shape)
                    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    ret, corners = cv2.findChessboardCornersSB(
                        img_gray, StereoVision.CHESSBOARD_SIZE, None)
                    if ret:
                        all_corners[key] = corners
                        all_snapshots[key] = img

                        img = cv2.drawChessboardCorners(
                            img, StereoVision.CHESSBOARD_SIZE, corners, ret)
                        img = cv2.rectangle(img, (1, 1), (img.shape[1]-2, img.shape[0]-2), (0, 255, 0), 2)
                    else:
                        img = cv2.rectangle(img, (1, 1), (img.shape[1]-2, img.shape[0]-2), (0, 0, 255), 2)

                    outputs[key] = Image(img, opencv=True)

                if len(all_corners) == len(images):
                    # found corners in all images; saving
                    self.snapshot_count += 1
                    self.last_snapshot = time.time()

                    for key in all_corners:
                        self.corners[key].append(all_corners[key])
                        self.snapshots[key].append(all_snapshots[key])

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

            # calibrate camera pairs (pairwise, chained in given order)
            for index_l, index_r in zip(range(0, len(images)-1), range(1, len(images))):
                key_l, key_r = str(index_l), str(index_r)

                camparams_img_l = self.camparams[key_l]
                camparams_img_r = self.camparams[key_r]
                img_size = self.snapshots[key_l][0].shape[1::-1]
                img_size_r = self.snapshots[key_r][0].shape[1::-1]
                if img_size != img_size_r:
                    print("ERROR: images should be of the same size; current sizes:", img_size, img_size_r)

                pairparams = utils.calibrate_camera_pair(
                    world_points, self.corners[key_l], self.corners[key_r],
                    camparams_img_l, camparams_img_r, img_size)
                self.pairparams[(key_l, key_r)] = pairparams

                print((key_l, key_r))
                pprint(pairparams)

            # save to disk
            self.save_params()

            self.stage = StereoVision.STAGES["RUNNING"]

        return outputs

    def process_stereo(self, images: Sequence[Image]):
        outputs = {}

        if len(self.camparams.keys()) != len(images):
            print("Error: number of images doesn't match the number of saved parameters.")
            print("Recalibrate (touch 'c') if old parameters were loaded from the cache.")
            return {str(i): image for (i, image) in enumerate(images)}

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
        if not self.stereo_matcher:
            if self.sgbm:
                self.stereo_matcher = cv2.StereoSGBM_create(
                    minDisparity=5, numDisparities=64, blockSize=5,
                    speckleRange=5, speckleWindowSize=15)
            else:
                self.stereo_matcher = cv2.StereoBM_create(
                    numDisparities=64, blockSize=5)
                self.stereo_matcher.setMinDisparity(5)
                self.stereo_matcher.setSpeckleRange(9)
                self.stereo_matcher.setSpeckleWindowSize(21)

        # process camera pairs (pairwise, chained in given order)
        for index_l, index_r in zip(range(0, len(images)-1), range(1, len(images))):
            key_l, key_r = str(index_l), str(index_r)
            pairkey = "{}|{}".format(key_l, key_r)

            img_l = outputs[key_l].get(ImageType.OPENCV)
            img_r = outputs[key_r].get(ImageType.OPENCV)
            pairparams = self.pairparams[(key_l, key_r)]

            img_l_rect = cv2.remap(
                img_l, pairparams["map_x"][0], pairparams["map_y"][0], cv2.INTER_LINEAR)
            img_r_rect = cv2.remap(
                img_r, pairparams["map_x"][1], pairparams["map_y"][1], cv2.INTER_LINEAR)

            img_l_gray = cv2.cvtColor(img_l_rect, cv2.COLOR_BGR2GRAY)
            img_r_gray = cv2.cvtColor(img_r_rect, cv2.COLOR_BGR2GRAY)
            outputs[key_l + "_rect_" + pairkey] = Image(img_l_rect, opencv=True)
            outputs[key_r + "_rect_" + pairkey] = Image(img_r_rect, opencv=True)

            img_depth = self.stereo_matcher.compute(img_l_gray, img_r_gray)

            img_depth_8bit = (img_depth.astype(np.float64) / img_depth.max() * 255).astype(np.uint8)
            outputs["depth_" + pairkey] = Image(img_depth_8bit, opencv=True)

            img_depth_colour = cv2.applyColorMap(img_depth_8bit, cv2.COLORMAP_JET) # _AUTUMN/_JET
            outputs["depth_" + pairkey + "_colour"] = Image(img_depth_colour, opencv=True)

        return outputs

    def save_params(self):
        filename_cams = os.path.join(StereoVision.CACHE, "camparams.pkl")
        filename_pairs = os.path.join(StereoVision.CACHE, "pairparams.pkl")
        os.makedirs(StereoVision.CACHE, exist_ok=True)
        with open(filename_cams, "wb") as fp:
            pickle.dump(self.camparams, fp)
        with open(filename_pairs, "wb") as fp:
            pickle.dump(self.pairparams, fp)

    def load_params(self):
        filename_cams = os.path.join(StereoVision.CACHE, "camparams.pkl")
        filename_pairs = os.path.join(StereoVision.CACHE, "pairparams.pkl")
        if not (os.path.exists(filename_cams) and os.path.exists(filename_pairs)):
            return False

        with open(filename_cams, "rb") as fp:
            self.camparams = pickle.load(fp)
        with open(filename_pairs, "rb") as fp:
            self.pairparams = pickle.load(fp)
        return True

    def handle_keystroke(self, key):
        if key == ' ' and self.stage == StereoVision.STAGES["CALIBRATE_WAIT"]:
            self.stage = StereoVision.STAGES["CALIBRATING"]
        elif key == 'c':
            self.reset_calibration()
        elif key == 'a':
            self.sgbm = not self.sgbm
            self.stereo_matcher = None
            print("Now using", "SGBM" if self.sgbm else "BM")

