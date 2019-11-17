from ..inference import Inference
from images.image import Image
from images.image_type import ImageType

import cv2
import numpy as np
from typing import Sequence, Dict


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
