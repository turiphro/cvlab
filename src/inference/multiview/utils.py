import cv2
import numpy as np


def calibrate_camera(world_coords, img_coords, img_size, crop_alpha=1):
    """
    Calibrate a single camera, returning the camera parameters

    Parameters:
      img_coords: list of matching feature points for all images,
                  in image coordinates
      img_size: size of the images (x, y), used for intrinsic_crop
    """
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        world_coords, img_coords, img_size, None, None)
    camera_matrix_crop, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, img_size, crop_alpha, img_size)

    camparams = {
        # overall RMS re-projection error
        "ret": ret,
        # intrinsic
        # [fx  0 cx]
        # [ 0 fy cy]
        # [ 0  0  1]
        "intrinsic": camera_matrix,
        # new intrinsic matrix that removes distorted pixels
        "intrinsic_crop": camera_matrix_crop,
        "intrinsic_roi": roi,
        # extrinsic (R)
        # list of [r1, r2, r3] (len(img_coords)/img_size times)
        "extrinsic_r": rvecs,
        # extrinsic (T)
        # list of [t1, t2, t3] (len(img_coords)/img_size times)
        "extrinsic_t": tvecs,
        # distortion
        # (k1, k2, p1, p2, k3)
        # k=radial, p=tangential
        "distortion": dist_coeffs
    }
    return camparams


def calibrate_camera_pair(world_coords, img_coords_l, img_coords_r,
                          camparams_l, camparams_r, img_size):
    # Note: images should be of the same size
    _, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        world_coords, img_coords_l, img_coords_r,
        camparams_l["intrinsic"], camparams_l["distortion"],
        camparams_r["intrinsic"], camparams_r["distortion"],
        img_size)
    rect_l, rect_r, proj_l, proj_r, disparity2depth, roi_l, roi_r = cv2.stereoRectify(
        camparams_l["intrinsic"], camparams_l["distortion"],
        camparams_r["intrinsic"], camparams_r["distortion"],
        img_size,
        R, T)
    map_x_l, map_y_l = cv2.initUndistortRectifyMap(
        camparams_l["intrinsic"], camparams_l["distortion"],
        rect_l, proj_l, img_size, cv2.CV_32FC1)
    map_x_r, map_y_r = cv2.initUndistortRectifyMap(
        camparams_r["intrinsic"], camparams_r["distortion"],
        rect_r, proj_r, img_size, cv2.CV_32FC1)

    pairparams = {
        "R": R,
        "T": T,
        "rect": (rect_l, rect_r),
        "proj": (proj_l, proj_r),
        "roi": (roi_l, roi_r),
        "disparity2depth": disparity2depth,
        "map_x": (map_x_l, map_x_r),
        "map_y": (map_y_l, map_y_r),
    }

    return pairparams
