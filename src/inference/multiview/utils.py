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

