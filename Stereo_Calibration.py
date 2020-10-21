import numpy as np
import cv2
def stereo_calibration_opencv(objpoints,imgpoints_left,imgpoints_right,mtx_l,dist_l,mtx_r,dist_r,size):
    _, _ , _ , _, _, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_l,
                                                                        dist_l, mtx_r, dist_r, size)
    return R,T,E,F
def stereo_calibration():
    pass
