import numpy as np
import cv2
from Stereo_rectify_step.EstimateFundamentalMatrix import *


def get_lines(points, F_matrix, from_where=2):
    ## use from_where as 2, when u insert points in 2nd img.
    pts_homo = np.column_stack((points, np.ones(points.shape[0])))
    if from_where == 2:
        lines = np.dot(pts_homo, F_matrix)
    else:
        lines = np.dot(pts_homo, F_matrix.T)

    lines /= np.sqrt(((lines[:, 0]) ** 2 + (lines[:, 1]) ** 2)).reshape(-1, 1)
    return lines


def compute_fundamental_matrix(imgpoints_l, imgpoints_r):
    # Normalizing the points, because normalizing is good. (as stated in the 8-point algorithm)
    mean_A = np.mean(imgpoints_l, axis=0)
    scale_A = np.linalg.norm((imgpoints_l - mean_A)) / len(imgpoints_l)

    mean_B = np.mean(imgpoints_r, axis=0)
    scale_B = np.linalg.norm((imgpoints_r - mean_B)) / (len(imgpoints_r))

    scale_A = np.sqrt(2) / scale_A
    scale_B = np.sqrt(2) / scale_B

    norm_pts_A = (imgpoints_l - mean_A) * scale_A;
    norm_pts_B = (imgpoints_r - mean_B) * scale_B;

    denorm_A = np.asarray([[scale_A, 0, -scale_A * mean_A[0]],
                           [0, scale_A, -scale_A * mean_A[1]],
                           [0, 0, 1]], dtype=np.float32)

    denorm_B = np.asarray([[scale_B, 0, -scale_B * mean_B[0]],
                           [0, scale_B, -scale_B * mean_B[1]],
                           [0, 0, 1]], dtype=np.float32)

    norm_pts_A = np.column_stack((norm_pts_A, np.ones(norm_pts_A.shape[0])))
    norm_pts_B = np.column_stack((norm_pts_B, np.ones(norm_pts_B.shape[0])))

    # Our fundamental-matrix is rank-2, so we need to set the smallest singular-value to 0.
    Fundam = svdecompose(norm_pts_A, norm_pts_B, rank=2)

    # Denormalizing the F-matrix with our previously stored stats.
    F = np.dot(denorm_B.T, np.dot(Fundam, denorm_A))

    # print("Estimated Fundamental-Matrix is \n", F)


    return F


if __name__ == "__main__":
    # imgpoints_l,imgpoints_r=
    F = compute_fundamental_matrix(pts_2dA, pts_2dB)
    print("Estimated Fundamental-matrix is..\n ", F)