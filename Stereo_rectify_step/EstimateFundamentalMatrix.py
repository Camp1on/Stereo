import numpy as np
import cv2
import os

def load_points(file):
    l = []
    with open(file,"r") as f:
        for pnt in f.readlines():
            l.append(pnt.strip().split())
        f.close()
    return np.asarray(l,dtype = np.float32)




def svdecompose(imgpoints_l,imgpoints_r, rank =2):
    num_pts = imgpoints_l.shape[0]
    xa = imgpoints_l[:,0]
    ya = imgpoints_l[:,1]
    xb = imgpoints_r[:,0]
    yb = imgpoints_r[:,1]
    ones = np.ones(num_pts)
    A = np.column_stack((xa*xb, ya*xb, xb, xa*yb, ya*yb, yb, xa, ya, ones))
    _,_,V = np.linalg.svd(A, full_matrices=True)
    F = V.T[:,-1]
    F = F.reshape((3,3))

    ## F is a rank-2 matrix actually, so we throw off the least eigen value by again decomposing the F
    # into rotation, squeeze, rotation, set last value of S to 0, then multiply them back, to get rank-2 matrix.
    if rank==2:
        U, S, V = np.linalg.svd(F)
        S[-1] = 0
        S = np.diag(S)
        F = np.dot(np.dot(U, S), V)
    return F




