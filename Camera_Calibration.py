
import numpy as np
import cv2
import glob
from Zhang_method_step.homography import get_homography
from Zhang_method_step.instrinsics import get_intrinsics_param
from Zhang_method_step.extrinsics import get_extrinsics_param
from Zhang_method_step.distortion_coefficient import get_distortion
from Zhang_method_step.refine_all import refinall_all_param
def monocularCameraCalibration(path,objp,img_size):
    #termination criteria
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,30,0.001)

    imgpoints=[]
    objpoints=[]
    images=glob.glob(path)
    images.sort()
    for fname in images:
        img=cv2.imread(fname)
        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret,corners=cv2.findChessboardCorners(gray,(9,6),None)

        if ret==True:
            objpoints.append(objp)
            corners2=cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img,(9,6),corners2,ret)
            cv2.imshow(fname,img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()
    # return imgpoints   # this reval can be used for 8-points-algorithm
    _,mtx, dist , rvecs, tvecs = cv2.calibrateCamera(objpoints,imgpoints,img_size,None,None)
    return mtx,dist,rvecs,tvecs,np.array(objpoints),np.array(imgpoints)
def Zhang_method_calibration(path,objp,img_size):

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    imgpoints = []
    objpoints = []
    objpoints_x_y=[]
    images = glob.glob(path)
    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

        if ret == True:
            objpoints.append(objp)
            objpoints_x_y.append(objp[:,:2])
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            corners2=corners2.reshape(-1,2)
            imgpoints.append(corners2)
            cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
            cv2.imshow('img', img)
            cv2.waitKey(500)
    cv2.destroyAllWindows()

    H = get_homography(imgpoints,objpoints_x_y)

    intrinsics_param=get_intrinsics_param(H)

    extrinsics_param=get_extrinsics_param(H,intrinsics_param)

    k=get_distortion(intrinsics_param,extrinsics_param,imgpoints,objpoints_x_y)

    [new_intrinsics_param,new_k,new_extrinsics_param]=refinall_all_param(intrinsics_param,
                                                                          k,extrinsics_param,objpoints,imgpoints)
    print("intrinsics_parm:\t", new_intrinsics_param)
    print("distortionk:\t", new_k)
    print("extrinsics_parm:\t", new_extrinsics_param)
    return
if __name__=='__main__':
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    path='Output/left/*.jpg' # use one series of pictures to calibrate
    img_size=(640,480)
    # monocularCameraCalibration(path,objp,img_size) # directly use opencv to calibrate
    Zhang_method_calibration(path,objp,img_size) # use Zhang's method to calibrate one series of pictures




