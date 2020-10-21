import numpy as np
import cv2
from Camera_Calibration import *
from Stereo_Calibration import *
from Undistortion import *
from Stereo_Rectify import *
from Stereo_Match import *
from Is_Horizontal import *
# ===================Part 1: Binocular Camera Calibration==================
objp=np.zeros((6*9,3),np.float32)
objp[:,:2]=np.mgrid[0:9,0:6].T.reshape(-1,2)  #get chess's world coordinate
image_size=(640,480)  # pictures' resolution

print('calibrate left_camera')
left_path='Input/left/*.jpg'

mtx_l,dist_l,rvecs_l,tvecs_l,objpoints,imgpoints_l=monocularCameraCalibration(left_path,objp,image_size)
#_,rvecs_test,tvecs_test=cv2.solvePnP(objpoints,imgpoints_l,mtx_l,dist_l)
print('rotation vector for each image=', *rvecs_l, sep = "\n")
#print('rotation vector for each image test',*rvecs_test,sep="\n")
print('translation vector for each image=', *tvecs_l, sep= "\n")
#print('transition vector for each image test',*tvecs_test,sep="\n")

input('Program paused. Press ENTER to continue')

print('calibrate right_camera')

right_path='Input/right/*.jpg'

mtx_r,dist_r,rvecs_r,tvecs_r,_ , imgpoints_r=monocularCameraCalibration(right_path,objp,image_size)

input('Program paused. Press ENTER to continue')

# ===================Part 2: Stereo Calibration==================
R,T,E,F=stereo_calibration_opencv(objpoints,imgpoints_l,imgpoints_r,mtx_l,dist_l,mtx_r,dist_r,image_size)

np.savez("parameters_stereo for calibration.npz",mtx_l=mtx_l,mtx_r=mtx_r,dist_l=dist_l,dist_r=dist_r,R=R,T=T)
np.savez("points.npz",objpoints=objpoints,imgpoints1=imgpoints_l,imgpoints2=imgpoints_r)

print('intrinsic matrix of left camera=\n', mtx_l)
print('intrinsic matrix of right camera=\n', mtx_r)
print('distortion coefficients of left camera=\n', dist_l)
print('distortion coefficients of right camera=\n', dist_r)
print('Transformation from left camera to right:\n')
print('R=\n', R)
print('\n')
print('T=\n', T)
print('\n')
input('Program paused. Press ENTER to continue')
# ===================Part 3: Stereo Rectification==================
R1,R2,P1,P2,Q,ROI1,ROI2=stereo_rectify_calibrate(mtx_l,dist_l,mtx_r,dist_r,image_size,R,T)

np.savez('parameters_stereo_rectify',R1=R1,R2=R2,P1=P1,P2=P2)


print('rectify left image')
image_l_path = []  # store images_l' path
image_l = []  # store images_l
left_path='Input/left'
listdir(left_path, image_l_path)
load_img(image_l_path, image_l)
# for k in range(len(image_l)):  # only undistort left pictures
    # image_l[k]=undistort1(image_l[k],mtx_l,dist_l,image_size)
# save_img('left','new_',image_l_path,image_l)
for k in range(len(image_l)):
    image_l[k]=undistort2(image_l[k],mtx_l,dist_l,R1,P1,image_size)
save_img('left','rectified_',image_l_path,image_l)
input('Program paused. Press ENTER to continue')
print('rectify right image')

image_r_path = []  # store images_r' path
image_r = []  # store images_r
right_path='Input/right'
listdir(right_path, image_r_path)
load_img(image_r_path, image_r)
# for k in range(len(image_r)):     # only undistort right pictures
    # image_r[k]=undistort1(image_r[k],mtx_r,dist_r,image_size)
# save_img('right','new_',image_r_path,image_r)

for k in range(len(image_r)):
    image_r[k]=undistort2(image_r[k],mtx_r,dist_r,R2,P2,image_size)
save_img('right','rectified_',image_r_path,image_r)

print('Whether epipolar line is horizontal')

for i in range(1,15):
    if i==10: continue
    order_number='{0:0=2d}'.format(i)
    pathL='Output/left/rectified_left'+order_number+'.jpg'
    pathR='Output/right/rectified_right'+order_number+'.jpg'
    pic1=cv2.imread(pathL)
    pic2=cv2.imread(pathR)
    show_line(pic1,pic2,image_size)

input('Program paused. Press ENTER to continue')
print('Calculate re-projection error')
mean_error_left=0

for i in range(len(objpoints)):
    imgpoints_c,_=cv2.projectPoints(objpoints[i],rvecs_l[i],tvecs_l[i],mtx_l,dist_l)
    error=cv2.norm(imgpoints_l[i],imgpoints_c,cv2.NORM_L2)/len(imgpoints_c)
    mean_error_left+=error
print('total error of left_pictures: ',mean_error_left/len(objpoints))

mean_error_right=0

for i in range(len(objpoints)):
    imgpoints_c,_=cv2.projectPoints(objpoints[i],rvecs_r[i],tvecs_r[i],mtx_r,dist_r)
    error=cv2.norm(imgpoints_r[i],imgpoints_c,cv2.NORM_L2)/len(imgpoints_c)
    mean_error_right+=error
print('total error of right_pictures: ',mean_error_right/len(objpoints))








# ===================Part 4: Stereo Match==================
print('loading images...')
'''
# stereo match with chessBoard pictures
for i in range(1,15):
    if i==10: continue
    order_number='{0:0=2d}'.format(i)
    # pathL='Output/left/new_left'+order_number+'.jpg'   # stereo Match with only undistorted pictures
    # pathR='Output/right/new_right'+order_number+'.jpg'

    pathL='Output/left/rectified_left'+order_number+'.jpg'
    pathR='Output/right/rectified_right'+order_number+'.jpg'
    imgL=cv2.imread(pathL)
    imgR=cv2.imread(pathR)
    SGBM(imgL,imgR)

'''
imgl=cv2.imread('Input/000000_10L.png')

imgr=cv2.imread('Input/000000_10R.png')

SGM(imgl,imgr)


