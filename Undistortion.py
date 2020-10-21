import cv2
import numpy as np
import os
from Is_Horizontal import *
def listdir(path,list_name):
    for file in os.listdir(path):
        file_path=os.path.join(path,file)
        if os.path.isdir(file_path):
            listdir(file_path,list_name)
        else:
            list_name.append(file_path)
def load_img(list_name,img_list):
    for img_path in list_name:
        img_list.append(cv2.imread(img_path))

def save_img(position,which_function,list_name,img_list):
    # which_function stand for undistort1 or undistort2
    j=0
    for i in list_name:
        path = 'Output/'+position+'/'+which_function+position+i[-6:]
        cv2.imwrite(path,img_list[j])
        j+=1


def undistort1(img,mtx,dist,size):
    #use cv2.undisotrt() to undistort picture


    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, size, 1, size)
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    return dst
def undistort2(img,mtx,dist,R,P,size):

    #newcameramtx_l,roi_l=cv2.getOptimalNewCameraMatrix(mtx_l,dist_l,size,1,size)

    map1,map2=cv2.initUndistortRectifyMap(mtx,dist,R,P,size=size,m1type=cv2.CV_16SC2)

    #newcameramtx_r,roi_r=cv2.getOptimalNewCameraMatrix(mtx_r,dist_r,size,1,size)

    dst=cv2.remap(img,map1,map2,cv2.INTER_LINEAR)

    return dst
if __name__=='__main__':
    #test one picture
    path_l='Input/left/left01.jpg'
    path_r='Input/right/right01.jpg'
    parameters_stereo=np.load('parameters_stereo for calibration.npz')
    mtx_l,dist_l=parameters_stereo['mtx_l'],parameters_stereo['dist_l']
    mtx_r, dist_r = parameters_stereo['mtx_r'], parameters_stereo['dist_r']
    img_l=cv2.imread(path_l)
    img_r=cv2.imread(path_r)
    size=(640,480)
    dst=undistort1(img_l,mtx_l,dist_l,size)
    dst2=undistort1(img_r,mtx_r,dist_r,size)
    cv2.imwrite('Output/left/new_left01.jpg',dst)
    cv2.imwrite('Output/right/new_right01.jpg',dst2)
    '''
    undistort pictures and stereo rectify
    parameters_rectify=np.load('parameters_stereo_rectify.npz')
    R1,R2=parameters_rectify['R1'],parameters_rectify['R2']
    P1,P2=parameters_rectify['P1'],parameters_rectify['P2']
    dst=undistort2(img_l,mtx_l,dist_l,R1,P1,size)
    dst2=undistort2(img_r,mtx_r,dist_r,R2,P2,size)
    '''
    show_line(img_l,img_r,size)
    show_line(dst,dst2,size)










