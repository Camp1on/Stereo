import cv2
import numpy as np
def show_line(imgl,imgr,size):
    w,h=size
    for i in range(20):
        cv2.line(imgl, (0, i*24), (w, i*24), (0,255,0), 1)

    for i in range(20):
        cv2.line(imgr, (0, i*24), (w, i*24), (0,255,0), 1)
    canvas = np.zeros((h,w*2,3), dtype="uint8")
    canvas[:, :w] = imgl
    canvas[:, w:] = imgr
    cv2.line(canvas, (w, 0), (w, h), (255,0,0), 1)
    cv2.imshow('canvas', canvas)
    cv2.waitKey(500)
    cv2.imwrite('stereo_unrectified01.jpg',canvas)
    cv2.destroyAllWindows()
if __name__=='__main__':
    # judge whether epipolar line is horizontal
    #imgl=cv2.imread('Output/left/new_left03.jpg')  # only undistort picture
    #imgr=cv2.imread('Output/left/new_right03.jpg')
    imgl=cv2.imread('Input/left/left03.jpg')    # firstly undistort pictures and then stereo_rectify
    imgr=cv2.imread('Input/right/right03.jpg')
    size=imgl[:,:,0].shape[::-1]
    #imgl=cv2.imread('Output/left03(rectified).jpg') # Use StereoCalibrate and StereoRecify
    #imgr=cv2.imread('Output/right03(rectified).jpg')

    show_line(imgl,imgr,size)