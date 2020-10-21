import  numpy as np
import math
import cv2
def sobel_filter(image):
    # sobel filtering for preprocessing
    image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    height, width = image.shape
    out_image = np.zeros((height, width))

    table_x = np.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    table_y = np.array(([-1, 0, 1],  [-2, 0, 2], [-1, 0, 1]))

    for y in range(2, width-2):
        for x in range(2, height-2):
            cx, cy = 0, 0
            for offset_y in range(0, 3):
                for offset_x in range(0, 3):
                    pix = image[x + offset_x -
                                1, y + offset_y - 1]
                    if offset_x != 1:
                        cx += pix * table_x[offset_x, offset_y]
                    if offset_y != 1:
                        cy += pix * table_y[offset_x, offset_y]
            out_pix = math.sqrt(cx ** 2 + cy ** 2)
            out_image[x, y] = out_pix if out_pix > 0 else 0
    np.putmask(out_image, out_image > 255, 255)
    # out_image=cv2.convertScaleAbs(out_image)
    return out_image

def Gaussian_filter(img,size):
    """
    read and blur stereo image pair.
    :param left_name: name of the left image.
    :param right_name: name of the right image.
    :param parameters: structure containing parameters of the algorithm.
    :return: blurred left and right images.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, size, 0, 0)
    return gray

if __name__=='__main__':

    imgl=cv2.imread('../Input/000000_10L.png')
    imgr=cv2.imread('../Input/000000_10R.png')
    #picl=Gaussian_filter(imgl,(5,5))
    picl=sobel_filter(imgl)
    cv2.imwrite('pre_picl.png',picl)
    cv2.imshow('pre_picl',picl)
    cv2.waitKey(0)
    cv2.destroyAllWindows()