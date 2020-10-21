import  numpy as np
import cv2
import sys
import time as t

from Stereo_match_step.Pre_process import *
class Parameters:
    def __init__(self, max_disparity=96, P1=5, P2=70, csize=(7, 7), bsize=(3, 3)):
        """
        represent all parameters used in the sgm algorithm.
        :param max_disparity: maximum distance between the same pixel in both images.
        :param P1: penalty for disparity difference = 1
        :param P2: penalty for disparity difference > 1
        :param csize: size of the kernel for the census transform.
        :param bsize: size of the kernel for blurring the images and median filtering.
        """
        self.max_disparity = max_disparity
        self.P1 = P1
        self.P2 = P2
        self.csize = csize
        self.bsize = bsize

def SAD_L(imgl,imgr,num_disparity=32,block_size=11):

    height, width = imgl.shape
    disparity_matrix = np.zeros((height, width), dtype=np.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):

        for j in range(half_block, width - half_block):
            left_block = imgl[i - half_block:i +
                             half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(j - half_block - 1, num_disparity)):
                right_block = imgr[i - half_block:i +
                                  half_block, j - half_block - d:j + half_block - d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp

    return disparity_matrix
def SAD_R(imgl, imgr , num_disparity=32, block_size=11):

    height, width = imgr.shape
    disparity_matrix = np.zeros((height, width), dtype=np.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            right_block = imgr[i - half_block:i +
                              half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(width - j - half_block, num_disparity)):

                left_block = imgl[i - half_block:i +
                                 half_block, j - half_block + d:j + half_block + d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    return disparity_matrix

def normalize(volume, parameters):
    """
    transforms values from the range (0, 64) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity


def census(imgl,imgr,parameters,save_images):

    """
    first Zhang_method_step of the sgm algorithm, matching cost based on census transform and hamming distance.
    :param left: left image.
    :param right: right image.
    :param parameters: structure containing parameters of the algorithm.
    :param save_images: whether to save census images or not.
    :return: H x W x D array with the matching costs.
    """
    assert imgl.shape[0] == imgr.shape[0] and imgl.shape[1] == imgr.shape[1], 'left & right must have the same shape.'
    assert parameters.max_disparity > 0, 'maximum disparity must be greater than 0.'

    height = imgl.shape[0]
    width = imgr.shape[1]
    cheight = parameters.csize[0]
    cwidth = parameters.csize[1]
    y_offset = int(cheight / 2)
    x_offset = int(cwidth / 2)
    disparity = parameters.max_disparity


    left_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    right_img_census = np.zeros(shape=(height, width), dtype=np.uint8)
    left_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    right_census_values = np.zeros(shape=(height, width), dtype=np.uint64)
    print('\tComputing left and right cnesus...',end='')
    sys.stdout.flush()
    dawn=t.time()
    # pixels on the border will have no census values
    for y in range(y_offset, height - y_offset):
        for x in range(x_offset, width - x_offset):
            left_census = np.int64(0)
            center_pixel=imgl[y,x]
            #center_pixel = recalculate_center(imgl,y,x,30)
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = imgl[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        left_census = left_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        left_census = left_census | bit
            left_img_census[y, x] = np.uint8(left_census)
            left_census_values[y, x] = left_census

            right_census = np.int64(0)
            center_pixel=imgr[y,x]
            #center_pixel = recalculate_center(imgr,y,x,2)
            reference = np.full(shape=(cheight, cwidth), fill_value=center_pixel, dtype=np.int64)
            image = imgr[(y - y_offset):(y + y_offset + 1), (x - x_offset):(x + x_offset + 1)]
            comparison = image - reference
            for j in range(comparison.shape[0]):
                for i in range(comparison.shape[1]):
                    if (i, j) != (y_offset, x_offset):
                        right_census = right_census << 1
                        if comparison[j, i] < 0:
                            bit = 1
                        else:
                            bit = 0
                        right_census = right_census | bit
            right_img_census[y, x] = np.uint8(right_census)
            right_census_values[y, x] = right_census
    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))
    if save_images:
        cv2.imwrite('../Output/SGM/left_census_7_sobel.png', left_img_census)
        cv2.imwrite('../Output/SGM/right_census_7_sobel.png', right_img_census)

    print('\tComputing cost volumes...', end='')
    sys.stdout.flush()
    dawn = t.time()
    left_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    right_cost_volume = np.zeros(shape=(height, width, disparity), dtype=np.uint32)
    lcensus = np.zeros(shape=(height, width), dtype=np.int64)
    rcensus = np.zeros(shape=(height, width), dtype=np.int64)
    for d in range(0, disparity):
        rcensus[:, (x_offset + d):(width - x_offset)] = right_census_values[:, x_offset:(width - d - x_offset)]
        left_xor = np.int64(np.bitwise_xor(np.int64(left_census_values), rcensus))
        left_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(left_xor == 0):
            tmp = left_xor - 1
            mask = left_xor != 0
            left_xor[mask] = np.bitwise_and(left_xor[mask], tmp[mask])
            left_distance[mask] = left_distance[mask] + 1
        left_cost_volume[:, :, d] = left_distance

        lcensus[:, x_offset:(width - d - x_offset)] = left_census_values[:, (x_offset + d):(width - x_offset)]
        right_xor = np.int64(np.bitwise_xor(np.int64(right_census_values), lcensus))
        right_distance = np.zeros(shape=(height, width), dtype=np.uint32)
        while not np.all(right_xor == 0):
            tmp = right_xor - 1
            mask = right_xor != 0
            right_xor[mask] = np.bitwise_and(right_xor[mask], tmp[mask])
            right_distance[mask] = right_distance[mask] + 1
        right_cost_volume[:, :, d] = right_distance

    dusk = t.time()
    print('\t(done in {:.2f}s)'.format(dusk - dawn))

    return left_cost_volume, right_cost_volume
if __name__ =='__main__':
    size=(1242,375)
    print(sys.argv[0])
    #imgl=cv2.imread('../Input/000000_10L.png')
    #imgr=cv2.imread('../Input/000000_10R.png')
    imgl = cv2.imread('Input/000000_10L.png')
    imgr = cv2.imread('Input/000000_10R.png')
    #imgl= cv2.GaussianBlur(imgl, (3,3), 0, 0)
    #imgr = cv2.GaussianBlur(imgr, (3,3), 0, 0)
    imgl=sobel_filter(imgl)
    imgr=sobel_filter(imgr)
    census(imgl,imgr,Parameters(),True)
