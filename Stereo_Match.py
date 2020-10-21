import math
import argparse
import sys
import time as t
from Stereo_match_step.Compute_costs import *
from Stereo_match_step.Pre_process import *
from  Stereo_match_step.Post_process import *
from  Stereo_match_step.Aggregate_costs import *

class Parameters:
    def __init__(self, max_disparity=128, P1=10, P2=120, csize=(7, 7), bsize=(3, 3)):
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


class Paths:
    def __init__(self):
        """
        represent the relation between the directions.
        """
        self.paths = [N, NE, E, SE, S, SW, W, NW]
        self.size = len(self.paths)
        self.effective_paths = [(E,  W), (SE, NW), (S, N), (SW, NE)]

def BM(imgl,imgr):
    # range of disparity
    num_disparity=32
    print('Read images')
    start_time = t.time()



    # Preprocessing
    sobel_left = sobel_filter(imgl)
    sobel_right = sobel_filter(imgr)


    print('Start LeftBM')
    # Calculate left disparity
    disparity_left = SAD_L(
        sobel_left, sobel_right, num_disparity, 11)
    cv2.imwrite('Output/BM/disparity_left_32_21.jpg', disparity_left)
    disparity_left_color =  cv2.applyColorMap(cv2.convertScaleAbs(
        disparity_left, alpha=256 / num_disparity), cv2.COLORMAP_JET)
    cv2.imwrite('Output/BM/disparity_leftRGB_112_7.jpg', disparity_left_color)

    print('Start RightBM')
    # Calculate right disparity
    disparity_right = SAD_R(
        sobel_left, sobel_right, num_disparity, 11)
    disparity_right_color = cv2.applyColorMap(cv2.convertScaleAbs(
        disparity_right, alpha=256 / num_disparity), cv2.COLORMAP_JET)
    cv2.imwrite('Output/BM/disparity_rightRGB_32_21.jpg', disparity_right_color)


    print('Duration: %s seconds\n' % (t.time() - start_time))


    # Display result
    cv2.imshow('Left', imgl)
    cv2.imshow('Disparity RGB', disparity_left_color)
    cv2.waitKey(0)


def SGBM(dst,dst2):
    imgL = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(dst2, cv2.COLOR_BGR2GRAY)
    window_size = 9
    min_disp = 0
    num_disp = 112 - min_disp
    stereo = cv2.StereoSGBM_create(minDisparity=min_disp,
                                   numDisparities=num_disp,
                                   blockSize=3,
                                   P1=8 * 3 * window_size ** 2,
                                   P2=32 * 3 * window_size ** 2,
                                   disp12MaxDiff=1,
                                   uniquenessRatio=10,
                                   speckleWindowSize=10,
                                   speckleRange=1
                                   )

    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0
    new_disp = (disp - min_disp) / num_disp

    count = 0
    for j in range(new_disp.shape[0]):
        if new_disp[0, j] > 0:
            count = j
            break
    new_disp_pp = np.zeros_like(new_disp)
    for i in range(new_disp.shape[0]):
        for j in range(count, new_disp.shape[1] - 1):
            if new_disp[i, j] <= 0:
                left, right = get_l_r(j, new_disp[i])
                #up, down  = get_f_r(i, new_disp[:, j])
                new_disp_pp[i, j] = (new_disp[i, left - 1] +new_disp[i, right+1]) /2
            else:
                new_disp_pp[i, j] = new_disp[i, j]
    new_disp_pp = cv2.medianBlur(new_disp_pp, 5)
    new_disp=new_disp*num_disp
    new_disp_pp=new_disp_pp*num_disp
    new_disp=cv2.convertScaleAbs(new_disp)
    new_disp_pp=cv2.convertScaleAbs(new_disp_pp)
    cv2.imwrite('Output/SGBM/org_disp_13.jpg', new_disp)
    cv2.imwrite('Output/SGBM/disp_pp_13.jpg', new_disp_pp)

    cv2.imshow('org_disp', new_disp)
    cv2.imshow("disp_pp", new_disp_pp)
    cv2.waitKey(0)
    '''
    right_stereo = cv2.StereoSGBM_create(minDisparity=-num_disp,
                                  numDisparities=num_disp,
                                  blockSize=8,
                                  P1=8 * 3 * window_size ** 2,
                                  P2=32 * 3 * window_size ** 2,
                                  disp12MaxDiff=1,
                                  uniquenessRatio=10,
                                  speckleWindowSize=100,
                                  speckleRange=32
                                  )

    right_disp1 = right_stereo.compute(imgR, imgL).astype(np.float32) / 16.0
    right_new_disp = -(right_disp1 - min_disp) / num_disp

    for i in range(375):
        for j in range(1242):
            if right_new_disp[i, j] >= 1:
                right_new_disp[i, j] = 0

    right_new_disp = cv2.medianBlur(right_new_disp, 5)
    '''
    disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        new_disp_pp, alpha=256 / num_disp), cv2.COLORMAP_JET)
    cv2.imwrite('Output/SGBM/disp_color_13.jpg', disparity_left_color)
    cv2.destroyAllWindows()







def SGM(imgl,imgr):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='disparity_map.png', help='name of the output image')
    parser.add_argument('--disp', default=128, type=int, help='maximum disparity for the stereo pair')
    parser.add_argument('--images', default=False, type=bool, help='save intermediate representations')
    args = parser.parse_args()

    output_name = args.output
    disparity = args.disp
    save_images = args.images

    dawn = t.time()

    parameters = Parameters(max_disparity=disparity, P1=10, P2=120, csize=(7, 7), bsize=(3, 3))

    paths = Paths()

    print('\nLoading images...')
    left = Gaussian_filter(imgl,parameters.bsize)
    right= Gaussian_filter(imgr,parameters.bsize)
    # left=sobel_filter(imgl)
    # right=sobel_filter(imgr)
    print('\nStarting cost computation...')
    left_cost_volume, right_cost_volume = census(left, right, parameters, save_images)
    if save_images:
        left_disparity_map = np.uint8(normalize(np.argmin(left_cost_volume, axis=2), parameters))
        cv2.imwrite('Output/SGM/disp_map_left_cost_volume.png', left_disparity_map)
        right_disparity_map = np.uint8(normalize(np.argmin(right_cost_volume, axis=2), parameters))
        cv2.imwrite('Output/SGM/disp_map_right_cost_volume.png', right_disparity_map)

    print('\nStarting left aggregation computation...')
    left_aggregation_volume = aggregate_costs(left_cost_volume, parameters, paths)
    print('\nStarting right aggregation computation...')
    right_aggregation_volume = aggregate_costs(right_cost_volume, parameters, paths)

    print('\nSelecting best disparities...')
    left_disparity_map = np.uint8(select_disparity('left',left_aggregation_volume,parameters))
    right_disparity_map = np.uint8(select_disparity('right',right_aggregation_volume,parameters))
    if save_images:
        left_disparity_map_norm=np.unit8(normalize(left_disparity_map,parameters))
        right_disparity_map_norm=np.unit8(normalize(right_disparity_map,parameters))
        cv2.imwrite('Output/SGM/left_disp_map_no_post_processing.png', left_disparity_map_norm)
        cv2.imwrite('Output/SGM/right_disp_map_no_post_processing.png', right_disparity_map_norm)

    print('\nApplying post process...')
    left_disparity_map=left_right_check(left_disparity_map,right_disparity_map,1)
    left_disparity_map=fill_hole(left_disparity_map)

    left_disparity_map_norm=np.uint8(normalize(left_disparity_map,parameters))
    right_disparity_map_norm=np.uint8(normalize(right_disparity_map,parameters))

    left_disparity_map_norm = cv2.medianBlur(left_disparity_map_norm, parameters.bsize[0])
    right_disparity_map_norm = cv2.medianBlur(right_disparity_map_norm, parameters.bsize[0])
    cv2.imwrite(f'Output/SGM/left_{output_name}', left_disparity_map_norm)
    cv2.imwrite(f'Output/SGM/right_{output_name}', right_disparity_map_norm)
    disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        left_disparity_map_norm, alpha=256 / parameters.max_disparity), cv2.COLORMAP_JET)
    cv2.imwrite(f'Output/SGM/left_Color_{output_name}', disparity_left_color)
    dusk = t.time()
    print('\nFin.')
    print('\nTotal execution time = {:.2f}s'.format(dusk - dawn))
    cv2.imshow('left_disparity_color',disparity_left_color)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()

if __name__=='__main__':
    imgl=cv2.imread('Input/000000_10L.png') # stereo match with stereo-rectified pictures
    imgr=cv2.imread('Input/000000_10R.png')

    #SGBM(imgl,imgr) # use BlockMatch to Stereo match
    #BM(imgl,imgr)

    #imgl=cv2.imread('calibrate_result/new_left03.jpg') # stereo match with no stereo-rectified pictures

    #imgr=cv2.imread('calibrate_result/new_right03.jpg')

    SGM(imgl,imgr)