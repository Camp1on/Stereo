import numpy as np
import cv2
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

def get_l_r(pos, arr):
    l = r = pos
    for i in range(pos, 1, -1):
        if arr[i] > 0:
            l = i
            break
    for i in range(pos, len(arr)-1):
        if arr[i] > 0:
            r = i
            break

    return l, r
def fill_hole(disparity_map):
    height,width=disparity_map.shape
    new_disparity_map=np.ones(shape=(height,width),dtype=np.uint8)
    for y in range(height):
        for x in range(width ):
            if disparity_map[y, x] <= 0:
                left, right = get_l_r(x, disparity_map[y])

                new_disparity_map[y, x] = (disparity_map[y, left ] +disparity_map[y, right ]) //2
            else:
                new_disparity_map[y, x] = disparity_map[y, x]
    return new_disparity_map
def select_disparity(directions,aggregation_volume,parameters):
    """
    last Zhang_method_step of the sgm algorithm, corresponding to equation 14 followed by winner-takes-all approach.
    :param aggregation_volume: H x W x D x N array of matching cost for all defined directions.
    :return: disparity image.
    """
    volume = np.sum(aggregation_volume, axis=3)
    np.savez('aggregation_value_'+directions,aggregation_volume=aggregation_volume)
    disparity_map = np.argmin(volume, axis=2)
    height,width=disparity_map.shape
    disparity_sub_pixel=np.zeros(shape=(height,width),dtype=np.float16)
    '''
    # compute sub-disparity
    for y in range(height):
        for x in range(width):
            if disparity_map[y,x]==0 or disparity_map[y,x]==parameters.max_disparity-1:
                disparity_sub_pixel[y,x]=disparity_map[y,x]
            else:
                cost_min=volume[y,x,disparity_map[y,x]]
                idx1=disparity_map[y,x]-1
                idx2=disparity_map[y,x]+1
                cost_l=volume[y,x,idx1]
                cost_r=volume[y,x,idx2]
                denom=max(1,cost_l+cost_r-2*cost_min)
                disparity_sub_pixel[y,x]=disparity_map[y,x]+(cost_l-cost_r)/(denom*2.0)
    '''
    np.savez('disparity_'+directions, disparity=disparity_map,disparity_sub=disparity_sub_pixel)

    return disparity_map

def normalize(volume, parameters):
    """
    transforms values from the range (0, 128) to (0, 255).
    :param volume: n dimension array to normalize.
    :param parameters: structure containing parameters of the algorithm.
    :return: normalized array.
    """
    return 255.0 * volume / parameters.max_disparity



def left_right_check(disp_left, disp_right,threshold):
    # Left-right verification
    height, width = disp_left.shape
    disparity_post=np.zeros(shape=(height,width),dtype=np.int8)

    for h in range(height):

        for w in range(width):
            left = int((disp_left[h,w]))
            idx_right=(w-left)
            if idx_right >= 0 and idx_right<width:
                right=int(disp_right[h,idx_right])
                if abs(left-right)>threshold:
                    disparity_post[h,w]=-1 # 左右视差不一致
                else:
                    disparity_post[h,w]=disp_left[h,w]
            else:
                disparity_post[h,w]=-1  #通过视差值在右视差中找不到同名像素

    return disparity_post
'''
def is_unique(sec_min_cost,min_cost,uniqueness_ratio):
    pass


def removeSpecles(disp,width,height,diff_insame,min_speckle_area,invalid_val):
    if width<0 or height <0:
        return
    visited=np.array([False for _ in range(height*width)]).reshape(height,width)
    for y in range(height):
        for x in range(width):
            if visited[y,x] or disp[y,x]==invalid:

                continue
            visited[y,x]=True
            vec=[(y,x)]
            cur,Next=0,0
            while(True):
                Next=len(vec)
                for k in range(cur,Next):
                    pixel=vec[k]
                    row,col=pixel[0],pixel[1]
                    disp_base=disp[row,col]
                    for r in range(-1,2):
                        for c in range(-1,2):
                            if r==0 and c==0:
                                continue
                            row_r=row+r
                            col_c=col+c
                            if row_r>=0 and row_r<height and col_c>=0 and col_c<width:
                                if not visited[row_r,col_c] and abs(disp[row_r,col_c]-disp_base)<= diff_insame:
                                    vec.append((row_r,col_c))
                                    visited[row_r,col_c]=True
                cur=Next
                if Next>=len(vec):
                    break
            if len(vec)<min_speckle_area:
                for pixel in vec:
                    disp[pixel[0],pixel[1]]=invalid_val
'''
if __name__=='__main__':
    # post-process disparity
    disparity_left=np.load('../disparity_left.npz')
    # disparity_left_cost=np.load('../aggregation_value_left.npz')
    disparity_left_map=disparity_left['disparity']

    # disparity_sub_left=disparity_left['disparity_sub']
    # disparity_left_volume=disparity_left_cost['aggregation_volume']
    disparity_right=np.load('../disparity_right.npz')

    disparity_right_map=disparity_right['disparity']

    # disparity_sub_right=disparity_right['disparity_sub']

    disparity_left_post=left_right_check(disparity_left_map,disparity_right_map,1)

    new_disparity_left=fill_hole(disparity_left_post)

    disparity_left_show=np.uint8(normalize(disparity_left_post,Parameters()))

    new_disparity_left_show=np.uint8(normalize(new_disparity_left,Parameters()))

    new_disparity_left_show=cv2.medianBlur(new_disparity_left_show,Parameters().bsize[0])
    disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        disparity_left_show,alpha=256 / Parameters().max_disparity),cv2.COLORMAP_JET)

    new_disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
        new_disparity_left_show,alpha=256 / Parameters().max_disparity) , cv2.COLORMAP_JET)
    cv2.imshow('post_picture_7.jpg',disparity_left_color)
    cv2.imshow('new_post_picture_7.jpg', new_disparity_left_color)
    '''
    cv2.imwrite('disparity_128_org_7.jpg',disparity_left_show)
    cv2.imwrite('disparity_128_fill_7.jpg',new_disparity_left_show)
    cv2.imwrite('disparity_128_org_color_7.jpg', disparity_left_color)
    cv2.imwrite('disparity_128_fill_color_7.jpg', new_disparity_left_color)
    '''
    cv2.waitKey(0)
    cv2.destroyAllWindows()
