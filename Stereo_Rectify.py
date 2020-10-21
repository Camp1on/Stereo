# -*- coding: utf-8 -*-

from Camera_Calibration import *  # to get two pairs coordinate of two pictures
from Stereo_rectify_step.FindEpipolarLines import *
from Stereo_rectify_step.FastHomography import warpHomo

def compute_epipole(F):
    print("F is ",F.shape)
    e1 = np.linalg.svd(F.T)[-1]
    e1 = e1[-1, :]/e1[-1, :][-1]

    e2 = np.linalg.svd(F)[-1]
    e2 = e2[-1, :]/e2[-1, :][-1]

    print("E shape is ",e1.shape,e2.shape)
    return e1,e2


def rectify_points(H, points):
    points = points.T
    for i in range(points.shape[1]):
        points[:, i] = np.dot(H, points[:, i])
        # convert the points to cartesian
        points[:, i] = points[:, i] / points[:, i][-1]

    return points

def find_H1_H2(e2, F, img2, pts1, pts2):

    w,h = img2.shape[:2]

    T = np.identity(3)
    T[0][2] = -1.0 * w / 2
    T[1][2] = -1.0 * h / 2

    e = T.dot(e2)
    print("e is ",e.shape)
    e1_prime = e[0]
    e2_prime = e[1]
    if e1_prime >= 0:
        alpha = 1.0
    else:
        alpha = -1.0

    R = np.identity(3)

    norm = np.sqrt(e1_prime**2 + e2_prime**2)
    R[0][0] = alpha * e1_prime / norm
    R[0][1] = alpha * e2_prime /norm
    R[1][0] = - alpha * e2_prime / norm
    R[1][1] = alpha * e1_prime / norm

    f = R.dot(e)[0]
    G = np.identity(3)
    G[2][0] = - 1.0 / f

    H2 = np.linalg.inv(T).dot(G.dot(R.dot(T)))

    e_prime = np.asarray([ [0, -e2[2], e2[1] ],
                           [e2[2], 0 , -e2[0]],
                           [-e2[1], e2[0], 0 ] ])

    v = np.array([1, 1, 1])
    M = e_prime.dot(F) + np.outer(e2, v)

    points1_hat = H2.dot(M.dot(pts1.T)).T
    points2_hat = H2.dot(pts2.T).T

    W = points1_hat / points1_hat[:, 2].reshape(-1, 1)
    b = (points2_hat / points2_hat[:, 2].reshape(-1, 1))[:, 0]
    # Minimisizng the least-squares to get the Matching-Transform..!
    a1, a2, a3 = np.linalg.lstsq(W, b)[0]
    HA = np.identity(3)
    HA[0] = np.array([a1, a2, a3])

    H1 = HA.dot(H2).dot(M)
    return H1, H2


def plot_epilines(pts_2dA, pts_2dB, img_a, img_b, F):
    eplines_a = get_lines(pts_2dB, F, from_where=2)
    eplines_b = get_lines(pts_2dA, F, from_where=1)

    n, m, _ = img_a.shape
    leftmost = np.cross([0, 0, 1], [n, 0, 1])
    rightmost = np.cross([0, m, 1], [n, m, 1])
    for i in range(len(eplines_a)):
        line_a, line_b = eplines_a[i], eplines_b[i]
        pt_a, pt_b = pts_2dA[i], pts_2dB[i]

        color = tuple(np.random.randint(0, 255, 3).tolist())
        leftmost_a = np.cross(line_a, leftmost)
        rightmost_a = np.cross(line_a, rightmost)
        leftmost_a = (leftmost_a[:2] / leftmost_a[2]).astype(int)
        rightmost_a = (rightmost_a[:2] / rightmost_a[2]).astype(int)
        cv2.line(img_a, tuple(leftmost_a[:2]), tuple(rightmost_a[:2]), color, thickness=1)
        cv2.circle(img_a, tuple(map(int, pt_a)), 4, color, -1)

        leftmost_b = np.cross(line_b, leftmost)
        rightmost_b = np.cross(line_b, rightmost)
        leftmost_b = (leftmost_b[:2] / leftmost_b[2]).astype(int)
        rightmost_b = (rightmost_b[:2] / rightmost_b[2]).astype(int)
        cv2.line(img_b, tuple(leftmost_b[:2]), tuple(rightmost_b[:2]), color, thickness=1)
        cv2.circle(img_b, tuple(map(int, pt_b)), 4, color, -1)

    return img_a, img_b

def stereo_rectify_uncalibrate(imgl,imgr,imgpoints_l,imgpoints_r,image_size):

    # Compute fundamental matrix
    F = compute_fundamental_matrix(imgpoints_l, imgpoints_r)

    # F, _ = cv2.findFundamentalMat(imgpoints_l,imgpoints_r) # compute fundamemtal matrix based on opencv's API
    # Compute epipoles
    e1, e2 = compute_epipole(F)

    print("Epipole shapes are ",e1.shape,e2.shape)

    points1 = np.column_stack((imgpoints_l,np.ones(len(imgpoints_l))))
    points2 = np.column_stack((imgpoints_r,np.ones(len(imgpoints_r))))
    # Compute homography
    H1, H2 = find_H1_H2(e2, F.T, imgr, points1, points2)
    # _, H1, H2 = cv2.stereoRectifyUncalibrated(imgpoints_l,imgpoints_r,F,image_size) # compute H1,H2 based on opencv's API
    # rectify images
    rectified_im1 = warpHomo(np.linalg.inv(H1), imgl)
    rectified_im2 = warpHomo(np.linalg.inv(H2), imgr)
    # rectify points
    new_cor1 = rectify_points(H1, points1)
    new_cor2 = rectify_points(H2, points2)

    return rectified_im1, rectified_im2, new_cor1, new_cor2, F
def stereo_rectify_calibrate(mtx_l,dist_l,mtx_r,dist_r,image_size,R,T):

    R1,R2,P1,P2,Q,ROI1,ROI2=cv2.stereoRectify(mtx_l,dist_l,mtx_r,dist_r,image_size,R,T)

    return R1, R2,P1,P2,Q,ROI1,ROI2
if __name__ == "__main__":
    objp = np.zeros((6 * 9, 3), np.float32)
    objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)
    img_size = (640, 480)
    path_l = 'Output/left/new_left01.jpg'  # use distiort1 to picutes and then get all corners' coordinate
    path_r = 'Output/right/new_right01.jpg'
    imgl=cv2.imread(path_l,cv2.IMREAD_COLOR)
    imgr=cv2.imread(path_r,cv2.IMREAD_COLOR)
    imgpoints_l=monocularCameraCalibration(path_l,objp,img_size)  # need to edit this function's rval
    imgpoints_r=monocularCameraCalibration(path_r,objp,img_size)  # neet to edit this function's rval
    imgpoints_l=imgpoints_l[0].reshape(-1,2)
    imgpoints_r=imgpoints_r[0].reshape(-1,2)
    rect_img1, rect_img2, epilines1, epilines2, F =stereo_rectify_uncalibrate(imgl,imgr,imgpoints_l,imgpoints_r,img_size)

    np.savetxt('left.txt',imgpoints_l,fmt='%f')
    np.savetxt('right.txt',imgpoints_r,fmt='%f')

    cv2.imshow("rect1,",rect_img1)
    cv2.imshow("rect2,",rect_img2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("Output/Rectified_left.jpg",rect_img1)
    cv2.imwrite("Output/Rectified_right.jpg", rect_img2)






