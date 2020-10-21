
import numpy as np
from scipy import optimize as opt


# get normalized matrix
def normalizing_input_data(coor_data):
    x_avg = np.mean(coor_data[:, 0])
    y_avg = np.mean(coor_data[:, 1])
    sx = np.sqrt(2) / np.std(coor_data[:, 0])
    sy = np.sqrt(2) / np.std(coor_data[:, 1])

    norm_matrix = np.matrix([[sx, 0, -sx * x_avg],
                             [0, sy, -sy * y_avg],
                             [0, 0, 1]])
    return norm_matrix


# get initial homography
def get_initial_H(imgpoints, objpoints_x_y):
    # get normalized matrix
    pic_norm_mat = normalizing_input_data(imgpoints)
    real_norm_mat = normalizing_input_data(objpoints_x_y)

    M = []
    for i in range(len(imgpoints)):
        # transform as homogenous coordinate
        single_pic_coor = np.array([imgpoints[i][0], imgpoints[i][1], 1])
        single_real_coor = np.array([objpoints_x_y[i][0], objpoints_x_y[i][1], 1])

        # normlize the coordinate
        pic_norm = np.dot(pic_norm_mat, single_pic_coor)
        real_norm = np.dot(real_norm_mat, single_real_coor)

        # construct M matrix
        M.append(np.array([-real_norm.item(0), -real_norm.item(1), -1,
                      0, 0, 0,
                      pic_norm.item(0) * real_norm.item(0), pic_norm.item(0) * real_norm.item(1), pic_norm.item(0)]))

        M.append(np.array([0, 0, 0,
                      -real_norm.item(0), -real_norm.item(1), -1,
                      pic_norm.item(1) * real_norm.item(0), pic_norm.item(1) * real_norm.item(1), pic_norm.item(1)]))

    # use SVD tu compute h in equation M * h = 0
    U, S, VT = np.linalg.svd((np.array(M, dtype='float')).reshape((-1, 9)))
    # get  singular vector of  minor singular value  as result
    H = VT[-1].reshape((3, 3))
    H = np.dot(np.dot(np.linalg.inv(pic_norm_mat), H), real_norm_mat)
    H /= H[-1, -1]

    return H


# get difference between estimated coordinate and real coordinate
def value(H, pic_coor, real_coor):
    Y = np.array([])
    for i in range(len(real_coor)):
        single_real_coor = np.array([real_coor[i, 0], real_coor[i, 1], 1])
        U = np.dot(H.reshape(3, 3), single_real_coor)
        U /= U[-1]
        Y = np.append(Y, U[:2])

    Y_NEW = (pic_coor.reshape(-1) - Y)

    return Y_NEW


# get jacobian matrix
def jacobian(H, pic_coor, real_coor):
    J = []
    for i in range(len(real_coor)):
        sx = H[0]*real_coor[i][0] + H[1]*real_coor[i][1] +H[2]
        sy = H[3]*real_coor[i][0] + H[4]*real_coor[i][1] +H[5]
        w = H[6]*real_coor[i][0] + H[7]*real_coor[i][1] +H[8]
        w2 = w * w

        J.append(np.array([real_coor[i][0]/w, real_coor[i][1]/w, 1/w,
                           0, 0, 0,
                           -sx*real_coor[i][0]/w2, -sx*real_coor[i][1]/w2, -sx/w2]))

        J.append(np.array([0, 0, 0,
                           real_coor[i][0]/w, real_coor[i][1]/w, 1/w,
                           -sy*real_coor[i][0]/w2, -sy*real_coor[i][1]/w2, -sy/w2]))

    return np.array(J)


# use Levenberg Marquart algorithm to refine H
def refine_H(pic_coor, real_coor, initial_H):
    initial_H = np.array(initial_H)
    final_H = opt.leastsq(value,
                          initial_H,
                          Dfun=jacobian,
                          args=(pic_coor, real_coor))[0]

    final_H /= np.array(final_H[-1])
    return final_H


# get refined H
def get_homography(imgpoints, objpoints_x_y):
    refined_homographies =[]

    error = []
    for i in range(len(imgpoints)):
        initial_H = get_initial_H(imgpoints[i], objpoints_x_y[i])
        final_H = refine_H(imgpoints[i], objpoints_x_y[i], initial_H)
        refined_homographies.append(final_H)

    return np.array(refined_homographies)