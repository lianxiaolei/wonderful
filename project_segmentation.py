# coding: utf-8

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def read_img(file_name, color_inv_norm=True):
    """
    read a image from local file system
    :param file_name:
    :param color_inv_norm:
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if color_inv_norm:
        img = (255 - img) / 255.0
    else:
        img[img < 50] = 0
        img = img / 255.0

    return img


def get_areas(proj_list, epsilon=0):
    """

    :param proj_list:
    :param epsilon:
    :return:
    """
    area_list = []
    s = -1
    eps = 0
    for i in xrange(len(proj_list)):
        if not proj_list[i] == 0 and s == -1:
            s = i
        if not proj_list[i] == 0:
            eps = 0
        if proj_list[i] == 0 and not s == -1:
            e = i
            if eps >= epsilon or i == len(proj_list) - 1:
                area_list.append((s, e - eps))
                s = -1
                eps = 0
            else:
                eps += 1
    return area_list


def project_cut(img, row_eps, col_eps):
    """

    :param img:
    :param row_eps:
    :param col_eps:
    :return:
    """
    print img.shape
    row_proj = np.sum(img, axis=0)
    col_proj = np.sum(img, axis=1)

    row_list = get_areas(row_proj, epsilon=row_eps)
    col_list = get_areas(col_proj, epsilon=col_eps)

    areas = []
    i = 0
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            areas.append([x0, y0, x1 - x0, y1 - y0])

            tmp = img[y0: y1, x0: x1]
            tmp = cv2.resize(tmp,(28, 28))
            plt.imshow(tmp)
            plt.show()
            cv2.imwrite('tmp1/item%s.jpg' % i, tmp * 255)
            i += 1
    cv2.destroyAllWindows()

    # tmp = img[areas[10][1]: areas[10][1] + areas[10][3],
    #       areas[10][0]: areas[10][0] + areas[10][2]]
    # np.savetxt('tmp2.txt', tmp)
    # plt.imshow(tmp)
    # plt.show()
    # cv2.imwrite('tmp/question.jpg', tmp * 255)
    # cv2.destroyAllWindows()
    # sys.exit(0)

    plt.figure(0)
    plt.imshow(img, cmap='gray')
    cu = plt.gca()
    for i in xrange(len(areas)):
        cu.add_patch(
            patches.Rectangle(
                (areas[i][0], areas[i][1]),
                areas[i][2], areas[i][3],
                linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()


def run(file_name):
    img = read_img(file_name, color_inv_norm=False)

    np.savetxt('tmp.txt', img)
    # plt.show()
    # project_cut(img, row_eps=img.shape[1] / 30, col_eps=10)
    project_cut(img, row_eps=0, col_eps=0)


if __name__ == '__main__':

    # run('images/12.jpg')
    # run('images/24.jpg')
    # run('images/14.jpg')
    # run('images/13.png')
    # run('images/15.jpg')
    # run('images/16.jpg')

    # run('images/11.png')
    run('tmp/question.jpg')
