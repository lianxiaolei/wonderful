# coding: utf-8

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

def read_img(file_name):
    """
    read a image from local file system
    :param file_name:
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

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
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            areas.append([x0, y0, x1 - x0, y1 - y0])

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