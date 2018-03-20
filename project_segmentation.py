# coding: utf-8

import selectivesearch
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def read_img(file_name):
    """
    read a image from local file system
    :param file_name:
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
    h = img.shape[0]
    w = img.shape[1]
    # if h > 3000 or w > 3000:
    #     w = w / 10
    #     h = h / 10
    #     img = cv2.resize(img, (w, h))
    ret, img = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)
    img[img == 255] = 1
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


def project_cut(img):
    """

    :param img:
    :return:
    """
    print img.shape
    row_proj = np.sum(img, axis=0)
    col_proj = np.sum(img, axis=1)
    # print row_proj
    # print col_proj

    row_list = get_areas(row_proj, epsilon=img.shape[1] / 30)
    col_list = get_areas(col_proj, epsilon=10)
    print row_list
    print col_list

    areas = []
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            areas.append([x0, y0, x1 - x0, y1 - y0])

    tmp = img[areas[10][1]: areas[10][1] + areas[10][3],
          areas[10][0]: areas[10][0] + areas[10][2]]

    plt.imshow(tmp)
    plt.show()
    sys.exit(0)

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
    img = read_img(file_name)

    project_cut(img)
    

if __name__ == '__main__':
    run('images/11.png')
    run('images/12.jpg')
    run('images/24.jpg')
    run('images/14.jpg')
    run('images/13.png')
    run('images/15.jpg')
    run('images/16.jpg')