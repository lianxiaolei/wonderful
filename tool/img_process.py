# coding: utf-8

import numpy as np
import cv2
import os


def read_img(file_name, color_inv_norm=True):
    """
    read a image from local file system
    :param file_name:
    :param color_inv_norm:
    :return:
    """
    img = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)

    if color_inv_norm:
        img = 255 - img
        img[img < 150] = 0
        img = img / 255.0
    else:
        img[img < 50] = 0
        img = img / 255.0

    return img


def region2ndarray(img, region):
    """
    convert a region of img to ndarray
    :param img:
    :param region:
    :return:
    """
    array = img[region.get_y(): region.get_y() + region.get_height(),
            region.get_x(): region.get_x() + region.get_width()]
    return array


def get_hist(img, axis=0):
    """
    return the hist of img with axis
    :param img:
    :param axis:
    :return:
    """
    return np.sum(img, axis=axis)


def save_region_as_jpg():
    pass


def save_questions(img, areas, base_fname):
    i = 0
    for area in areas:
        tmp = img[area[1]: area[1] + area[3],
              area[0]: area[0] + area[2]]
        if not os.path.isdir('questions'):
            os.mkdir('questions')
        cv2.imwrite('questions/%s%s.jpg' % (base_fname, i), tmp * 255)
        cv2.destroyAllWindows()
        i += 1


def save_numbers(img, row_list, col_list, base_fname):
    i = 0
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            tmp = img[y0: y1, x0: x1]
            tmp = cv2.resize(tmp, (20, 24))
            tmp = np.pad(tmp, ((2,), (4,)), mode='constant')
            if not os.path.isdir('numbers'):
                os.mkdir('numbers')
            cv2.imwrite('numbers/%s%s.jpg' % (base_fname, i), tmp * 255)
            i += 1
            cv2.destroyAllWindows()
            # plt.imshow(tmp)
            # plt.show()