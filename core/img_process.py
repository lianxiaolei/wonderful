# coding: utf-8

import numpy as np
import cv2


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
