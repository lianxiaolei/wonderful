# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches


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


def get_region_img(img, region):
    """

    :param img:
    :param region:
    :return:
    """
    return img[
           region.get_y(): region.get_y() + region.get_height(),
           region.get_x(): region.get_x() + region.get_width()]


def get_resize_padding_img(img, size, padding):
    """

    :param img:
    :param size:
    :param padding:
    :return:
    """
    if size and padding:
        sub_img = cv2.resize(img, size)
        sub_img = np.pad(sub_img, padding, mode='constant')
    else:
        sub_img = cv2.resize(img, (28, 28))
    return sub_img


def show_all_regions(img, regions, layer=0):
    """
    show all question regions and number regions with matplotlib
    :param img:
    :param regions:
    :param layer:
    :return:
    """
    plt.figure(0)
    plt.imshow(img, cmap='gray')
    cu = plt.gca()

    for i, question_region in regions.items():
        if layer in [0, 1]:
            cu.add_patch(patches.Rectangle(
                (question_region.get_x(), question_region.get_y()),
                question_region.get_width(), question_region.get_height(),
                linewidth=2, edgecolor='c', facecolor='none'
            ))
        if layer in [0, 2]:
            for j, number_region in question_region.get_sub_regions().items():
                cu.add_patch(patches.Rectangle(
                    (question_region.get_x() + number_region.get_x(),
                     question_region.get_y() + number_region.get_y()),
                    number_region.get_width(), number_region.get_height(),
                    linewidth=1, edgecolor='y', facecolor='none'
                ))
    plt.show()


def save_all_regions(regions, dir_name, layer=0):
    """
    save regions as image
    :param regions:
    :param dir_name:
    :param layer:
    :return:
    """
    k = 0
    for i, question_region in regions.items():
        if layer in [0, 1]:
            cv2.imwrite('%s/%s.jpg' % (dir_name[0], k), question_region.get_img() * 255.0)
            cv2.destroyAllWindows()
            k += 1
        if layer in [0, 2]:
            l = 0
            for j, number_region in question_region.get_sub_regions().items():
                cv2.imwrite('%s/%s.jpg' % (dir_name[1], l), number_region.get_img() * 255.0)
                cv2.destroyAllWindows()
                l += 1
