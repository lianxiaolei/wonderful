# coding: utf-8

from core.segmentation import *
import numpy as np
import os


def cut(img, row_eps, col_eps):
    """
    cut a image
    :param img:
    :param row_eps:
    :param col_eps:
    :return:
    """
    question_areas = project_cut(img, row_eps, col_eps)

    for k, v in question_areas.items():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(
            region_arr, 0, 0, rp_size=(20, 24), rp_padding=((2,), (4,)))

        v.sub_regions = number_areas

        # TODO recognize the content


    return question_areas


def save_region_as_jpg(fname, img, region, diastolic=True, resize_padding=False):
    """

    :param fname:
    :param img:
    :param region:
    :param diastolic:
    :param resize_padding:
    :return:
    """
    sub_img = get_region_img(img, region)

    if diastolic:
        cv2.imwrite(fname, sub_img * 255)
    else:
        cv2.imwrite(fname, sub_img)

    cv2.destroyAllWindows()


def cnn_model_maker():

