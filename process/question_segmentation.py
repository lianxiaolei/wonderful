# coding: utf-8

import cv2
from tool.img_process import *
from tool.segmentation import *
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os


def get_cut(img, row_eps, col_eps):
    question_areas = project_cut(img, row_eps, col_eps)

    for k, v in question_areas.iteritems():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(region_arr, 0, 0)

        v.sub_regions = number_areas

    return question_areas



def project_cut_old(img, row_eps, col_eps):
    """

    :param img:
    :param row_eps:
    :param col_eps:
    :return:
    """
    row_proj = np.sum(img, axis=0)
    col_proj = np.sum(img, axis=1)

    row_list = get_areas(row_proj, epsilon=row_eps)
    col_list = get_areas(col_proj, epsilon=col_eps)
    print '列分割：', row_list
    print '行分割：', col_list
    # questions area dict
    question_areas = get_area_dict(img, row_list, col_list)
    question_numbers_areas = {}
    for k, v in question_areas.iteritems():
        region_arr = region2ndarray(img, v)

        question_row_proj = get_hist(region_arr, axis=0)
        question_col_proj = get_hist(region_arr, axis=1)

        question_row_list = get_areas(question_row_proj, epsilon=row_eps)
        question_col_list = get_areas(question_col_proj, epsilon=col_eps)

        number_areas = get_area_dict(img, question_row_list, question_col_list)

        question_numbers_areas[k] = number_areas
