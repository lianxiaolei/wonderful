# coding: utf-8

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
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


def get_areas(proj_list, epsilon=0):
    """
    cut the image with the histgram
    :param proj_list: histgram
    :param epsilon: gap length
    :return: the divided areas
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
        if i == len(proj_list) - 1 and not proj_list[i] == 0:
            area_list.append((s, i))
    return area_list


def get_area_dict(img, row_list, col_list):
    areas = {}
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            # x0 x1-x0 y0 y1-y0 are abscissa width ordinate height
            # judge whether the area is only-black
            if np.sum(img[y0: y1, x0: x1]) == 10: continue

            areas['%s%s' % (x0, y0)] = [x0, y0, x1 - x0, y1 - y0]
    return areas


def project_cut(img, row_eps, col_eps):
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
        region = img[v[1]: v[1] + v[3], v[0]: v[0] + v[2]]
        question_row_proj = np.sum(region, axis=0)
        question_col_proj = np.sum(region, axis=1)
        question_row_list = get_areas(question_row_proj, epsilon=row_eps)
        question_col_list = get_areas(question_col_proj, epsilon=col_eps)
        number_areas = get_area_dict(img, question_row_list, question_col_list)

        question_numbers_areas[k] = number_areas





def get_questions(img, areas, base_fname):
    i = 0
    for area in areas:
        tmp = img[area[1]: area[1] + area[3],
              area[0]: area[0] + area[2]]
        if not os.path.isdir('questions'):
            os.mkdir('questions')
        cv2.imwrite('questions/%s%s.jpg' % (base_fname, i), tmp * 255)
        cv2.destroyAllWindows()
        i += 1

def get_numbers(img, row_list, col_list, base_fname):
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
