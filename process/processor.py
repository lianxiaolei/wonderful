# coding: utf-8

from core.img_process import *
from core.segmentation import *
import matplotlib.patches as patches
import matplotlib.pyplot as plt
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

    for k, v in question_areas.iteritems():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(region_arr, 0, 0)

        v.sub_regions = number_areas

    return question_areas


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

    for k, question_region in regions.iteritems():
        if layer in [0, 1]:
            cu.add_patch(patches.Rectangle(
                (question_region.get_x(), question_region.get_y()),
                question_region.get_width(), question_region.get_height(),
                linewidth=2, edgecolor='c', facecolor='none'
            ))
        if layer in [0, 2]:
            for j, number_region in question_region.get_sub_regions().iteritems():
                cu.add_patch(patches.Rectangle(
                    (question_region.get_x() + number_region.get_x(),
                     question_region.get_y() + number_region.get_y()),
                    number_region.get_width(), number_region.get_height(),
                    linewidth=1, edgecolor='y', facecolor='none'
                ))
    plt.show()


def save_region_as_jpg():
    pass


def save_questions(img, areas, base_fname):
    # 调用方式:save_questions(img, areas, file_name[file_name.find('/'): file_name.rfind('.')])
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
    # 调用方式:save_numbers(img, row_list, col_list, file_name[file_name.find('/'): file_name.rfind('.')])
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