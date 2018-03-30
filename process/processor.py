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

    for k, v in question_areas.iteritems():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(
            region_arr, 0, 0, rp_size=(20, 24), rp_padding=((2,), (4,)))

        v.sub_regions = number_areas

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
