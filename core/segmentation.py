# coding: utf-8

from bean.region import Region
from collections import OrderedDict
from core.img_process import *


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
    for i in range(len(proj_list)):
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


def get_area_dict(img, row_list, col_list, rp_size=None, rp_padding=None):
    """
    obtain the region dict with format {'x0_y0', region}
    :param img:
    :param row_list:
    :param col_list:
    :param rp_size:
    :param rp_padding:
    :return:
    """
    areas = OrderedDict()
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            # x0 x1-x0 y0 y1-y0 are abscissa width ordinate height
            # judge whether the area is only-black
            sub_img = img[y0: y1, x0: x1]
            if np.sum(sub_img) <= 50: continue
            if rp_size and rp_padding:
                region = Region(
                    x0, y0, x1 - x0, y1 - y0, get_resize_padding_img(
                        sub_img, size=rp_size, padding=rp_padding))
            else:
                region = Region(x0, y0, x1 - x0, y1 - y0, sub_img)
            areas['%s_%s' % (x0, y0)] = region
    return areas


def project_cut(img, row_eps, col_eps, rp_size=None, rp_padding=None):
    """
    cut img with axis project
    :param img:
    :param row_eps:
    :param col_eps:
    :param rp_size:
    :param rp_padding:
    :return:
    """
    row_proj = np.sum(img, axis=0)
    col_proj = np.sum(img, axis=1)

    row_list = get_areas(row_proj, epsilon=row_eps)  # 横坐标list
    col_list = get_areas(col_proj, epsilon=col_eps)  # 纵坐标list

    # questions area dict
    areas = get_area_dict(
        img, row_list, col_list, rp_size=rp_size, rp_padding=rp_padding)

    return areas
