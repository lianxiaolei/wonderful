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


def get_area_dict(img, row_list, col_list, resize=False, display=False):
    """
    obtain the region dict with format {'x0_y0', region}
    :param img:
    :param row_list:
    :param col_list:
    :param resize:
    :param display:
    :return:
    """
    areas = OrderedDict()
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            # x0 x1-x0 y0 y1-y0 are abscissa width ordinate height
            # judge whether the area is only-black
            sub_img = img[y0: y1, x0: x1]
            # 去除较小的区域
            if np.sum(sub_img) <= 5:
                continue

            if resize:
                # sub_img = cv2.blur(sub_img, (4, 4))
                if x1 - x0 < y1 - y0:  # 铅直边较长
                    change_rate = (y1 - y0 - 24) / float((y1 - y0))
                    changed_width = int((x1 - x0) * (1 - change_rate))
                    if changed_width % 2 == 1:
                        changed_width += 1
                    if changed_width == 0:
                        changed_width = 2
                    pad = (24 - changed_width) / 2
                    padding = ((0,), (int(pad),))
                    # print(y1 - y0, x1 - x0, 1 - change_rate, changed_width, pad)
                    # plt.imshow(sub_img)
                    # plt.show()
                    sub_img = get_resize_padding_img(sub_img, size=(changed_width, 24), padding=padding)
                    kernel = np.ones((2, 2), np.uint8)
                    sub_img = cv2.dilate(sub_img, kernel, iterations=1)
                    if display:
                        plt.imshow(sub_img)
                        plt.show()
                else:  # 水平边较长
                    change_rate = (x1 - x0 - 24) / float((x1 - x0))
                    changed_height = int((y1 - y0) * (1 - change_rate))
                    if changed_height % 2 == 1:
                        changed_height += 1
                    if changed_height == 0:
                        changed_height = 2
                    pad = (24 - changed_height) / 2
                    padding = ((int(pad),), (0,))
                    # print(y1 - y0, x1 - x0, 1 - change_rate, changed_height, pad)
                    # plt.imshow(sub_img)
                    # plt.show()
                    sub_img = get_resize_padding_img(sub_img, size=(24, changed_height), padding=padding)
                    kernel = np.ones((2, 2), np.uint8)
                    sub_img = cv2.dilate(sub_img, kernel, iterations=1)
                    if display:
                        plt.imshow(sub_img)
                        plt.show()

            region = Region(x0, y0, x1 - x0, y1 - y0, sub_img)

            areas['%s_%s' % (x0, y0)] = region
    return areas


def project_cut(img, row_eps, col_eps, resize=False, display=False):
    """
    cut img with axis project
    :param img:
    :param row_eps:
    :param col_eps:
    :param resize:
    :param display:
    :return:
    """
    row_proj = np.sum(img, axis=0)
    col_proj = np.sum(img, axis=1)

    row_list = get_areas(row_proj, epsilon=row_eps)  # 横坐标list
    col_list = get_areas(col_proj, epsilon=col_eps)  # 纵坐标list

    # questions area dict
    areas = get_area_dict(
        img, row_list, col_list, resize=resize, display=display)

    return areas
