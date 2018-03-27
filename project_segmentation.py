# coding: utf-8

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


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

    :param proj_list:
    :param epsilon:
    :return:
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
    areas = []
    for x0, x1 in row_list:
        for y0, y1 in col_list:
            areas.append([x0, y0, x1 - x0, y1 - y0])

    return areas, row_list, col_list


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


def run(file_name):
    img = read_img(file_name, color_inv_norm=True)
    np.savetxt('tmp1.txt', img)

    # 截取图片保存试题
    # areas, row_list, col_list = \
    #     project_cut(img, row_eps=img.shape[1] / 30, col_eps=10)
    #
    # save_questions(img, areas, file_name[file_name.find('/'): file_name.rfind('.')])

    # 截取试题保存数字
    areas, row_list, col_list = project_cut(img, row_eps=0, col_eps=0)
    save_numbers(img, row_list, col_list, file_name[file_name.find('/'): file_name.rfind('.')])

    plt.figure(0)
    plt.imshow(img, cmap='gray')
    cu = plt.gca()
    for i in xrange(len(areas)):
        cu.add_patch(
            patches.Rectangle(
                (areas[i][0], areas[i][1]),
                areas[i][2], areas[i][3],
                linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()


def get_questions(file_name):
    img = read_img(file_name, color_inv_norm=True)
    # 截取图片保存试题
    areas, row_list, col_list = \
        project_cut(img, row_eps=img.shape[1] / 30, col_eps=10)

    save_questions(img, areas, file_name[file_name.find('/'): file_name.rfind('.')])


def get_numbers(file_name):
    img = read_img(file_name, color_inv_norm=False)
    # 截取试题保存数字
    areas, row_list, col_list = project_cut(img, row_eps=0, col_eps=0)

    save_numbers(img, row_list, col_list, file_name[file_name.find('/'): file_name.rfind('.')])

if __name__ == '__main__':

    # run('images/12.jpg')
    # run('images/24.jpg')
    # run('images/14.jpg')
    # run('images/13.png')
    # run('images/15.jpg')
    # run('images/16.jpg')

    # get_questions('images/cgd.jpg')
    for i in xrange(10):
        print i
        get_numbers('questions/cgd%s.jpg' % i)
