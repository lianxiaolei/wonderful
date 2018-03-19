# coding: utf-8

# import tflearn
import selectivesearch
import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import os
import sys


def get_common_square(r0, r1):
    minx = max(r0[0], r1[0])
    miny = max(r0[1], r1[1])
    maxx = min(r0[0] + r0[2], r1[0] + r1[2])
    maxy = min(r0[1] + r0[3], r1[1] + r1[3])
    return (maxy - miny) * (maxx - minx)


def nms(r):
    for i in xrange(len(r)):
        if r[i][2] - r[i][0] > 200 or r[i][3] - r[i][1] > 100:
            continue
        for j in xrange(len(r)):
            if r[j][0]:
                pass
            np.nonzero()


# IOU Part 1
def if_intersection(xmin_a, xmax_a, ymin_a, ymax_a, xmin_b, xmax_b, ymin_b, ymax_b):
    if_intersect = False
    # 通过四条if来查看两个方框是否有交集。如果四种状况都不存在，我们视为无交集
    if xmin_a < xmax_b <= xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_a <= xmin_b < xmax_a and (ymin_a < ymax_b <= ymax_a or ymin_a <= ymin_b < ymax_a):
        if_intersect = True
    elif xmin_b < xmax_a <= xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    elif xmin_b <= xmin_a < xmax_b and (ymin_b < ymax_a <= ymax_b or ymin_b <= ymin_a < ymax_b):
        if_intersect = True
    else:
        return False
    # 在有交集的情况下，我们通过大小关系整理两个方框各自的四个顶点， 通过它们得到交集面积
    if if_intersect:
        x_sorted_list = sorted([xmin_a, xmax_a, xmin_b, xmax_b])
        y_sorted_list = sorted([ymin_a, ymax_a, ymin_b, ymax_b])
        x_intersect_w = x_sorted_list[2] - x_sorted_list[1]
        y_intersect_h = y_sorted_list[2] - y_sorted_list[1]
        area_inter = x_intersect_w * y_intersect_h
        return area_inter


def get_IOU(r0, r1):
    ver0 = [r0[0], r0[0] + r0[2], r0[1], r0[1] + r0[3]]
    ver1 = [r1[0], r1[0] + r1[2], r1[1], r1[1] + r1[3]]
    ver_long = ver0 + ver1
    area_inter = if_intersection(*ver_long)
    if area_inter:
        area0 = r0[2] * r0[3]
        area1 = r1[2] * r1[3]
        IOU = float(area_inter) / (area0 + area1 - area_inter)
        return IOU, area0, area1
    return 0, 0, 0


def selective_search(file_name):
    img = cv2.imread(file_name)
    h = img.shape[0]
    w = img.shape[1]

    if h > 1000 and w > 1000:
        w = w / 10
        h = h / 10
        img = cv2.resize(img, (w, h))
    ret, img1 = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY_INV)

    # 去除大片的空白区域
    # img1 = img1[np.sum(np.sum(img1, axis=1), axis=1) != 0,:,:]

    img_lbl, regions = selectivesearch.selective_search(
        img1, scale=400, sigma=1.8, min_size=100)
    print 'region number', len(regions)
    print regions

    sort_key = lambda item: -item['rect'][2] * item['rect'][3]

    regions = sorted(regions, key=sort_key)
    candidates = []
    for i in xrange(len(regions) - 1):
        correct = regions[i]['rect']
        if regions[i]['rect'] in candidates:
            continue
        if abs(regions[i]['rect'][2] - w) < 100:
            continue
        for j in xrange(i + 1, len(regions)):
            if regions[j]['rect'] in candidates:
                # print 'there is a repeat region'
                continue
            if abs(regions[j]['rect'][2] - w) < 100:
                continue
            IOU, square0, square1 = get_IOU(regions[i]['rect'], regions[j]['rect'])
            # print regions[i]['rect'], regions[j]['rect'], square0, square1, IOU
            # print regions[i]['rect'][2], regions[j]['rect'][2], w

            if IOU > 0:
                if square1 < square0:
                    correct = regions[j]['rect']
        candidates.append(correct)

    # draw
    plt.figure(8)
    plt.imshow(img1)
    cu = plt.gca()
    for i in xrange(len(candidates)):
        # print len(candidates)
        # print candidates
        # if regions[i]['rect'][2] - regions[i]['rect'][0] > 200 or \
        #         regions[i]['rect'][3] - regions[i]['rect'][1] > 200:
        #     continue
        cu.add_patch(
            patches.Rectangle(
                (candidates[i][0], candidates[i][1]),
                candidates[i][2], candidates[i][3],
                linewidth=1, edgecolor='r', facecolor='none'))
    plt.show()


if __name__ == '__main__':
    # selective_search('images/11.png')
    selective_search('images/12.jpg')
    selective_search('images/24.jpg')
    selective_search('images/13.png')
    # selective_search('images/14.jpg')
