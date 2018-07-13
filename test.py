# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys


def detect(img):
    """
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray[gray > 120] = 255
    gray[gray <= 120] = 0

    # region = find_text_region(gray)

    region = []

    # 1. 查找轮廓
    image, contours, hierarchy = cv2.findContours(gray, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 画图像的边轮廓
    cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        # if (area < 1000):
        #     continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        # rect = cv2.minAreaRect(cnt)
        rect = cv2.boundingRect(cnt)
        box = rect
        print("rect is: ")
        print(rect)

        # box是四个点的坐标
        # box = cv2.boxPoints(rect)
        # box = np.int0(box)
        #
        # # 计算高和宽
        # height = abs(box[0][1] - box[2][1])
        # width = abs(box[0][0] - box[2][0])

        region.append(box)

    print('resgion length', len(region))
    for box in region:
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 1)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    # cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':

    img = cv2.imread('1234.jpg')

    detect(img)
