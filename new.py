# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt


def detect(img):
    """
    :param img:
    :return:
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(gray.shape)

    dilation = preprocess(gray)

    region = find_text_region(dilation)

    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(gray):
    """
    :param img:
    :return:
    """
    sobel = cv2.Sobel(gray, cv2.CV_8U, dx=1, dy=0, ksize=3)
    plt.imshow(sobel)
    plt.show()

    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)

    print(ret)
    plt.imshow(binary)
    plt.show()

    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 9))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 6))

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 18))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (72, 12))

    dilation = cv2.dilate(binary, element2, iterations=1)
    erosion = cv2.erode(dilation, element1, iterations=1)
    dilation1 = cv2.dilate(erosion, element2, iterations=3)
    erosion1 = cv2.erode(dilation1, element2, iterations=3)
    # dilation1 = dilation
    plt.imshow(erosion1)
    plt.show()

    return dilation1


def find_text_region(dilation):
    """
    :param dilation:
    :return:
    """
    region = []

    # 1. 查找轮廓
    image, contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if (area < 1000):
            continue

        # 轮廓近似，作用很小
        epsilon = 0.001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)
        print("rect is: ")
        print(rect)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])

        # 筛选那些太细的矩形，留下扁的
        if height > width * 1.2:
            continue

        region.append(box)

    return region


if __name__ == '__main__':
    # img = cv2.imread('images/bk.jpg')
    # img = cv2.imread('images/000.jpg')
    img = cv2.imread('images/bk1.jpg')
    # img = cv2.imread('images/0.png')
    detect(img)
