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
    print(gray.shape)

    gray[gray > 120] = 255
    gray[gray <= 120] = 0

    region = find_text_region(gray)

    for box in region:
        cv2.drawContours(img, [box], 0, (0, 255, 0), 2)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    # cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def preprocess(gray):
    """
    :param img:
    :return:
    """
    sobel = cv2.Sobel(gray, cv2.CV_8U, dx=1, dy=0, ksize=3)
    # plt.imshow(sobel)
    # plt.show()

    ret, binary = cv2.threshold(sobel, 0, 255, cv2.THRESH_OTSU)

    print(ret)
    plt.imshow(binary)
    plt.title('ori binary')
    plt.show()

    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 6))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 4))
    #
    # element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (90, 18))
    # element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (72, 12))

    dilation = cv2.dilate(binary, element2, iterations=1)
    plt.imshow(dilation)
    plt.title('dil0')
    plt.show()

    erosion = cv2.erode(dilation, element1, iterations=1)
    plt.imshow(erosion)
    plt.title('ero0')
    plt.show()

    median = cv2.medianBlur(erosion, 33)
    plt.imshow(median)
    plt.title('median')
    plt.show()

    # mask = 255 - cv2.adaptiveThreshold(
    #     sobel, 255,
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
    #     3, 3)
    # plt.imshow(mask)
    # plt.title('mask')
    # plt.show()

    dilation1 = cv2.dilate(median, element2, iterations=2)
    plt.imshow(dilation1)
    plt.title('dil1')
    plt.show()

    erosion1 = cv2.erode(dilation1, element2, iterations=1)
    plt.imshow(erosion1)
    plt.title('ero1')
    plt.show()

    dilation2 = cv2.dilate(erosion1, element2, iterations=2)
    plt.imshow(dilation2)
    plt.title('dil2')
    plt.show()

    return dilation2


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
        # if (area < 1000):
        #     continue

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
        # if height > width * 1.2:
        #     continue

        region.append(box)

    return region


def dododo(fname):
    img = 255 - cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.dilate(img, kernel)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # img = img / 255.0
    img = cv2.resize(img, (42, 42))

    img = np.pad(img, ((3,), (3,)), mode='constant')
    return img


def resave_img(base_path, target_bpath):
    """
    :param img:
    :return:
    """
    symb_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', 'times', 'div', '=', '(', ')']
    # symb_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', 'times', 'div', '=']

    symbols = os.listdir(base_path)
    for symbol in symbols:
        calc = 0

        if symbol not in symb_list: continue
        print('now operate: ', symbol)

        jpgs = os.listdir(os.path.join(base_path, symbol))

        tmp_path = os.path.join(target_bpath, symbol)
        if not os.path.exists(tmp_path):
            os.mkdir(tmp_path)

        for jpg in jpgs:
            calc += 1
            if calc > 6000:
                print('the symbol %s is more than 6000' % symbol)
                break

            fname = os.path.join(base_path, symbol, jpg)
            img = dododo(fname)

            # cv2.imwrite(os.path.join(tmp_path, jpg), img)

            plt.imshow(img)
            plt.show()
        print(os.path.join(base_path, symbol), '-->', os.path.join(target_bpath, symbol))


def mser(img):
    mser = cv2.MSER_create(_delta=2, _min_area=2000, _max_variation=0.7)
    regions, boxes = mser.detectRegions(img)
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    cv2.imshow("img2", img)
    cv2.waitKey()


def remove_lines(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    # plt.imshow(edges)
    # plt.show()

    lines = cv2.HoughLines(edges, 0.8, np.pi / 180, 200)
    print(lines)

    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imwrite('houghlines.jpg', img)
    print('done')


if __name__ == '__main__':
    # img = cv2.imread('images/000.jpg')
    # img = cv2.imread('images/bk.jpg')
    img = cv2.imread('../images/bk1.jpg')
    img = cv2.imread('../1234.jpg')
    # img = cv2.imread('images/mine7.jpg')
    # img = cv2.imread('images/lyq.jpg')
    # img = cv2.imread('images/grid.jpg')
    # img = cv2.imread('images/grid1.jpg')
    # img = cv2.imread('images/0.png')

    detect(img)
    # remove_lines(img)
    sys.exit(13)

    # resave_img('/Users/imperatore/tmp/extracted_images',
    #            '/Users/imperatore/tmp/pre_ocr')
    #
    # import sys
    # sys.exit(13)

    img = 255 - cv2.imread('F:/datas/extracted_images/(/(_104771.jpg', cv2.IMREAD_GRAYSCALE)
    # img = cv2.imread('F:/datas/pre_ocr/15/(_1000.jpg', cv2.IMREAD_GRAYSCALE)

    # img = cv2.resize(img, (22, 22))
    plt.imshow(img)
    plt.title('origin with inv color')
    plt.show()
