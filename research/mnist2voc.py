# coding: utf-8

import cv2
import numpy as np
import os
from xml.dom import minidom


def find_text_region(img):
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
    # cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

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
    xx = 0
    for box in region:
        if not xx == 0:
            break
        xx += 1
        # cv2.drawContours(img, [box], 0, (0, 255, 0), 2)
        cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]), (255, 0, 0), 1)
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.imshow('img', img)

    # cv2.imwrite("contours.png", img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return region[0]


def mnist2voc(in_path, out_path, size=(224, 224)):
    """

    :param in_path:
    :param out_path:
    :param size:
    :return:
    """
    # make dirs
    anno_dir = os.path.join(out_path, 'Annotations')
    if 'Annotations' not in os.listdir(out_path):
        os.mkdir(anno_dir)

    set_dir = os.path.join(out_path, 'ImageSets')
    if 'ImageSets' not in os.listdir(out_path):
        os.mkdir(set_dir)

    jpg_dir = os.path.join(out_path, 'JPEGImages')
    if 'JPEGImages' not in os.listdir(out_path):
        os.mkdir(jpg_dir)

    root_list = ['source', 'filename',
                 'folder', 'owner', 'size', 'segmented', 'object']

    # iterate files
    for cls in os.listdir(in_path):
        file_dir = os.path.join(in_path, cls)
        for fname in os.listdir(file_dir):
            lfname = os.path.join(file_dir, fname)
            img = cv2.imread(lfname)
            if size:
                img = cv2.resize(img, dsize=size)

            width = img.shape[1]
            height = img.shape[0]
            depth = 3

            x, y, w, h = find_text_region(img)

            cv2.imwrite(os.path.join(jpg_dir, fname), img)

            impl = minidom.getDOMImplementation()
            doc = impl.createDocument(None, None, None)

            root = doc.createElement('annotation')

            filename = doc.createElement('filename')
            filename.appendChild(doc.createTextNode(fname))

            sz = doc.createElement('size')
            width = doc.createElement('width')
            width.appendChild(doc.createTextNode(str(width)))
            height = doc.createElement('height')
            height.appendChild(doc.createTextNode(str(height)))
            depth = doc.createElement('depth')
            depth.appendChild(doc.createTextNode(str(depth)))
            sz.appendChild(width)
            sz.appendChild(height)
            sz.appendChild(depth)

            obj = doc.createElement('object')
            name = doc.createElement('name')
            name.appendChild(doc.createTextNode(cls))
            bndbox = doc.createElement('bndbox')
            xmin = doc.createElement('xmin')
            xmin.appendChild(doc.createTextNode(str(x)))
            ymin = doc.createElement('ymin')
            ymin.appendChild(doc.createTextNode(str(y)))
            xmax = doc.createElement('xmax')
            xmax.appendChild(doc.createTextNode(str(x + w)))
            ymax = doc.createElement('ymax')
            ymax.appendChild(doc.createTextNode(str(y + h)))
            bndbox.appendChild(xmin)
            bndbox.appendChild(xmax)
            bndbox.appendChild(ymin)
            bndbox.appendChild(ymax)
            obj.appendChild(name)
            obj.appendChild(bndbox)

            root.appendChild(sz)
            root.appendChild(obj)

            print(root.toprettyxml(encoding='utf8'))


if __name__ == '__main__':
    mnist2voc('F:\\num_ocr', 'F:\\outpu')
