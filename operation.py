# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random

img = cv2.imread('images/111.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img)
plt.show()
cv2.imshow(img)
frame = 2
bs = cv2.createBackgroundSubtractorKNN(
    detectShadows=True)  # 背景减除器，设置阴影检测
fg_mask = bs.apply(frame)

th = cv2.threshold(fg_mask.copy(), 244, 255, cv2.THRESH_BINARY)[1]
dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 3)), iterations=2)
# 获取所有检测框
image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

for c in contours:
    # 获取矩形框边界坐标
    x, y, w, h = cv2.boundingRect(c)
    # 计算矩形框的面积
    area = cv2.contourArea(c)
    if 500 < area < 3000:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

cv2.imshow("detection", frame)
cv2.imshow("back", dilated)
cv2.waitKey()
cv2.destroyAllWindows()
