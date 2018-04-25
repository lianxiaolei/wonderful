# coding: utf-8

import numpy as np
import cv2
import matplotlib.pyplot as plt
# from process.processor import *
import matplotlib.patches as patches
import random

# img = cv2.imread("images/111.jpg", cv2.IMREAD_GRAYSCALE)
img = cv2.imread("images/cz.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("images/bk.jpg", cv2.IMREAD_GRAYSCALE)
# img = cv2.imread("images/mine10.jpg", cv2.IMREAD_GRAYSCALE)
img = 255 - img

img1 = img.copy()
img1[img1 < 128] = 0
img1 = img1 / 255.0

plt.imshow(img)
plt.show()

mask = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 7)
plt.imshow(mask)
plt.show()

kernel = np.ones((5, 5), np.uint8)

img_mask = img * (1 - mask)


img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
img_mask = cv2.morphologyEx(img_mask, cv2.MORPH_OPEN, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)

img_mask[img_mask > 0] = 1

img[img < 128] = 0
img = img * img_mask
img = img / 255.0

# regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10, display=True)
# regions_recognition(regions, 'model/Test_CNN_Model.ckpt')

plt.imshow(img1)
plt.show()
plt.imshow(img)
plt.show()
