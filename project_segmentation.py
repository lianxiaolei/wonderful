# coding: utf-8

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from core.img_process import *
from core.segmentation import *
from process.processor import *
import os
import sys


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    img = read_img(file_name, color_inv_norm=True)
    regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10)
    show_all_regions(img, regions, layer=0)


if __name__ == '__main__':

    # run('images/12.jpg')
    run('images/24.jpg')
    # run('images/14.jpg')
    # run('images/13.png')
    # run('images/15.jpg')
    # run('images/16.jpg')

