# coding: utf-8

from core.segmentation import *
from core.cnn import CNN
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os


def cut(img, row_eps, col_eps):
    """
    cut a image
    :param img:
    :param row_eps:
    :param col_eps:
    :return:
    """
    question_areas = project_cut(img, row_eps, col_eps)

    for k, v in question_areas.items():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(
            region_arr, 0, 0, rp_size=(20, 24), rp_padding=((2,), (4,)))

        v.sub_regions = number_areas

        # TODO recognize the content


    return question_areas


def save_region_as_jpg(fname, img, region, diastolic=True):
    """

    :param fname:
    :param img:
    :param region:
    :param diastolic:
    :param resize_padding:
    :return:
    """
    sub_img = get_region_img(img, region)

    if diastolic:
        cv2.imwrite(fname, sub_img * 255)
    else:
        cv2.imwrite(fname, sub_img)

    cv2.destroyAllWindows()


def cnn_model_maker(model_name, p_keep_conv=1.0, p_keep_hidden=1.0,
                    batch_size=128, test_size=256, epoch_time=3):
    """

    :param batch_size:
    :param test_size:
    :return:
    """
    cnn = CNN(p_keep_conv=p_keep_conv, p_keep_hidden=p_keep_hidden,
              batch_size=batch_size, test_size=test_size, epoch_time=epoch_time)

    mnist = input_data.read_data_sets('./', one_hot=True)

    train_x, train_y, test_x, test_y = \
        mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    cnn.fit(train_x, train_y, test_x, test_y)

    cnn.save(model_name)
