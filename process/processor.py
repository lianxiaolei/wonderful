# coding: utf-8

from core.segmentation import *
from core.cnn import CNN
from tensorflow.examples.tutorials.mnist import input_data
import os
import cv2


def cut(img, row_eps, col_eps, display=False):
    """
    cut a image
    :param img:
    :param row_eps:
    :param col_eps:
    :return:
    """
    question_areas = project_cut(img, row_eps, col_eps)
    # show_all_regions(img, question_areas, layer=1)
    for k, v in question_areas.items():
        region_arr = region2ndarray(img, v)

        number_areas = project_cut(
            region_arr, 0, 0, resize=True, display=display)

        v.sub_regions = number_areas

    return question_areas


def save_region_as_jpg(fname, img, region, diastolic=True):
    """

    :param fname:
    :param img:
    :param region:
    :param diastolic:
    :return:
    """
    sub_img = get_region_img(img, region)

    if diastolic:
        cv2.imwrite(fname, sub_img * 255)
    else:
        cv2.imwrite(fname, sub_img)

    cv2.destroyAllWindows()


def get_extra_data(base_path):
    """

    :param dir_name:
    :return:
    """
    nums = os.listdir(base_path)
    train_data = []
    train_label = []
    lbl = [[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]]
    for num in nums:
        jpgs = os.listdir(os.path.join(base_path, num))
        for jpg in jpgs:
            fname = os.path.join(base_path, num, jpg)
            pic = read_img(fname, color_inv_norm=False)
            train_data.append(pic)
            train_label.append(lbl[int(num)])
    train_data = np.array(train_data)
    train_label = np.array(train_label)
    # print(train_data.shape, train_label.shape)
    # print(train_data)
    # print(np.argmax(train_label, axis=1))
    return train_data, train_label


def alg_train(model_name, p_keep_conv=1.0, p_keep_hidden=1.0,
              batch_size=128, test_size=256, epoch_time=3):
    """
    :param model_name:
    :param p_keep_conv:
    :param p_keep_hidden:
    :param batch_size:
    :param test_size:
    :param epoch_time
    :return:
    """
    print('initializing CNN model')
    cnn = CNN(p_keep_conv=p_keep_conv, p_keep_hidden=p_keep_hidden,
              batch_size=batch_size, test_size=test_size, epoch_time=epoch_time)
    print('CNN has been initialized')
    print('reading mnist')
    mnist = input_data.read_data_sets('assets/', one_hot=True)

    train_x, train_y, test_x, test_y = \
        mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
    train_x = train_x.reshape(-1, 28, 28, 1)
    test_x = test_x.reshape(-1, 28, 28, 1)

    _y = np.zeros((55000, 7))
    train_y = np.hstack([train_y, _y])

    print('load mnist done')
    print('load extra data')
    extra_data, extra_label = get_extra_data('data/jpg')
    extra_data = extra_data.reshape(-1, 28, 28, 1)
    print('load extra data done')
    print('training')
    cnn.fit(train_x, train_y, test_x, test_y, extra_tx=extra_data, extra_ty=extra_label)

    cnn.save(model_name)


def num_recognition(img, cnn):
    """

    :param img:
    :param cnn:
    :return:
    """
    result = cnn.predict(img)
    return result


def regions_recognition(regions, model_name):
    """

    :param img:
    :param regions:
    :param model_name:
    :return:
    """
    cnn = CNN()
    cnn.load_session(model_name)
    rec = {'0': '0', '1': '1', '2': '2', '3': '3', '4': '4', '5': '5',
           '6': '6', '7': '7', '8': '8', '9': '9', '10': '+', '11': '-',
           '12': '*', '13': '/', '14': '=', '15': '(', '16': ')'}

    for i, question_region in regions.items():
        question = []
        stem = []
        answer = []
        flag = 0
        for j, number_region in question_region.get_sub_regions().items():
            # recognize numbers
            result = num_recognition(number_region.get_img(), cnn)
            # add the number recognition result to question region
            regions[i].get_sub_regions()[j].set_recognition(rec[str(result[0])])

            question.append(rec[str(result[0])])

            if rec[str(result[0])] == '=':
                flag = 1
                continue

            if flag == 0:
                stem.append(rec[str(result[0])])
            else:
                answer.append(rec[str(result[0])])

            # if rec[str(result[0])] == '=':
            #     flag = 1

        regions[i].set_recognition(''.join(question))

        stem = ''.join(stem)
        answer = ''.join(answer)
        try:
            regions[i].set_result(eval(stem) == answer)
            print('reco', 'stem:', stem, 'answer:', answer, 'result:', eval(stem) == eval(answer))
        except:
            print('error', 'stem:', stem, 'answer:', answer)
            # raise ValueError('the recognition is incorrect')
            pass

    return regions
