# coding: utf-8

from core.segmentation import *
from core.cnn import CNN
from tensorflow.examples.tutorials.mnist import input_data


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
        # number_areas = project_cut(
        #     region_arr, 0, 0)

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
    print('load mnist done')
    print('training')
    cnn.fit(train_x, train_y, test_x, test_y)

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

    for i, question_region in regions.items():
        question = []
        stem = []
        answer = []
        flag = 0
        for j, number_region in question_region.get_sub_regions().items():
            # recognize numbers
            result = num_recognition(number_region.get_img(), cnn)
            # add the number recognition result to question region
            regions[i].get_sub_regions()[j].set_recognition(str(result[0]))

            question.append(str(result[0]))

            if flag == 0:
                stem.append(str(result[0]))
            else:
                answer.append(str(result[0]))

            if result == '=':
                flag = 1

        regions[i].set_recognition(''.join(question))

        stem = ''.join(stem)
        answer = ''.join(answer)
        try:
            regions[i].set_result(eval(stem) == answer)
            print('reco', stem, answer)
        except:
            print('error', stem, answer)
            # raise ValueError('the recognition is incorrect')
            pass

    return regions
