# coding: utf-8

from process.processor import *


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    img = read_img(file_name, color_inv_norm=True)
    regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10)
    # alg_train('model/Test_CNN_Model.ckpt', epoch_time=3, p_keep_conv=0.8, p_keep_hidden=0.6)
    # regions_recognition(regions, 'model/Test_CNN_Model.ckpt')
    show_all_regions(img, regions, layer=0)
    # save_all_regions(regions, dir_name=['./question', './number'])


if __name__ == '__main__':

    run('images/11.png')
    # run('images/24.jpg')
    # run('images/14.jpg')
    # run('images/13.png')
    # run('images/15.jpg')
    # run('images/16.jpg')
