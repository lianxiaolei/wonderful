# coding: utf-8

from process.processor import *


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    img = read_img(file_name, color_inv_norm=True)
    regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10, display=False)
    alg_train('model/Test_CNN_Model.ckpt', epoch_time=3, p_keep_conv=0.8, p_keep_hidden=0.6)
    # regions_recognition(regions, 'model/Test_CNN_Model.ckpt')
    # show_all_regions(img, regions, layer=0)
    # save_all_regions(regions, dir_name=['data/ques', 'data/nums'])


if __name__ == '__main__':

    # run('images/10.jpg')
    run('images/11.png')
    # run('images/12.png')
    # run('images/13.jpg')
    # run('images/14.jpg')
    # run('images/zx.jpg')
    # run('images/zx2.png')
    # run('images/zx3.png')
    # run('images/zm.jpg')
    # run('images/zm1.jpg')
    # run('images/mine.jpg')
    # run('images/mine1.jpg')
    # run('images/mine2.jpg')
    # run('images/cgd.jpg')
