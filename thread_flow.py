# coding: utf-8

from process.processor import *
import warnings
warnings.filterwarnings("ignore")


def run(file_name):
    """
    处理主流程
    :param file_name:
    :return:
    """
    img = read_img(file_name, color_inv_norm=True)
    regions = cut(img, row_eps=img.shape[1] / 30, col_eps=10, display=False)
    # alg_train('num_model/Nums_CNN_Model.ckpt', epoch_time=3, p_keep_conv=0.8, p_keep_hidden=0.6)
    # regions_recognition(regions, 'new_model/Test_CNN_Model.ckpt')
    save_all_regions(regions, dir_name=['data/ques', 'data/nums'])
    print('save done')
    # show_all_regions(img, regions, layer=0)

    # alg_train_new('new_model/Test_CNN_Model.ckpt', epoch_time=3, p_keep_conv=0.8, p_keep_hidden=0.6)


def get_dilate_img(base_path):
    nums = os.listdir(base_path)
    for num in nums:
        jpgs = os.listdir(os.path.join(base_path, num))
        for jpg in jpgs:
            fname = os.path.join(base_path, num, jpg)
            pic = read_img(fname, color_inv_norm=False)
            kernel = np.ones((2, 2), np.uint8)
            img = cv2.dilate(pic, kernel, iterations=1)
            cv2.imwrite(fname, img * 255.0)
            cv2.destroyAllWindows()


if __name__ == '__main__':

    # run('images/10.jpg')
    # run('images/11.png')
    # run('images/12.png')
    # run('images/111.jpg')
    # run('images/112.jpg')
    # run('images/113.jpg')
    # run('images/114.jpg')
    # run('images/115.jpg')
    # run('images/zx.jpg')
    # run('images/zx1.jpg')
    # run('images/zm.jpg')
    # run('images/zm1.jpg')
    # run('images/zm2.jpg')
    # run('images/mine.jpg')
    # run('images/mine1.jpg')
    # run('images/mine2.jpg')
    # run('images/mine3.jpg')
    # run('images/mine4.jpg')
    # run('images/mine6.jpg')
    # run('images/mine7.jpg')
    # run('images/mine8.jpg')
    # run('images/mine9.jpg')
    # run('images/mine10.jpg')
    # run('images/xb.jpg')
    # run('images/cz.jpg')
    # run('images/000.jpg')
    # run('images/001.jpg')
    run('images/222.jpeg')
    # run('images/cz1.jpg')
    # get_dilate_img('data/jpg')
