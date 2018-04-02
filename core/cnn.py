# coding: utf-8

import tensorflow as tf
import numpy as np


class CNN(object):
    """

    """

    def __init__(self, p_keep_conv, p_keep_hidden, batch_size, test_size, epoch_time):
        """
        Initialization
        :param batch_size
        :param test_size
        """
        self.w = self._init_weights([3, 3, 1, 32])  # 第一层卷积核大小为3x3,输入一张图,输出32个feature map
        self.w2 = self._init_weights([3, 3, 32, 64])  # 第二层卷积核大小为3x3,输入32个feature map,输出64个feature map
        self.w3 = self._init_weights([3, 3, 64, 128])  # 第三层卷积核大小为3x3,输入64个feature map,输出128个feature map
        self.w4 = self._init_weights([128 * 4 * 4, 625])  # FC 128 * 4 * 4 inputs, 625 outputs
        self.w_o = self._init_weights([625, 10])  # FC 625 inputs, 10 outputs (labels)
        self.batch_size = batch_size
        self.test_size = test_size
        self.p_keep_conv = tf.placeholder("float")  # 卷积层的dropout概率
        self.p_keep_hidden = tf.placeholder("float")  # 全连接层的dropout概率
        self.X = tf.placeholder("float", [None, 28, 28, 1])
        self.Y = tf.placeholder("float", [None, 10])
        self.sess = None  # tensorflow session
        self.p_keep_conv = p_keep_conv
        self.p_keep_hidden = p_keep_hidden
        self.epoch_time = epoch_time

    def _init_weights(self, shape):
        """
        初始化参数
        :param shape:
        :return:
        """
        return tf.Variable(tf.random_normal(self, shape, stddev=0.01))

    def _cnn_main(self):
        """

        :return:
        """
        # 第一个卷积层:padding=SAME,保证输出的feature map与输入矩阵的大小相同
        l1a = tf.nn.relu(tf.nn.conv2d(self.X, self.w,  # l1a shape=(?, 28, 28, 32)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        # max_pooling,窗口大小为2x2
        l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                            strides=[1, 2, 2, 1], padding='SAME')
        # dropout:每个神经元有p_keep_conv的概率以1/p_keep_conv的比例进行归一化,有(1-p_keep_conv)的概率置为0
        l1 = tf.nn.dropout(l1, self.p_keep_conv)

        # 第二个卷积层
        l2a = tf.nn.relu(tf.nn.conv2d(l1, self.w2,  # l2a shape=(?, 14, 14, 64)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                            strides=[1, 2, 2, 1], padding='SAME')
        l2 = tf.nn.dropout(l2, self.p_keep_conv)

        # 第三个卷积层
        l3a = tf.nn.relu(tf.nn.conv2d(l2, self.w3,  # l3a shape=(?, 7, 7, 128)
                                      strides=[1, 1, 1, 1], padding='SAME'))
        l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                            strides=[1, 2, 2, 1], padding='SAME')
        # 将所有的feature map合并成一个2048维向量
        l3 = tf.reshape(l3, [-1, self.w4.get_shape().as_list()[0]])  # reshape to (?, 2048)
        l3 = tf.nn.dropout(l3, self.p_keep_conv)
        # 后面两层为全连接层
        l4 = tf.nn.relu(tf.matmul(l3, self.w4))
        l4 = tf.nn.dropout(l4, self.p_keep_hidden)
        out = tf.matmul(l4, self.w_o)

        return out

    def fit(self, train_x, train_y, test_x, test_y):
        """

        :param train_x:
        :param train_y:
        :param test_x:
        :param test_y:
        :param p_keep_conv:
        :param p_keep_hidden:
        :return:
        """
        out = self._cnn_main()
        cost = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                logits=out, labels=self.Y))  # 交叉熵目标函数

        train_op = tf.train.RMSPropOptimizer(0.001, 0.9) \
            .minimize(cost)  # RMSPro算法最小化目标函数

        predict_op = tf.argmax(out, 1)  # 返回每个样本的预测结果

        self.sess = tf.Session()
        for i in range(self.epoch_time):
            training_batch = zip(range(0, len(train_x), self.batch_size),
                                 range(self.batch_size, len(train_x) + 1, self.batch_size))
            for start, end in training_batch:
                self.sess.run(train_op, feed_dict={self.X: train_x[start: end],
                                                   self.Y: train_y[start: end]})
            # test
            test_indices = np.arange(len(train_x))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0: test_indices]

            print(i, np.mean(np.argmax(test_y[test_indices], axis=1) ==
                             self.sess.run(
                                 predict_op, feed_dict={
                                     self.X: test_x[test_indices],
                                     self.p_keep_conv: 1.0, self.p_keep_hidden: 1.0})))
        print('Train done')

    def save(self, model_name):
        """

        :param model_name:
        :return:
        """
        saver = tf.train.Saver()
        saver.save(self.sess, model_name)
        print("save model:{0} Finished".format(model_name))
