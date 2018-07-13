# coding: utf-8

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
from keras.layers.merge import concatenate, add
from keras.layers.core import Lambda
from keras.models import Model
from keras.models import load_model
import tensorflow as tf
import os
import skimage
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, BatchNormalization, Reshape
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import VGG16
from keras.layers import GRU
from keras.callbacks import *
from keras.layers.merge import *

characters = '0123456789+-*/=()'
width, height, n_len, n_class = 400, 80, 4, len(characters) + 1


# Evaluator
def ctc_lambda_func(args):
    y_pred, labels, input_length, label_length = args
    y_pred = y_pred[:, 2:, :]
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def evaluate(batch_size=128, steps=10):
    batch_acc = 0
    generator = gen(batch_size)
    for i in range(steps):
        [X_test, y_test, _, _], _ = next(generator)
        y_pred = base_model.predict(X_test)
        shape = y_pred[:, 2:, :].shape
        ctc_decode = K.ctc_decode(y_pred[:, 2:, :], input_length=np.ones(shape[0]) * shape[1])[0][0]
        print(K.ctc_decode(y_pred, input_length=np.ones(shape[0]) * shape[1]))
        out = K.get_value(ctc_decode)[:, :n_len]
        if out.shape[1] == n_len:
            batch_acc += (y_test == out).all(axis=1).mean()
    return batch_acc


class Evaluator(Callback):
    def __init__(self):
        self.accs = []

    def on_epoch_end(self, epoch, logs=None):
        acc = evaluate(steps=20) * 100
        self.accs.append(acc)
        print('')
        print('acc: %f%%' % acc)


evaluator = Evaluator()


# Toast Data
def generate():
    ds = '0123456789'
    ts = ['{}{}{}{}{}', '({}{}{}){}{}', '{}{}({}{}{})']
    os = '+-*/'
    # os = ['+', '-', 'times', 'div']
    cs = [random.choice(ds) if x % 2 == 0 else random.choice(os) for x in range(5)]
    return random.choice(ts).format(*cs)


def get_img_by_char(char, base_path='./pre_ocr'):
    """
    get a img by giving char
    :param char:
    :param base_path:
    :return:
    """
    opdict = {'+': 10, '-': 11, '*': 12, '/': 13, '=': 14, '(': 15, ')': 16}
    if char in opdict.keys():
        char = opdict[char]
    path = os.path.join(base_path, str(char))
    files = os.listdir(path)

    rdm = random.randint(0, len(files) - 1)

    if rdm >= len(files):
        print(path, len(files), rdm)

    file = files[rdm]
    path = os.path.join(path, file)
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)


def get_sequence_img(chars):
    x = get_img_by_char(chars[0])
    for i in range(1, len(chars)):
        x = np.hstack([x, get_img_by_char(chars[i])])
    x = cv2.resize(x, (400, 80))
    #     x = skimage.util.random_noise(x, mode='gaussian', clip=True)
    #     print('get_sequence_img output')
    #     plt.imshow(x)
    #     plt.show()
    #     print (chars, x.shape)
    return x


def gen(batch_size=128, gene=4):
    #     X = np.zeros((batch_size, width, height, 1), dtype=np.uint8)
    X = np.zeros((batch_size, width, height, 3), dtype=np.uint8)  # make channel = 3

    y = np.zeros((batch_size, n_len), dtype=np.uint8)
    while True:
        for i in range(batch_size):
            random_str = ''.join([random.choice(characters) for j in range(n_len)])
            #             random_str = '60/3=20'
            tmp = np.array(get_sequence_img(random_str))
            tmp = tmp.reshape(tmp.shape[0], tmp.shape[1], 1)

            #  make channel = 3
            tmp0 = np.copy(tmp)
            tmp = np.concatenate([tmp, tmp0], axis=2)
            tmp = np.concatenate([tmp, tmp0], axis=2)

            tmp = tmp.transpose(1, 0, 2)

            X[i] = tmp
            y[i] = [characters.find(x) for x in random_str]

        yield [X, y, np.ones(batch_size) * rnn_length, np.ones(batch_size) * n_len], np.ones(batch_size)


# Model Struct
input_tensor = Input((width, height, 3))
vgg = VGG16(weights='./vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            include_top=False, input_tensor=input_tensor)
tensor_shape = vgg.output.shape
print('vgg output shape: ', tensor_shape)

rnn_length = tensor_shape[1].value
rnn_dimen = tensor_shape[2].value * tensor_shape[3].value
units = tensor_shape[3].value

print('rnnlength', rnn_length, 'rnndimension', rnn_dimen, 'units', units)

x = Reshape(input_shape=(vgg.output.shape), target_shape=(rnn_length, rnn_dimen))(vgg.output)
print('reshape:', x.shape)

rnn_length -= 2

x = Dense(128, kernel_initializer='he_normal')(x)
x = BatchNormalization()(x)
x = Activation('relu')(x)
print('now x\'s shape:', x.shape)

rnn_size = 128
gru_1 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru1')(x)
gru_1b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru1b')(x)
gru1_merged = add([gru_1, gru_1b])
print('gru1_merged\'s shape:', gru1_merged.shape)

gru_2 = GRU(rnn_size, return_sequences=True, kernel_initializer='he_normal', name='gru2')(gru1_merged)
gru_2b = GRU(rnn_size, return_sequences=True, go_backwards=True, kernel_initializer='he_normal', name='gru2b')(gru1_merged)
x = concatenate([gru_2, gru_2b])
print('now x\'s shape:', x.shape)

x = Dropout(0.25)(x)
x = Dense(n_class, kernel_initializer='he_normal', activation='softmax')(x)

base_model = Model(input=input_tensor, output=x)
print('base_model output shape:', base_model.output.shape)

labels = Input(name='the_labels', shape=[n_len], dtype='float32')
input_length = Input(name='input_length', shape=(1,), dtype='int64')
label_length = Input(name='label_length', shape=(1,), dtype='int64')
loss_out = Lambda(ctc_lambda_func, name='ctc')([base_model.output, labels, input_length, label_length])

model = Model(inputs=(input_tensor, labels, input_length, label_length), outputs=loss_out)  # 输入列表，输出列表
# model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer='adam')
model.compile(loss={'ctc':  lambda y_true, y_pred: y_pred}, optimizer='adam')  # y_pred其实是loss_out

h = model.fit_generator(gen(128), steps_per_epoch=200, epochs=2,
                        callbacks=[evaluator],
                        validation_data=gen(128), validation_steps=2)

base_model.save('./vcrnn_model_4.h5')
print('save model done!')

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(h.history['loss'])
plt.plot(h.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
plt.plot(evaluator.accs)
plt.ylabel('acc')
plt.xlabel('epoch')
plt.savefig('fig.jpg')
plt.close()
print('all done')
