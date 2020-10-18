# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/4/16'

import tensorflow as tf

from tensorflow.keras.layers import Dense, Input
import numpy as np
from tensorflow.keras.models import Model
# this is a logistic regression in Keras
# x = Input(shape=(32,))
# y = Dense(16, activation='softmax')(x)
# model = Model(x, y)
#
x = Input(shape=(32,))
y = tf.square(x)

x = Input(shape=(32,))
y = Dense(16, activation='softmax')(x)

model = Model(x, y)

print(model.summary())

sample = np.zeros(shape=[1, 32])
target = np.array([100, 16])

print(sample)
# print(model.predict(sample))
print(model(sample))


