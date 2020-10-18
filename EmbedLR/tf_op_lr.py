# -*- coding:utf-8 -*-
# @version: 1.0
# @author: wuxikun
# @date: '2020/10/17 3:27 PM'


import numpy as np
import tensorflow as tf


a = tf.constant([1, 2, 3, 0, 9])

b = tf.constant([[1, 2, 3], [3, 2, 1], [4, 5, 6], [6, 5, 4]])

correction = tf.equal(tf.arg_max(b, 1), tf.arg_max(a, 1))

init = tf.global_variables_initializer()
with tf.Session() as sess:

    sess.run(init)
    print(sess.run(tf.arg_max(a, 0)))
    print(b.shape)
    print(sess.run(tf.arg_max(b, 0)))
    print(sess.run(tf.arg_max(b, 1)))



