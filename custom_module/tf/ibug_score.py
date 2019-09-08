from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np


def ibug_score(labels, predictions):
    right_bottom = tf.reduce_max(labels, axis=-2, keepdims=True)
    left_top = tf.reduce_min(labels, axis=-2, keepdims=True)

    diag_dis = tf.norm(right_bottom - left_top, axis=-1, keepdims=True)
    norm = tf.norm(labels - predictions, axis=-1, keepdims=True) / diag_dis
    norm_batch = tf.reduce_mean(norm, axis=(-1, -2))
    return norm_batch
