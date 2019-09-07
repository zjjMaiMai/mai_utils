from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
import cv2


def from_rad(rad):
    ret = np.eye(3)
    ret[:2, :2] = [[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]]
    return ret


def from_angle(angle):
    return from_rad(np.radians(angle))


def from_uniform_scale(s):
    return np.diag([s, s, 1])


def from_translate(t):
    ret = np.eye(3)
    ret[:2, 2] = t
    return ret


def from_center_rotate(center, angle, scale):
    ret = np.eye(3)
    ret[:2, :] = cv2.getRotationMatrix2D(center, angle, scale)
    return ret
