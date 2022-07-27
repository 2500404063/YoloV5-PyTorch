from calendar import c
import cv2 as cv
import numpy as np


def contrast_light(src, alpha, beta):
    des = src.copy()
    for y in range(des.shape[0]):
        for x in range(des.shape[1]):
            for c in range(des.shape[2]):
                color = int((alpha * des[y, x, c] + beta))
                if color > 255:
                    des[y, x, c] = 255
                else:
                    des[y, x, c] = color
    return des


def channel_transform(src, alpha, beta, channel):
    des = src.copy()
    for y in range(des.shape[0]):
        for x in range(des.shape[1]):
            for c in range(des.shape[2]):
                delta = int((alpha * des[y, x, c] + beta))
                if c == channel:
                    if des[y, x, c] + delta > 255:
                        des[y, x, c] = 255
                    else:
                        des[y, x, c] = des[y, x, c] + delta
                else:
                    if des[y, x, c] - delta < 0:
                        des[y, x, c] = 0
                    else:
                        des[y, x, c] = des[y, x, c] - delta
    return des


def random_noise(src, times, extent=0):
    des = src.copy()
    for _ in range(times):
        y = np.random.randint(0, des.shape[0])
        x = np.random.randint(0, des.shape[1])
        des[y-extent:y+extent, x-extent:x+extent, :] = 0
    return des
