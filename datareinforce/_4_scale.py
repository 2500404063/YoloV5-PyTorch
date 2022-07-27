import cv2 as cv
import numpy as np


def vertical_flip(src, xmin, xmax, ymin, ymax):
    points = np.array([[xmin, ymin],
                       [xmax, ymin],
                       [xmax, ymax],
                       [xmin, ymax]])
    points[:, 0:1, :] = src.shape[0] - points[:, 0:1, :]
    cx = ((points[1, 0, :] + points[3, 0, :]) / 2).astype(np.int32)
    cy = ((points[1, 1, :] + points[3, 1, :]) / 2).astype(np.int32)
    w = ((points[3, 0, :] - points[1, 0, :])).astype(np.int32)
    h = ((points[3, 1, :] - points[1, 1, :])).astype(np.int32)
    return cv.flip(src, 1), cx, cy, w, h


def rotate(src, angle, center=None, scale=1):
    if center is None:
        center = (src.shape[0]/2, src.shape[1]/2)
    rotation_matrix = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(src, rotation_matrix, (src.shape[0], src.shape[1]))
