import json
import os
import cv2 as cv
import numpy as np
import _4_scale as tool

root = './processed_img'

cv.namedWindow('main')
files = os.listdir(os.path.join(root, 'data'))
for file in files:
    with open(os.path.join(root, 'data', file)) as f:
        data = json.load(f)
        img_file = data['img']
        x = np.array(data['x'])
        y = np.array(data['y'])
        w = np.array(data['w'])
        h = np.array(data['h'])
        src = cv.imread(os.path.join(root, img_file))
        xmin = x - w/2
        xmax = x + w/2
        ymin = y - h/2
        ymax = y + h/2
        # 验证vertical_flip函数
        img, cx, cy, _w, _h = tool.vertical_flip(src, xmin, xmax, ymin, ymax)
        for i in range(cx.shape[0]):
            p1 = (int(cx[i] - _w[i] / 2), int(cy[i] - _h[i] / 2))
            p2 = (int(cx[i] + _w[i] / 2), int(cy[i] + _h[i]/2))
            cv.rectangle(img, p1, p2, (0, 255, 0), 1)
        cv.imshow('main', img)
        cv.waitKey()

        # 验证当前结果
        # points = np.array([[xmin, ymin],
        #                    [xmax, ymin],
        #                    [xmax, ymax],
        #                    [xmin, ymax]], dtype=np.int32)
        # src2 = src.copy()
        # for i in range(points.shape[2]):
        #     cv.rectangle(src2, points[0, :, i], points[2, :, i], (0, 255, 0), 1)
        # cv.imshow('main', src2)
        # cv.waitKey()

        # points[:, 0:1, :] = src.shape[0] - points[:, 0:1, :]
        # src = cv.flip(src, 1)
        # for i in range(points.shape[2]):
        #     cv.rectangle(src, points[1, :, i], points[3, :, i], (0, 0, 255), 1)
        # cv.imshow('main', src)
        # cv.waitKey()
