import cv2 as cv
import numpy as np

cv.namedWindow('main')

src = cv.imread('cs0.jpg')
center = (src.shape[0]/2, src.shape[1]/2)
rotation_matrix = cv.getRotationMatrix2D(center, 30, 1)
# rotation_matrix_rect = cv.getRotationMatrix2D((312, 381), 30, 1)
# 312 381 128 360
# points = np.array([[248 - center[0], 201 - center[1], 1],
#                    [376 - center[0], 201 - center[1], 1],
#                    [248 - center[0], 561 - center[1], 1],
#                    [376 - center[0], 561 - center[1], 1]])
points = np.array([[248, 201, 1],
                   [376, 201, 1],
                   [248, 561, 1],
                   [376, 561, 1]])

cv.rectangle(src, points[0][0:2], points[3][0:2], (0, 0, 255), 1)
cv.imshow('main', src)
#cv.waitKey()


img = cv.warpAffine(src, rotation_matrix, (src.shape[0], src.shape[1]))

# rotation_matrix = np.concatenate([rotation_matrix, [[0, 0, 1]]])
points = np.dot(rotation_matrix, points.T).astype(np.uint8)
cv.rectangle(img, points.T[0], points.T[3], (0, 0, 255), 1)
cv.imshow('main', img)
cv.waitKey()
