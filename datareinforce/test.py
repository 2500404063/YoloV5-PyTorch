import cv2 as cv
import numpy as np
import _5_color as test

cv.namedWindow('main')
img = cv.imread('./processed_img/csgo0.jpg')
# img = test.contrast_light(img, np.random.randint(2, 10)/10, 0)
# img = test.contrast_light(img, np.random.randint(15, 30)/10, 0)
img = test.random_noise(img, 500, 1)
cv.imshow('main', img)
cv.waitKey()
