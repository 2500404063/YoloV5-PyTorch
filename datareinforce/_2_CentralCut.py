import os
import cv2 as cv
import pyautogui as pg

screen_size = pg.size()
shot_edge = 640
cut_left = int(screen_size.width/2 - (shot_edge/2))
cut_top = int(screen_size.height/2 - (shot_edge/2))
cut_right = int(screen_size.width/2 + (shot_edge/2))
cut_bottom = int(screen_size.height/2 + (shot_edge/2))

dir = 'raw_cs'
files = os.listdir(dir)
for f in files:
    img = cv.imread(os.path.join(dir, f))
    img2 = img[cut_top:cut_bottom, cut_left:cut_right, :]
    cv.imwrite("processed.jpg", img2)
