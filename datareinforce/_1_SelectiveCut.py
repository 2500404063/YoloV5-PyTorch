import os
import cv2 as cv

shot_edge = 640
central_point = (0, 0)
cut_left = 0
cut_top = 0
cut_right = 0
cut_bottom = 0

dir = 'raw_img'
outdir = 'processed_img'
prefix = 'csgo'
output_count = 0


def onMouse(event, x, y, flag, parameters):
    global central_point
    global cut_left
    global cut_top
    global cut_right
    global cut_bottom
    global output_count
    if event == cv.EVENT_LBUTTONUP:
        central_point = (x, y)
        cut_left = int(central_point[0] - (shot_edge/2))
        cut_top = int(central_point[1] - (shot_edge/2))
        cut_right = int(central_point[0] + (shot_edge/2))
        cut_bottom = int(central_point[1] + (shot_edge/2))
        img2 = img.copy()
        cv.rectangle(img2, (cut_left, cut_top), (cut_right,
                     cut_bottom), color=(0, 0, 255), thickness=1,)
        cv.imshow('main', img2)
    elif event == cv.EVENT_RBUTTONUP:
        img2 = img[cut_top:cut_bottom, cut_left:cut_right, :]
        cv.imwrite(os.path.join(outdir, f"{prefix}{output_count}.jpg"), img2)
        output_count = output_count + 1


files = os.listdir(dir)
cv.namedWindow('main')
cv.setMouseCallback('main', onMouse)
for f in files:
    img = cv.imread(os.path.join(dir, f))
    print(f)
    cv.imshow('main', img)
    cv.waitKey()
