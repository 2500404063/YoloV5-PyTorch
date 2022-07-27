import os
import cv2 as cv
import numpy as np
import _4_scale as img_scale
import _5_color as img_corlor

img_dir = './processed_cs'
for file_name in os.listdir(img_dir):
    print(f'start: {file_name}')
    img = cv.imread(os.path.join(img_dir, file_name))
    # vertical_flip
    img1 = img_scale.vertical_flip(img)
    cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_1{file_name[-4:]}'), img1)
    # rotation_15
    img2 = img_scale.rotate(img, np.random.randint(10, 30), None, 1)
    cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_2{file_name[-4:]}'), img2)
    # rotation_-15
    img3 = img_scale.rotate(img, np.random.randint(-30, -10), None, 1)
    cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_3{file_name[-4:]}'), img3)
    count = 0
    for i in [img, img1, img2, img3]:
        # color_contrast_1
        img4 = img_corlor.contrast_light(i, np.random.randint(2, 9)/10, 0)
        cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_4_{count}{file_name[-4:]}'), img4)
        # color_contrast_2
        img5 = img_corlor.contrast_light(i, np.random.randint(11, 20)/10, 0)
        cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_5_{count}{file_name[-4:]}'), img5)
        # color_light_1
        img6 = img_corlor.contrast_light(i, 1, np.random.randint(20, 40))
        cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_6_{count}{file_name[-4:]}'), img6)
        # color_light_2
        img7 = img_corlor.contrast_light(i, 1, np.random.randint(41, 70))
        cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_7_{count}{file_name[-4:]}'), img7)
        # color_contrast_light
        img6 = img_corlor.contrast_light(i, np.random.randint(5, 10)/10, np.random.randint(20, 40))
        cv.imwrite(os.path.join(img_dir, f'{file_name[:-4]}_6_{count}{file_name[-4:]}'), img6)
        count = count + 1
