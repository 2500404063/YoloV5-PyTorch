from copy import copy
import json
import os
import cv2 as cv
import numpy as np
import _4_scale as img_scale
import _5_color as img_corlor

root = './processed_img'

files = os.listdir(os.path.join(root, 'data'))
for file in files:
    with open(os.path.join(root, 'data', file)) as f:
        data = json.load(f)
        img_file = data['img']
        print(img_file)
        x = np.array(data['x'])
        y = np.array(data['y'])
        w = np.array(data['w'])
        h = np.array(data['h'])
        src = cv.imread(os.path.join(root, img_file))
        xmin = x - w/2
        xmax = x + w/2
        ymin = y - h/2
        ymax = y + h/2

        img1, _x, _y, _w, _h = img_scale.vertical_flip(src, xmin, xmax, ymin, ymax)
        data_img1 = copy(data)
        data_img1['img'] = f'{img_file[:-4]}_1{img_file[-4:]}'
        data_img1['x'] = _x.tolist()
        data_img1['y'] = _y.tolist()
        data_img1['w'] = _w.tolist()
        data_img1['h'] = _h.tolist()
        cv.imwrite(os.path.join(root, f'{img_file[:-4]}_1{img_file[-4:]}'), img1)
        with open(os.path.join(root, 'data', f'{img_file[:-4]}_1{img_file[-4:]}.json'), 'w+') as fd:
            json.dump(data_img1, fd)

        id_1 = 0
        for i in [src, img1]:
            id_2 = 0
            # color_contrast_1
            img4 = img_corlor.contrast_light(i, np.random.randint(2, 10)/10, 0)
            # color_contrast_2
            img5 = img_corlor.contrast_light(i, np.random.randint(15, 30)/10, 0)
            # color_light_1
            img6 = img_corlor.contrast_light(i, 1, np.random.randint(30, 80))
            # color_contrast_light
            img7 = img_corlor.contrast_light(i, np.random.randint(15, 30)/10, np.random.randint(30, 60))

            for j in [img4, img5, img6, img7]:
                cv.imwrite(os.path.join(root, f'{img_file[:-4]}_{id_1}_{id_2}{img_file[-4:]}'), j)
                data['img'] = f'{img_file[:-4]}_{id_1}_{id_2}{img_file[-4:]}'
                with open(os.path.join(root, 'data', f'{img_file[:-4]}_{id_1}_{id_2}{img_file[-4:]}.json'), 'w+') as fd:
                    json.dump(data, fd)
                id_2 = id_2 + 1
            id_1 = id_1 + 1
