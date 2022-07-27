import json
import os
import random
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from yolo.yolov5 import Yolo


def load_dataset(model: Yolo, root, device):
    dataset_x = None
    dataset_y = None
    files = os.listdir(os.path.join(root, 'data'))
    count = 0
    with open(os.path.join(root, 'config.json')) as f:
        config = json.load(f)
        label = config['label']
    for file in files:
        with open(os.path.join(root, 'data', file)) as f:
            data = json.load(f)
            img_file = data['img']
            x = data['x']
            y = data['y']
            w = data['w']
            h = data['h']
            c = data['class']
            img = Image.open(os.path.join(root, img_file))
            x_temp = torch.tensor(np.transpose(np.array(img)/255, [2, 0, 1]), dtype=torch.float32, device=device)
            x_temp = torch.unsqueeze(x_temp, 0)
            if dataset_x is None:
                dataset_x = x_temp
            else:
                dataset_x = torch.concat([dataset_x, x_temp])
            y_temp = make_labels(model, torch.tensor(np.stack([x, y, w, h, c], 0), dtype=torch.long).to(device), device)
            y_temp = (torch.unsqueeze(y_temp[0], 0),
                      torch.unsqueeze(y_temp[1], 0),
                      torch.unsqueeze(y_temp[2], 0))
            if dataset_y is None:
                dataset_y = y_temp
            else:
                dataset_y = (torch.concat([dataset_y[0], y_temp[0]]),
                             torch.concat([dataset_y[1], y_temp[1]]),
                             torch.concat([dataset_y[2], y_temp[2]]))
            count = count + 1
    return dataset_x, dataset_y, label


def train(model: Yolo, input, target, batch_size, epoch, optimizer: optim.Adam, device):
    index = list(range(input.shape[0]))
    for i in range(epoch):
        random.shuffle(index)
        for j in range(int(np.ceil(input.shape[0] / batch_size))):
            selected = index[batch_size*j:batch_size*(j+1)]
            pred = model.forward(input[selected, ...].to(device))
            loss = model.compute_loss(pred, (target[0][selected, ...].to(device),
                                             target[1][selected, ...].to(device),
                                             target[2][selected, ...].to(device)))
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            print(f'Epoch={i}, Batch={j}: loss:{loss.item()}')
    print('finished')


def detect(model: Yolo, inputs):
    batch_size = inputs.shape[0]
    pred = model.forward(inputs)
    detect_result = list()
    for bs in range(batch_size):
        borders = list()
        for i in range(3):
            for row in range(pred[i].shape[2]):
                for col in range(pred[i].shape[3]):
                    possibility = pred[i][bs, 0, row, col]
                    if possibility.item() > 0.5:
                        x = pred[i][bs, 1, row, col]
                        y = pred[i][bs, 2, row, col]
                        w = pred[i][bs, 3, row, col]
                        h = pred[i][bs, 4, row, col]
                        classes = pred[i][bs, 5:, row, col]
                        c = torch.argmax(classes).item()
                        point_leftTop = (model.grid_length[i] * col, model.grid_length[i] * row)
                        point_mid = (point_leftTop[0] + x*model.grid_length[i], point_leftTop[1] + y*model.grid_length[i])
                        point_1 = (int(point_mid[0] - w*model.grid_length[i]/2),
                                   int(point_mid[1] - h*model.grid_length[i]/2))
                        point_2 = (int(point_mid[0] + w*model.grid_length[i]/2),
                                   int(point_mid[1] + h*model.grid_length[i]/2))
                        sorted_point_leftTop_rightBottom = [point_1, point_2]
                        sorted_point_leftTop_rightBottom.sort()
                        borders.append([possibility.item(), sorted_point_leftTop_rightBottom[0], sorted_point_leftTop_rightBottom[1], c])
        # Non Maximum Suspision
        borders.sort(reverse=True)
        filterd_borders = list()
        while len(borders) > 0:
            i = borders[0]
            borders.remove(i)
            filterd_borders.append(i)
            index = 0
            while index < len(borders):
                i_x_min = i[1][0]
                i_x_max = i[2][0]
                i_y_min = i[1][1]
                i_y_max = i[2][1]
                j_x_min = borders[index][1][0]
                j_x_max = borders[index][2][0]
                j_y_min = borders[index][1][1]
                j_y_max = borders[index][2][1]
                si = (i_x_max - i_x_min) * (i_y_max - i_y_min)
                sj = (j_x_max - j_x_min) * (j_y_max - j_y_min)
                intersection_w = np.max([0, np.min([i_x_max, j_x_max]) - np.max([i_x_min, j_x_min])])
                intersection_h = np.max([0, np.min([i_y_max, j_y_max]) - np.max([i_y_min, j_y_min])])
                s_intersection = intersection_w * intersection_h
                s_added = si + sj - s_intersection
                iou = s_intersection / s_added
                if iou >= 0.3:
                    borders.remove(borders[index])
                else:
                    index = index + 1
        # Filtered_borders
        detect_result.append(filterd_borders)
    return detect_result


def open_img(path, device):
    img = Image.open(path)
    origin = np.array(img)
    x = torch.unsqueeze(torch.tensor(np.transpose(np.array(img)/255, [2, 0, 1]), dtype=torch.float32, device=device), 0)
    return origin, x


def make_labels(model: Yolo, target, device):
    # target's shape =([x,y,w,h,class],n)
    # Initialize Empty placholder
    label_small = torch.zeros((model.output_vector, model.small_grid_size, model.small_grid_size), device=device)
    mask_small = torch.divide(target[0:2, :], torch.tensor(model.small_grid_length), rounding_mode='trunc')
    mask_rows = mask_small[1:2, :]
    mask_cols = mask_small[0:1, :]
    # Fill possibility
    label_small[0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[0:2, :] - model.small_grid_length * mask_small[0:2, :]) / model.small_grid_length
    shape = label_small[1:3, mask_rows, mask_cols].shape
    label_small[1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[2:4, :] / model.small_grid_length
    shape = label_small[3:5, mask_rows, mask_cols].shape
    label_small[3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[4, :], num_classes=model.cn)
    shape = label_small[5:, mask_rows, mask_cols].shape
    label_small[5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 0, 1), shape).float()

    label_medium = torch.zeros((model.output_vector, model.medium_grid_size, model.medium_grid_size), device=device)
    mask_medium = torch.divide(target[0:2, :], torch.tensor(model.medium_grid_length), rounding_mode='trunc')  # Shape=(Shape,[col,row],n)
    mask_rows = mask_medium[1:2, :]
    mask_cols = mask_medium[0:1, :]
    # Fill possibility
    label_medium[0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[0:2, :] - model.medium_grid_length * mask_medium[0:2, :]) / model.medium_grid_length
    shape = label_medium[1:3, mask_rows, mask_cols].shape
    label_medium[1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[2:4, :] / model.medium_grid_length
    shape = label_medium[3:5, mask_rows, mask_cols].shape
    label_medium[3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[4, :], num_classes=model.cn)
    shape = label_medium[5:, mask_rows, mask_cols].shape
    label_medium[5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 0, 1), shape).float()

    label_big = torch.zeros((model.output_vector, model.big_grid_size, model.big_grid_size), device=device)
    mask_big = torch.divide(target[0:2, :], torch.tensor(model.big_grid_length), rounding_mode='trunc')  # Shape=(Shape,[col,row],n)
    mask_rows = mask_big[1:2, :]
    mask_cols = mask_big[0:1, :]
    # Fill possibility
    label_big[0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[0:2, :] - model.big_grid_length * mask_big[0:2, :]) / model.big_grid_length
    shape = label_big[1:3, mask_rows, mask_cols].shape
    label_big[1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[2:4, :] / model.big_grid_length
    shape = label_big[3:5, mask_rows, mask_cols].shape
    label_big[3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[4, :], num_classes=model.cn)
    shape = label_big[5:, mask_rows, mask_cols].shape
    label_big[5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 0, 1), shape).float()
    return label_small, label_medium, label_big


"""
def make_labels(model: Yolo, target, device):
    batch_size = target.shape[0]
    # target's shape =(None,[x,y,w,h,class],n)
    # Initialize Empty placholder
    label_small = torch.zeros((batch_size, model.output_vector, model.small_grid_size, model.small_grid_size))
    mask_small = torch.divide(target[:, 0:2, :], torch.tensor(model.small_grid_length), rounding_mode='trunc')  # Shape=(Shape,[col,row],n)
    mask_rows = mask_small[:, 1:2, :]
    mask_cols = mask_small[:, 0:1, :]
    # Fill possibility
    label_small[:, 0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[:, 0:2, :] - model.small_grid_length * mask_small[:, 0:2, :]) / model.small_grid_length
    shape = label_small[:, 1:3, mask_rows, mask_cols].shape
    label_small[:, 1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[:, 2:4, :] / model.small_grid_length
    shape = label_small[:, 3:5, mask_rows, mask_cols].shape
    label_small[:, 3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[:, 4, :], num_classes=model.cn)
    shape = label_small[:, 5:, mask_rows, mask_cols].shape
    label_small[:, 5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 1, 2), shape).float()

    label_medium = torch.zeros((batch_size, model.output_vector, model.medium_grid_size, model.medium_grid_size))
    mask_medium = torch.divide(target[:, 0:2, :], torch.tensor(model.medium_grid_length), rounding_mode='trunc')  # Shape=(Shape,[col,row],n)
    mask_rows = mask_medium[:, 1:2, :]
    mask_cols = mask_medium[:, 0:1, :]
    # Fill possibility
    label_medium[:, 0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[:, 0:2, :] - model.medium_grid_length * mask_medium[:, 0:2, :]) / model.medium_grid_length
    shape = label_medium[:, 1:3, mask_rows, mask_cols].shape
    label_medium[:, 1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[:, 2:4, :] / model.medium_grid_length
    shape = label_medium[:, 3:5, mask_rows, mask_cols].shape
    label_medium[:, 3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[:, 4, :], num_classes=model.cn)
    shape = label_medium[:, 5:, mask_rows, mask_cols].shape
    label_medium[:, 5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 1, 2), shape).float()

    label_big = torch.zeros((batch_size, model.output_vector, model.big_grid_size, model.big_grid_size))
    mask_big = torch.divide(target[:, 0:2, :], torch.tensor(model.big_grid_length), rounding_mode='trunc')  # Shape=(Shape,[col,row],n)
    mask_rows = mask_big[:, 1:2, :]
    mask_cols = mask_big[:, 0:1, :]
    # Fill possibility
    label_big[:, 0, mask_rows, mask_cols] = 1
    # Fill x, y
    xy = (target[:, 0:2, :] - model.big_grid_length * mask_big[:, 0:2, :]) / model.big_grid_length
    shape = label_big[:, 1:3, mask_rows, mask_cols].shape
    label_big[:, 1:3, mask_rows, mask_cols] = torch.reshape(xy, shape)
    # Fill w,h
    wh = target[:, 2:4, :] / model.big_grid_length
    shape = label_big[:, 3:5, mask_rows, mask_cols].shape
    label_big[:, 3:5, mask_rows, mask_cols] = torch.reshape(wh, shape)
    # Fill class
    class_one_hot = F.one_hot(target[:, 4, :], num_classes=model.cn)
    shape = label_big[:, 5:, mask_rows, mask_cols].shape
    label_big[:, 5:, mask_rows, mask_cols] = torch.reshape(torch.swapaxes(class_one_hot, 1, 2), shape).float()
    return label_small.to(device), label_medium.to(device), label_big.to(device)
"""


def save_model(model: Yolo, optimizer: optim.Adam, path):
    torch.save({
        'model': model.state_dict(),
        'optim': optimizer.state_dict()
    }, path)


def load_model_train(model: Yolo, optimizer: optim.Adam, path):
    states = torch.load(path)
    model.load_state_dict(states['model'])
    optimizer.load_state_dict(states['optim'])


def load_model_detect(model: Yolo, path):
    states = torch.load(path)
    model.load_state_dict(states['model'])
