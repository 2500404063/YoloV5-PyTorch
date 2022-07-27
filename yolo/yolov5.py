from yolo.units import cbl, csp_cbl, csp_res, sppf
import torch
import torch.nn as nn
import torch.optim as optim

import cv2 as cv
import numpy as np


class Yolo(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        # (possibility,x,y,w,h,class)
        self.cn = 2
        self.output_vector = 5 + self.cn
        self.small_grid_size = 80
        self.small_grid_length = 8
        self.medium_grid_size = 40
        self.medium_grid_length = 16
        self.big_grid_size = 20
        self.big_grid_length = 32
        self.grid_length = [8, 16, 32]
        self.grid_size = [80, 40, 20]
        # Input Shape(640, 640)
        # Backbone
        self.cbl1 = cbl.CBL_Unit(3, 64, 7, 2, padding=3)  # 1/2
        self.cbl2 = cbl.CBL_Unit(64, 128, 3, 2, padding=1)  # 1/4
        self.csp_res1 = csp_res.CSP_RES_Unit(128, 128, 3, res_amount=1)  # 1/4
        self.cbl3 = cbl.CBL_Unit(128, 256, 3, 2, padding=1)  # 1/8
        self.csp_res2 = csp_res.CSP_RES_Unit(256, 256, 3, res_amount=2)  # 1/8
        self.cbl4 = cbl.CBL_Unit(256, 512, 3, 2, padding=1)  # 1/16
        self.csp_res3 = csp_res.CSP_RES_Unit(512, 512, 3, res_amount=3)  # 1/16
        self.cbl5 = cbl.CBL_Unit(512, 1024, 3, 2, padding=1)  # 1/32
        self.csp_res4 = csp_res.CSP_RES_Unit(1024, 1024, 3, res_amount=1)  # 1/32
        self.sppf = sppf.SPFF_Unit(1024, 1024, pool_kernal_size=5)  # 1/32
        # Neck
        self.cbl6 = cbl.CBL_Unit(1024, 512, 3, 1, padding='same')  # 1/32
        self.upsample1 = nn.UpsamplingNearest2d(scale_factor=2)  # 1/32
        self.csp_cbl1 = csp_cbl.CSP_CBL_Unit(1024, 1024, 3,
                                             cbl_amount=3)  # 1/16
        self.cbl7 = cbl.CBL_Unit(1024, 256, 3, 1, padding='same')
        self.upsample2 = nn.UpsamplingNearest2d(scale_factor=2)  # 1/16
        # Neck_output_1
        self.csp_cbl2 = csp_cbl.CSP_CBL_Unit(512, 256, 3, cbl_amount=3)
        self.conv1 = nn.Conv2d(256, self.output_vector, 1, 1, padding='same')
        # Neck_output_2
        self.cbl8 = cbl.CBL_Unit(256, 256, 3, 2, padding=1)
        self.csp_cbl3 = csp_cbl.CSP_CBL_Unit(512, 512, 3, cbl_amount=3)
        self.conv2 = nn.Conv2d(512, self.output_vector, 1, 1, padding='same')
        # Neck_output_3
        self.cbl9 = cbl.CBL_Unit(512, 512, 3, 2, padding=1)
        self.csp_cbl4 = csp_cbl.CSP_CBL_Unit(1024, 1024, 3, cbl_amount=3)
        self.conv3 = nn.Conv2d(1024, self.output_vector, 1, 1, padding='same')

    def forward(self, inputs):
        # Backbone
        x = self.cbl1(inputs)  # shape=(None,64,320,320)
        x = self.cbl2(x)  # shape=(None,128,160,160)
        x = self.csp_res1(x)  # shape=(None,128,160,160)
        x = self.cbl3(x)  # shape=(None,256,80,80)
        x1 = self.csp_res2(x)  # shape=(None,256,80,80)
        x = self.cbl4(x1)  # shape=(None,512,40,40)
        x2 = self.csp_res3(x)  # shape=(None,512,40,40)
        x = self.cbl5(x2)  # shape=(None,1024,20,20)
        x = self.csp_res4(x)  # shape=(None,1024,20,20)
        x = self.sppf(x)  # shape=(None,1024,20,20)
        # Neck
        x3 = self.cbl6(x)  # shape=(None,512,20,20)
        x4 = self.upsample1(x3)  # shape=(None,512,40,40)
        x4 = torch.concat([x2, x4], 1)  # shape=(None,1024,40,40)
        x4 = self.csp_cbl1(x4)  # shape=(None,1024,40,40)
        x4 = self.cbl7(x4)  # shape=(None,256,40,40)
        x5 = self.upsample2(x4)  # shape=(None,256,80,80)
        x5 = torch.concat([x1, x5], 1)  # shape=(None,512,80,80)
        x5 = self.csp_cbl2(x5)  # shape=(None,256,80,80)
        out_small = self.conv1(x5)  # shape=(None,255,80,80)
        x5 = self.cbl8(x5)  # shape=(None,256,40,40)
        x5 = torch.concat([x4, x5], 1)  # shape=(None,512,40,40)
        x5 = self.csp_cbl3(x5)  # shape=(None,512,40,40)
        out_medium = self.conv2(x5)  # shape=(None,255,40,40)
        x5 = self.cbl9(x5)  # shape=(None,512,20,20)
        x5 = torch.concat([x3, x5], 1)  # shape=(None,1024,20,20)
        x5 = self.csp_cbl4(x5)  # shape=(None,1024,20,20)
        out_big = self.conv3(x5)  # shape=(None,255,20,20)
        return out_small, out_medium, out_big

    def compute_loss(self, pred, true) -> torch.Tensor:
        loss_total = None
        for i in range(3):
            # Possbility Loss
            batch_size = pred[i].shape[0]
            # loss_possibility = torch.mean(torch.square(pred[i][:, 0, ...] - true[i][:, 0, ...]), 0)
            loss_possibility = torch.square(pred[i][:, 0, ...] - true[i][:, 0, ...])
            loss_possibility = torch.sum(loss_possibility)
            # GIoU Loss

            mask_hasObj = (true[i][:, 0:1, ...] > 0).expand(batch_size, self.output_vector, true[i].shape[2], true[i].shape[3])
            hasObj_pred = pred[i][mask_hasObj].reshape(1, self.output_vector, -1)
            hasObj_true = true[i][mask_hasObj].reshape(1, self.output_vector, -1)

            pred_rect_x_min = hasObj_pred[:, 1, ...] - hasObj_pred[:, 3, ...]/2
            pred_rect_x_max = hasObj_pred[:, 1, ...] + hasObj_pred[:, 3, ...]/2
            pred_rect_y_min = hasObj_pred[:, 2, ...] - hasObj_pred[:, 4, ...]/2
            pred_rect_y_max = hasObj_pred[:, 2, ...] + hasObj_pred[:, 4, ...]/2

            true_rect_x_min = hasObj_true[:, 1, ...] - hasObj_true[:, 3, ...]/2
            true_rect_x_max = hasObj_true[:, 1, ...] + hasObj_true[:, 3, ...]/2
            true_rect_y_min = hasObj_true[:, 2, ...] - hasObj_true[:, 4, ...]/2
            true_rect_y_max = hasObj_true[:, 2, ...] + hasObj_true[:, 4, ...]/2

            # _w = torch.min(pred_rect_x_max, true_rect_x_max) - torch.max(pred_rect_x_min, true_rect_x_min)
            # intersection_w = torch.max(torch.zeros((_w.shape), device=device), _w)
            # _h = torch.min(pred_rect_y_max, true_rect_y_max) - torch.max(pred_rect_y_min, true_rect_y_min)
            # intersection_h = torch.max(torch.zeros((_h.shape), device=device), _h)
            intersection_w = torch.min(pred_rect_x_max, true_rect_x_max) - torch.max(pred_rect_x_min, true_rect_x_min)
            intersection_h = torch.min(pred_rect_y_max, true_rect_y_max) - torch.max(pred_rect_y_min, true_rect_y_min)
            intersection_w[intersection_w < 0.0] = 0.0
            intersection_h[intersection_h < 0.0] = 0.0
            s_common = intersection_w[:, :] * intersection_h[:, :]
            s1 = hasObj_pred[:, 3, :] * hasObj_pred[:, 4, :]
            s2 = hasObj_true[:, 3, :] * hasObj_true[:, 4, :]
            s_added = s1 + s2 - s_common
            IoU = s_common / s_added
            wrap_w = torch.max(torch.stack([pred_rect_x_min, pred_rect_x_max, true_rect_x_min, true_rect_x_max], -1), -1).values - \
                torch.min(torch.stack([pred_rect_x_min, pred_rect_x_max, true_rect_x_min, true_rect_x_max], -1), -1).values
            wrap_h = torch.max(torch.stack([pred_rect_y_min, pred_rect_y_max, true_rect_y_min, true_rect_y_max], -1), -1).values - \
                torch.min(torch.stack([pred_rect_y_min, pred_rect_y_max, true_rect_y_min, true_rect_y_max], -1), -1).values

            s_wrap = wrap_w[:, :] * wrap_h[:, :]
            GIoU = IoU - (s_wrap - s_added) / s_wrap
            loss_giou = torch.mean(1 - GIoU, 0)
            loss_giou = torch.sum(loss_giou)
            # Classification Loss
            # loss_classification = torch.mean(torch.square(hasObj_pred[:, 5:, :] - hasObj_true[:, 5:, :]), 0)
            loss_classification = torch.square(hasObj_pred[:, 5:, :] - hasObj_true[:, 5:, :])
            loss_classification = torch.sum(loss_classification)
            if loss_total is None:
                loss_total = loss_possibility + loss_giou + loss_classification
            else:
                loss_total = loss_total + loss_possibility + loss_giou + loss_classification
        return loss_total
