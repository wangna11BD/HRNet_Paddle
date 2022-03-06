# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from itertools import cycle, islice
from collections import abc
import paddle
import paddle.nn as nn
from lib.utils.workspace import register, serializable

__all__ = ['KeyPointMSELoss']

@register
@serializable
class KeyPointMSELoss(nn.Layer):
    def __init__(self, use_target_weight=True, loss_scale=0.5):
        """
        KeyPointMSELoss layer

        Args:
            use_target_weight (bool): whether to use target weight
        """
        super(KeyPointMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_scale = loss_scale

    def forward(self, output, records):
        target = records['target']
        target_weight = records['target_weight']
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.loss_scale * self.criterion(
                    heatmap_pred.multiply(target_weight[:, idx]),
                    heatmap_gt.multiply(target_weight[:, idx]))
            else:
                loss += self.loss_scale * self.criterion(heatmap_pred,
                                                         heatmap_gt)
        keypoint_losses = dict()
        keypoint_losses['loss'] = loss / num_joints
        return keypoint_losses

@register
@serializable
class DistDMLLoss(nn.Layer):
    def __init__(self, use_target_weight=True, loss_scale=0.5, act="softmax", eps=1e-12):
        super().__init__()
        self.use_target_weight = use_target_weight
        self.loss_scale = loss_scale

        assert act in ["softmax", "sigmoid", None]
        if act == "softmax":
            self.act = nn.Softmax(axis=-1)
        elif act == "sigmoid":
            self.act = nn.Sigmoid()
        else:
            self.act = None
        self.eps = eps

    def _kldiv(self, x, target):
        class_num = x.shape[-1]
        cost = target * paddle.log(
            (target + self.eps) / (x + self.eps)) * class_num
        return cost

    def _forward(self, x, target):
        if self.act is not None:
            x = self.act(x)
            target = self.act(target)
        loss = self._kldiv(x, target) + self._kldiv(target, x)
        loss = loss / 2
        loss = paddle.mean(loss)
        return loss

    def forward(self, output, records):
        pass


@register
@serializable
class DistMSELoss(nn.Layer):
    def __init__(self, use_target_weight=True, loss_scale=0.5):
        super().__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.loss_scale = loss_scale

    def forward(self, output_dict, records):
        output = output_dict["student_out"]
        target = output_dict["teacher_out"]
        target_weight = records['target_weight']
        batch_size = output.shape[0]
        num_joints = output.shape[1]
        heatmaps_pred = output.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        heatmaps_gt = target.reshape(
            (batch_size, num_joints, -1)).split(num_joints, 1)
        loss = 0
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += self.loss_scale * self.criterion(
                    heatmap_pred.multiply(target_weight[:, idx]),
                    heatmap_gt.multiply(target_weight[:, idx]))
            else:
                loss += self.loss_scale * self.criterion(heatmap_pred,
                                                         heatmap_gt)
        keypoint_losses = dict()
        keypoint_losses['loss'] = loss / num_joints
        return keypoint_losses