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

import paddle
import paddle.nn as nn
from paddle.nn.initializer import Normal, Constant
import copy
import numpy as np
import math
import cv2

from lib.utils.keypoint_utils import transform_preds
from lib.utils.workspace import register, create
from lib.models.keypoint_hrnet import BaseArch, Conv2d
from lib.models.hrnet import HRNet
from lib.models.mobilenet_v3 import MobileNetV3
from lib.models.loss import DistMSELoss

__all__ = ['DistTopDownHRNet']

@register
class DistTopDownHRNet(BaseArch):
    __category__ = 'architecture'
    __inject__ = ['loss', 'dist_loss']

    def __init__(self,
                 student_width,
                 teacher_width,
                 num_joints,
                 student_backbone='HRNet',
                 teacher_backbone='HRNet',
                 loss='KeyPointMSELoss',
                 dist_loss="DistMSELoss",
                 post_process='HRNetPostProcess',
                 flip_perm=None,
                 flip=True,
                 shift_heatmap=True,
                 use_dark=True,
                 freeze_teacher_params=True):
        """
        HRNet network, see https://arxiv.org/abs/1902.09212

        Args:
            backbone (nn.Layer): backbone instance
            post_process (object): `HRNetPostProcess` instance
            flip_perm (list): The left-right joints exchange order list
            use_dark(bool): Whether to use DARK in post processing
        """
        super().__init__()
        self.student_backbone = student_backbone
        self.backbone = teacher_backbone
        self.student_final_conv = Conv2d(student_width, num_joints, 1, 1, 0, bias=True)
        self.final_conv = Conv2d(teacher_width, num_joints, 1, 1, 0, bias=True)

        self.post_process = HRNetPostProcess(use_dark)
        self.loss = loss
        self.dist_loss = dist_loss
        self.flip_perm = flip_perm
        self.flip = flip
        self.shift_heatmap = shift_heatmap
        self.deploy = False
        
        if freeze_teacher_params:
            for param in self.backbone.parameters():
                param.trainable = False
            for param in self.final_conv.parameters():
                param.trainable = False

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        # backbone
        arch_config = cfg

        s_backbone_cfg = copy.deepcopy(arch_config["student_backbone"])
        s_name = s_backbone_cfg.pop("name")
        s_backbone = eval(s_name)(**s_backbone_cfg)

        t_backbone_cfg = copy.deepcopy(arch_config["teacher_backbone"])
        t_name = t_backbone_cfg.pop("name")
        t_backbone = eval(t_name)(**t_backbone_cfg)

        return {'student_backbone': s_backbone, "teacher_backbone": t_backbone}

    def _forward(self):
        feats = self.student_backbone(self.inputs)
        s_outputs = self.student_final_conv(feats[0])

        if self.training:
            s_loss = self.loss(s_outputs, self.inputs)
            
            t_feats = self.backbone(self.inputs)
            t_outputs = self.final_conv(t_feats[0])

            dist_loss = self.dist_loss({"student_out": s_outputs, "teacher_out": t_outputs}, self.inputs)
            
            loss_dict = dict()
            loss_dict["loss"] = s_loss["loss"] + dist_loss["loss"]
            loss_dict["loss_student"] = s_loss["loss"]
            loss_dict["loss_dist"] = dist_loss["loss"]
            return loss_dict
            
        elif self.deploy:
            outshape = s_outputs.shape
            max_idx = paddle.argmax(
                s_outputs.reshape(
                    (outshape[0], outshape[1], outshape[2] * outshape[3])),
                axis=-1)
            return s_outputs, max_idx
        else:
            if self.flip:
                self.inputs['image'] = self.inputs['image'].flip([3])
                feats = self.student_backbone(self.inputs)
                output_flipped = self.final_conv(feats[0])
                output_flipped = self.flip_back(output_flipped.numpy(),
                                                self.flip_perm)
                output_flipped = paddle.to_tensor(output_flipped.copy())
                if self.shift_heatmap:
                    output_flipped[:, :, :, 1:] = output_flipped.clone(
                    )[:, :, :, 0:-1]
                s_outputs = (s_outputs + output_flipped) * 0.5
            imshape = (self.inputs['im_shape'].numpy()
                       )[:, ::-1] if 'im_shape' in self.inputs else None
            center = self.inputs['center'].numpy(
            ) if 'center' in self.inputs else np.round(imshape / 2.)
            scale = self.inputs['scale'].numpy(
            ) if 'scale' in self.inputs else imshape / 200.
            outputs = self.post_process(s_outputs, center, scale)
            return outputs

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        res_lst = self._forward()
        outputs = {'keypoint': res_lst}
        return outputs

    def flip_back(self, output_flipped, matched_parts):
        assert output_flipped.ndim == 4,\
                'output_flipped should be [batch_size, num_joints, height, width]'

        output_flipped = output_flipped[:, :, :, ::-1]

        for pair in matched_parts:
            tmp = output_flipped[:, pair[0], :, :].copy()
            output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
            output_flipped[:, pair[1], :, :] = tmp

        return output_flipped


class HRNetPostProcess(object):
    def __init__(self, use_dark=True):
        self.use_dark = use_dark

    def get_max_preds(self, heatmaps):
        '''get predictions from score maps

        Args:
            heatmaps: numpy.ndarray([batch_size, num_joints, height, width])

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 2]), the maximum confidence of the keypoints
        '''
        assert isinstance(heatmaps,
                          np.ndarray), 'heatmaps should be numpy.ndarray'
        assert heatmaps.ndim == 4, 'batch_images should be 4-ndim'

        batch_size = heatmaps.shape[0]
        num_joints = heatmaps.shape[1]
        width = heatmaps.shape[3]
        heatmaps_reshaped = heatmaps.reshape((batch_size, num_joints, -1))
        idx = np.argmax(heatmaps_reshaped, 2)
        maxvals = np.amax(heatmaps_reshaped, 2)

        maxvals = maxvals.reshape((batch_size, num_joints, 1))
        idx = idx.reshape((batch_size, num_joints, 1))

        preds = np.tile(idx, (1, 1, 2)).astype(np.float32)

        preds[:, :, 0] = (preds[:, :, 0]) % width
        preds[:, :, 1] = np.floor((preds[:, :, 1]) / width)

        pred_mask = np.tile(np.greater(maxvals, 0.0), (1, 1, 2))
        pred_mask = pred_mask.astype(np.float32)

        preds *= pred_mask

        return preds, maxvals

    def gaussian_blur(self, heatmap, kernel):
        border = (kernel - 1) // 2
        batch_size = heatmap.shape[0]
        num_joints = heatmap.shape[1]
        height = heatmap.shape[2]
        width = heatmap.shape[3]
        for i in range(batch_size):
            for j in range(num_joints):
                origin_max = np.max(heatmap[i, j])
                dr = np.zeros((height + 2 * border, width + 2 * border))
                dr[border:-border, border:-border] = heatmap[i, j].copy()
                dr = cv2.GaussianBlur(dr, (kernel, kernel), 0)
                heatmap[i, j] = dr[border:-border, border:-border].copy()
                heatmap[i, j] *= origin_max / np.max(heatmap[i, j])
        return heatmap

    def dark_parse(self, hm, coord):
        heatmap_height = hm.shape[0]
        heatmap_width = hm.shape[1]
        px = int(coord[0])
        py = int(coord[1])
        if 1 < px < heatmap_width - 2 and 1 < py < heatmap_height - 2:
            dx = 0.5 * (hm[py][px + 1] - hm[py][px - 1])
            dy = 0.5 * (hm[py + 1][px] - hm[py - 1][px])
            dxx = 0.25 * (hm[py][px + 2] - 2 * hm[py][px] + hm[py][px - 2])
            dxy = 0.25 * (hm[py+1][px+1] - hm[py-1][px+1] - hm[py+1][px-1] \
                + hm[py-1][px-1])
            dyy = 0.25 * (
                hm[py + 2 * 1][px] - 2 * hm[py][px] + hm[py - 2 * 1][px])
            derivative = np.matrix([[dx], [dy]])
            hessian = np.matrix([[dxx, dxy], [dxy, dyy]])
            if dxx * dyy - dxy**2 != 0:
                hessianinv = hessian.I
                offset = -hessianinv * derivative
                offset = np.squeeze(np.array(offset.T), axis=0)
                coord += offset
        return coord

    def dark_postprocess(self, hm, coords, kernelsize):
        '''DARK postpocessing, Zhang et al. Distribution-Aware Coordinate
        Representation for Human Pose Estimation (CVPR 2020).
        '''

        hm = self.gaussian_blur(hm, kernelsize)
        hm = np.maximum(hm, 1e-10)
        hm = np.log(hm)
        for n in range(coords.shape[0]):
            for p in range(coords.shape[1]):
                coords[n, p] = self.dark_parse(hm[n][p], coords[n][p])
        return coords

    def get_final_preds(self, heatmaps, center, scale, kernelsize=3):
        """the highest heatvalue location with a quarter offset in the
        direction from the highest response to the second highest response.

        Args:
            heatmaps (numpy.ndarray): The predicted heatmaps
            center (numpy.ndarray): The boxes center
            scale (numpy.ndarray): The scale factor

        Returns:
            preds: numpy.ndarray([batch_size, num_joints, 2]), keypoints coords
            maxvals: numpy.ndarray([batch_size, num_joints, 1]), the maximum confidence of the keypoints
        """
        coords, maxvals = self.get_max_preds(heatmaps)

        heatmap_height = heatmaps.shape[2]
        heatmap_width = heatmaps.shape[3]

        if self.use_dark:
            coords = self.dark_postprocess(heatmaps, coords, kernelsize)
        else:
            for n in range(coords.shape[0]):
                for p in range(coords.shape[1]):
                    hm = heatmaps[n][p]
                    px = int(math.floor(coords[n][p][0] + 0.5))
                    py = int(math.floor(coords[n][p][1] + 0.5))
                    if 1 < px < heatmap_width - 1 and 1 < py < heatmap_height - 1:
                        diff = np.array([
                            hm[py][px + 1] - hm[py][px - 1],
                            hm[py + 1][px] - hm[py - 1][px]
                        ])
                        coords[n][p] += np.sign(diff) * .25
        preds = coords.copy()

        # Transform back
        for i in range(coords.shape[0]):
            preds[i] = transform_preds(coords[i], center[i], scale[i],
                                       [heatmap_width, heatmap_height])

        return preds, maxvals

    def __call__(self, output, center, scale):
        preds, maxvals = self.get_final_preds(output.numpy(), center, scale)
        outputs = [[
            np.concatenate(
                (preds, maxvals), axis=-1), np.mean(
                    maxvals, axis=1)
        ]]
        return outputs