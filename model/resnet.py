#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license.
#
# Written by feymanpriv(yangminbupt@outlook.com)

"""ResNe(X)t model backbones."""

import paddle
import paddle.nn as nn
from core.config import cfg

# Stage depths for ImageNet models
_IN_STAGE_DS = {50: (3, 4, 6, 3), 101: (3, 4, 23, 3), 152: (3, 8, 36, 3)}


def get_trans_fun(name):
    """Retrieves the transformation function by name."""
    trans_funs = {
        "basic_transform": BasicTransform,
        "bottleneck_transform": BottleneckTransform,
    }
    err_str = "Transformation function '{}' not supported"
    assert name in trans_funs.keys(), err_str.format(name)
    return trans_funs[name]


class BasicTransform(nn.Layer):
    """Basic transformation: 3x3, BN, ReLU, 3x3, BN."""

    def __init__(self, w_in, w_out, stride, w_b=None, num_gs=1):
        """ """
        err_str = "Basic transform does not support w_b and num_gs options"
        assert w_b is None and num_gs == 1, err_str
        super(BasicTransform, self).__init__()
        self.a = nn.Conv2D(w_in, w_out, 3, stride=stride, padding=1, bias_attr=False, data_format='NCHW')
        self.a_bn = nn.BatchNorm2d(w_out, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.a_relu = nn.ReLU()
        self.b = nn.Conv2D(w_out, w_out, 3, stride=1, padding=1, bias_attr=False, data_format='NCHW')
        self.b_bn = nn.BatchNorm2D(w_out, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.b_bn.final_bn = True

    def forward(self, x):
        """ """
        for layer in self.children():
            x = layer(x)
        return x


class BottleneckTransform(nn.Layer):
    """Bottleneck transformation: 1x1, BN, ReLU, 3x3, BN, ReLU, 1x1, BN."""

    def __init__(self, w_in, w_out, stride, w_b, num_gs):
        """ """
        super(BottleneckTransform, self).__init__()
        # MSRA -> stride=2 is on 1x1; TH/C2 -> stride=2 is on 3x3
        (s1, s3) = (stride, 1) if cfg.RESNET.STRIDE_1X1 else (1, stride)
        self.a = nn.Conv2D(w_in, w_b, 1, stride=s1, padding=0, bias_attr=False, data_format='NCHW')
        self.a_bn = nn.BatchNorm2D(w_b, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.a_relu = nn.ReLU()
        self.b = nn.Conv2D(w_b, w_b, 3, stride=s3, padding=1, groups=num_gs, bias_attr=False, data_format='NCHW')
        self.b_bn = nn.BatchNorm2D(w_b, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.b_relu = nn.ReLU()
        self.c = nn.Conv2D(w_b, w_out, 1, stride=1, padding=0, bias_attr=False, data_format='NCHW')
        self.c_bn = nn.BatchNorm2D(w_out, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.c_bn.final_bn = True

    def forward(self, x):
        """ """
        for layer in self.children():
            x = layer(x)
        return x


class ResBlock(nn.Layer):
    """Residual block: x + F(x)."""

    def __init__(self, w_in, w_out, stride, trans_fun, w_b=None, num_gs=1):
        """ """
        super(ResBlock, self).__init__()
        # Use skip connection with projection if shape changes
        self.proj_block = (w_in != w_out) or (stride != 1)
        if self.proj_block:
            self.proj = nn.Conv2D(w_in, w_out, 1, stride=stride, padding=0, bias_attr=False, data_format='NCHW')
            self.bn = nn.BatchNorm2D(w_out, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.f = trans_fun(w_in, w_out, stride, w_b, num_gs)
        self.relu = nn.ReLU()

    def forward(self, x):
        """ """
        if self.proj_block:
            x = self.bn(self.proj(x)) + self.f(x)
        else:
            x = x + self.f(x)
        x = self.relu(x)
        return x


class ResStage(nn.Layer):
    """Stage of ResNet."""

    def __init__(self, w_in, w_out, stride, d, w_b=None, num_gs=1):
        """ """
        super(ResStage, self).__init__()
        for i in range(d):
            b_stride = stride if i == 0 else 1
            b_w_in = w_in if i == 0 else w_out
            trans_fun = get_trans_fun(cfg.RESNET.TRANS_FUN)
            res_block = ResBlock(b_w_in, w_out, b_stride, trans_fun, w_b, num_gs)
            self.add_sublayer("b{}".format(i + 1), res_block)

    def forward(self, x):
        """ """
        for block in self.children():
            x = block(x)
        return x


class ResStemIN(nn.Layer):
    """ResNet stem for ImageNet: 7x7, BN, ReLU, MaxPool."""

    def __init__(self, w_in, w_out):
        """ """
        super(ResStemIN, self).__init__()
        self.conv = nn.Conv2D(w_in, w_out, 7, stride=2, padding=3, bias_attr=False, data_format='NCHW')
        self.bn = nn.BatchNorm2D(w_out, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2D(3, stride=2, padding=1)

    def forward(self, x):
        """ """
        for layer in self.children():
            x = layer(x)
        return x


class ResNet(nn.Layer):
    """ResNet model."""

    def __init__(self):
        """ """
        super(ResNet, self).__init__()
        self._construct()

    def _construct(self):
        """ """
        g, gw = cfg.RESNET.NUM_GROUPS, cfg.RESNET.WIDTH_PER_GROUP
        (d1, d2, d3, d4) = _IN_STAGE_DS[cfg.MODEL.DEPTH]
        w_b = gw * g
        self.stem = ResStemIN(3, 64)
        self.s1 = ResStage(64, 256, stride=1, d=d1, w_b=w_b, num_gs=g)
        self.s2 = ResStage(256, 512, stride=2, d=d2, w_b=w_b * 2, num_gs=g)
        self.s3 = ResStage(512, 1024, stride=2, d=d3, w_b=w_b * 4, num_gs=g)
        self.s4 = ResStage(1024, 2048, stride=2, d=d4, w_b=w_b * 8, num_gs=g)
        #self.head = ResHead(2048, nc=cfg.MODEL.HEADS.REDUCTION_DIM)

    def forward(self, x):
        """ """
        x = self.stem(x)
        x1 = self.s1(x)
        x2 = self.s2(x1)
        x3 = self.s3(x2)
        x4 = self.s4(x3)
        #x = self.head(x4)
        return x3, x4
