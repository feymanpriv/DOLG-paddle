#!/usr/bin/env python3
# Written by feymanpriv(547559398@qq.com)

import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from core.config import cfg
from model.resnet import ResNet


""" Dolg models """

class DOLG(nn.Layer):
    """ DOLG model """
    def __init__(self):
        """ """
        super(DOLG, self).__init__()
        self.pool_l= nn.AdaptiveAvgPool2D((1, 1)) 
        self.pool_g = GeneralizedMeanPooling(norm=3.0)
        self.fc_t = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.S3_DIM)
        self.fc = nn.Linear(cfg.MODEL.S4_DIM, cfg.MODEL.HEADS.REDUCTION_DIM)
        self.globalmodel = ResNet()
        self.localmodel = SpatialAttention2d(cfg.MODEL.S3_DIM)
    
    def forward(self, x):
        """ Global and local orthogonal fusion """
        f3, f4 = self.globalmodel(x)
        fl, _ = self.localmodel(f3)
        
        fg_o = self.pool_g(f4)
        fg_o = fg_o.reshape([fg_o.shape[0], cfg.MODEL.S4_DIM])
        
        fg = self.fc_t(fg_o)
        fg_norm = paddle.norm(fg, p=2, axis=1)
        
        proj = paddle.bmm(fg.unsqueeze(1), paddle.flatten(fl, start_axis=2))
        proj = paddle.bmm(fg.unsqueeze(2), proj).reshape(fl.shape)
        proj = proj / (fg_norm * fg_norm).reshape([-1, 1, 1, 1])
        orth_comp = fl - proj

        fo = self.pool_l(orth_comp)
        fo = fo.reshape([fo.shape[0], cfg.MODEL.S3_DIM])

        final_feat = paddle.concat((fg, fo), 1)
        final_feat = self.fc(final_feat)
        return final_feat

    
class GeneralizedMeanPooling(nn.Layer):
    """Applies a 2D power-average adaptive pooling over an input signal composed of several input planes.
    The function computed is: :math:`f(X) = pow(sum(pow(X, p)), 1/p)`
        - At p = infinity, one gets Max Pooling
        - At p = 1, one gets Average Pooling
    The output is of size H x W, for any input size.
    The number of output features is equal to the number of input planes.
    Args:
        output_size: the target output size of the image of the form H x W.
                     Can be a tuple (H, W) or a single H for a square image H x H
                     H and W can be either a ``int``, or ``None`` which means the size will
                     be the same as that of the input.
    """

    def __init__(self, norm, output_size=1, eps=1e-6):
        """ just for infer """
        super(GeneralizedMeanPooling, self).__init__()
        assert norm > 0
        self.p = float(norm)
        self.output_size = output_size
        self.eps = eps

    def forward(self, x):
        """ """
        out = paddle.clip(x, min=self.eps).pow(self.p)
        return paddle.nn.functional.adaptive_avg_pool2d(out, self.output_size).pow(1. / self.p)

    def __repr__(self):
        """ """
        return self.__class__.__name__ + '(' \
               + str(self.p) + ', ' \
               + 'output_size=' + str(self.output_size) + ')'
    

class SpatialAttention2d(nn.Layer):
    """ SpatialAttention2D """
    def __init__(self, in_c, with_aspp=cfg.MODEL.WITH_MA):
        """ """
        super(SpatialAttention2d, self).__init__()

        self.with_aspp = with_aspp
        if self.with_aspp:
            self.aspp = ASPP(cfg.MODEL.S3_DIM)
        self.conv1 = nn.Conv2D(in_c, cfg.MODEL.S3_DIM, 1, 1, data_format='NCHW')
        self.bn = nn.BatchNorm2D(cfg.MODEL.S3_DIM, momentum=cfg.BN.MOM, epsilon=cfg.BN.EPS)
        self.act1 = nn.ReLU()
        self.conv2 = nn.Conv2D(cfg.MODEL.S3_DIM, 1, 1, 1, data_format='NCHW')
        self.softplus = nn.Softplus(beta=1, threshold=20) 

    def forward(self, x):
        '''
        x : spatial feature map. (b x c x w x h)
        att : softplus attention score 
        '''
        if self.with_aspp:
            x = self.aspp(x)
        x = self.conv1(x)
        x = self.bn(x)
        
        feature_map_norm = F.normalize(x, p=2, axis=1)
         
        x = self.act1(x)
        x = self.conv2(x)

        att_score = self.softplus(x)
        att = att_score.expand_as(feature_map_norm)
        x = att * feature_map_norm
        return x, att_score
    
    def __repr__(self):
        """ """
        return self.__class__.__name__
    
    
class ASPP(nn.Layer):
    """ Atrous Spatial Pyramid Pooling Module """
    def __init__(self, in_c):
        """ """
        super(ASPP, self).__init__()

        self.aspp = []
        self.aspp.append(nn.Conv2d(in_c, 512, 1, 1))

        for dilation in [6, 12, 18]:
            _padding = (dilation * 3 - dilation) // 2
            self.aspp.append(nn.Conv2D(in_c, 512, 3, 1, padding=_padding, dilation=dilation))
        self.aspp = nn.LayerList(self.aspp)

        self.im_pool = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                     nn.Conv2d(in_c, 512, 1, 1),
                                     nn.ReLU())
        conv_after_dim = 512 * (len(self.aspp)+1)
        self.conv_after = nn.Sequential(nn.Conv2d(conv_after_dim, cfg.MODEL.S3_DIM, 1, 1), nn.ReLU())

    def forward(self, x):
        """  """
        h, w = x.shape[2], x.shape[3]
        aspp_out = [F.interpolate(self.im_pool(x), scale_factor=(h,w), mode="bilinear", align_corners=False)]
        for i in range(len(self.aspp)):
            aspp_out.append(self.aspp[i](x))
        aspp_out = paddle.concat(aspp_out, 1)
        x = self.conv_after(aspp_out)
        return x
