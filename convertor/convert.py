#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
# Written by feymanpriv

#import init_path
import sys, os
import numpy as np
import cv2

import torch
import paddle

from core.config import cfg
import core.config as config
from model.t_dolg_model import TDOLG
from model.dolg_model import DOLG

T_MODEL_WEIGHTS = './weights/r101_dolg_512_wo_aspp.pyth'
P_MODEL_WEIGHTS = './weights/PP_DOLG_R101.pdparams'


def build_tdolg():
    t_model = TDOLG()
    #print(t_model)
    t_model.eval()
    return t_model


def build_ppdolg():
    p_model = DOLG()
    #print(p_model)
    p_model.eval() 
    return p_model


def torch2paddle():
    checkpoint = torch.load(T_MODEL_WEIGHTS, map_location="cpu")
    checkpoint_state_dict = checkpoint["model_state"]
    #print(checkpoint_state_dict.keys())
    p_model = build_ppdolg()
    t_model = build_tdolg()
    print ("torch build model has {} keys".format(len(t_model.state_dict().keys())))
    #print (t_model.state_dict().keys())

    new_torch_dict = {}
    for k, v in checkpoint_state_dict.items():
        if 'running_mean' in k:
            k = k.replace('running_mean', '_mean')
        elif 'running_var' in k:
            k = k.replace('running_var', '_variance')
        new_torch_dict[k] = v.detach().numpy()
    #print (new_torch_dict.keys())
    #print(new_torch_dict['globalmodel.stem.conv.weight'].shape) #test
    print("checkpoint has {} keys".format(len(checkpoint_state_dict)))
    print("new_torch_dict has {} keys".format(len(new_torch_dict)))
    
    paddle_state_dict = p_model.state_dict()
    print("paddle build model has {} keys".format(len(paddle_state_dict.keys())))
    not_loaded_dict = {}
    #for k in p_model.state_dict():
    for k, v in new_torch_dict.items():
        if k in p_model.state_dict():
            if 'fc' in k and 'weight' in k:
                paddle_state_dict[k].set_value(v.T)
            else:
                paddle_state_dict[k].set_value(v)
            print(k)
            print(type(v), v.shape)
        else:
            not_loaded_dict[k] = v
    print("#####"*20)
    for k in not_loaded_dict:
        print (k)
    print(len(not_loaded_dict.keys()))
        
    paddle.save(paddle_state_dict, P_MODEL_WEIGHTS)
    
    
def infer_torch():
    model = build_tdolg()
    load_checkpoint(T_MODEL_WEIGHTS, model)
    new_input = np.ones((1,3,448,448), dtype='float32')
    new_input = torch.from_numpy(new_input)
    fea = model(new_input)
    fea_numpy = fea.detach().numpy()
    print(fea_numpy[0, :10])
    
    
def infer_paddle():
    model_path = P_MODEL_WEIGHTS
    model = build_ppdolg()
    
    state_dict = paddle.load(model_path)
    model.set_dict(state_dict)
    
    new_input = np.ones((1,3,448,448), dtype='float32')
    new_input = paddle.to_tensor(new_input)
    fea = model(new_input)
    fea_numpy = fea.numpy()
    print(fea_numpy.shape)
    print(fea_numpy[0, :10])


def load_checkpoint(checkpoint_file, model):
    """Loads the checkpoint from the given file."""
    err_str = "Checkpoint '{}' not found"
    assert os.path.exists(checkpoint_file), err_str.format(checkpoint_file)
    checkpoint = torch.load(checkpoint_file, map_location="cpu")
    try:
        state_dict = checkpoint["model_state"]
    except KeyError:
        state_dict = checkpoint
    # Account for the DDP wrapper in the multi-gpu setting
    ms = model
    model_dict = ms.state_dict()
    pretrained_dict = {k: v for k, v in state_dict.items() 
                       if k in model_dict and model_dict[k].size() == v.size()}
    if len(pretrained_dict) == len(state_dict):
        print('All params loaded')
    else:
        print('construct model total {} keys and pretrin model total {} keys.' \
               .format(len(model_dict), len(state_dict)))
        print('{} pretrain keys load successfully.'.format(len(pretrained_dict)))
        not_loaded_keys = [k for k in state_dict.keys() 
                                if k not in pretrained_dict.keys()]
        print(('%s, ' * (len(not_loaded_keys) - 1) + '%s') % tuple(not_loaded_keys))

    model_dict.update(pretrained_dict)
    ms.load_state_dict(model_dict)
    #ms.load_state_dict(checkpoint["model_state"])
    return checkpoint


if __name__ == '__main__':
    print(sys.argv)
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()

    torch2paddle()
    #infer_paddle()
    #infer_torch()
