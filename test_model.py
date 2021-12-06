#!/usr/bin/env python3
# Written by feymanpriv(547559398@qq.com)

import init_path
import sys
import numpy as np
import paddle

from core.config import cfg
import core.config as config
from model.dolg_model import DOLG


def main():
    model = DOLG()
    model.eval()
    print(model)
    
    # load parameters
    state_dict = paddle.load(cfg.TEST.WEIGHTS)
    model.set_dict(state_dict)
    
    dummy_input = np.ones((1,3,512,512), dtype='float32')
    dummy_input = paddle.to_tensor(dummy_input)
    fea = model(dummy_input)
    fea_numpy = fea.numpy()
    print(fea_numpy.shape)


if __name__ == '__main__':
    print(sys.argv)
    config.load_cfg_fom_args("Extract feature.")
    config.assert_and_infer_cfg()
    cfg.freeze()
    
    main()
