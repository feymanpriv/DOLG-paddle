#!/usr/bin/env python3
# Written by feymanpriv(yangminbupt@outlook.com)

""" inference """

import sys
import numpy as np
import cv2
import math

import paddle


_MEAN = [0.48145466, 0.4578275, 0.40821073]
_SD = [0.26862954, 0.26130258, 0.27577711]


preprocess_ops = [
    [resize, 512],
    [center_crop, 512],
    [convert_color],
    [swapaxisimg2blob],
    [convert_to],
    [normalize, _MEAN, _SD],
]



def predict(imgpath):
    """infer"""
    img = cv2.imread(imgpath)
    img = preprocess(img, preprocess_ops)
    data = np.expand_dims(img, axis=0)

    paddle.enable_static()
    place = fluid.CPUPlace()
    exe = fluid.Executor(place)
    prog, feed_names, fetch_targets = fluid.io.load_inference_model(
                        "./DOLG-infermodel/infermodel50", exe, 'model','params')
    fetch_list_name = [a.name for a in fetch_targets]
    print(fetch_list_name)

    #data = np.ones((1,3,512,512))
    result = exe.run(prog, fetch_list=fetch_targets, 
                     feed={'image': data})
    print(result)



def normalize(im, mean, std):
    """Performs per-channel normalization (CHW format)."""
    for i in range(im.shape[0]):
        im[i] = im[i] - mean[i]
        im[i] = im[i] / std[i]
    return im


def zero_pad(im, pad_size):
    """Performs zero padding (CHW format)."""
    pad_width = ((0, 0), (pad_size, pad_size), (pad_size, pad_size))
    return np.pad(im, pad_width, mode="constant")


def resize(im, size):
    """Performs scaling (HWC format)."""
    h, w = im.shape[:2]
    if (w <= h and w == size) or (h <= w and h == size):
        return im
    h_new, w_new = size, size
    if w < h:
        h_new = int(math.floor((float(h) / w) * size))
    else:
        w_new = int(math.floor((float(w) / h) * size))
    im = cv2.resize(im, (w_new, h_new), interpolation=cv2.INTER_LINEAR)
    return im.astype(np.float32)


def center_crop(im, size):
    """Performs center cropping (HWC format)."""
    h, w = im.shape[:2]
    y = int(math.ceil((h - size) / 2))
    x = int(math.ceil((w - size) / 2))
    im_crop = im[y: (y + size), x: (x + size), :]
    assert im_crop.shape[:2] == (size, size)
    return im_crop


def swapaxisimg2blob(im):
    """Performs HWC -> CHW"""
    return im.transpose([2, 0, 1]).copy()


def convert_color(im):
    """Performs BGR -> RGB"""
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


def convert_to(im):
    """Convert to array"""
    im = im / 255.0
    return im


def preprocess(im, operators):
    """Wrapper"""
    for op in operators:
        func, args = op[0], op[1:]
        im = func(im, *args)
    return im



if __name__ == "__main__":
    if len(sys.argv)>1 :
        func = getattr(sys.modules[__name__], sys.argv[1])
        func(*sys.argv[2:])
    else:
        print('infer.py predict your/image/path')
