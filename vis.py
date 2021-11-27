# -*- coding: utf-8 -*-
# @Time    : 2021/11/18 22:40
# @Author  : zhao pengfei
# @Email   : zsonghuan@gmail.com
# @File    : run_mae_vis.py
# --------------------------------------------------------
# Based on BEiT, timm, DINO and DeiT code bases
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/rwightman/pytorch-image-models/tree/master/timm
# https://github.com/facebookresearch/deit
# https://github.com/facebookresearch/dino
# --------------------------------------------------------'

import argparse
import datetime
import numpy as np
import time
import paddle
import json
import os

from PIL import Image

from pathlib import Path


import utils
import mae
from mask import RandomMaskingGenerator
from mae_bak import build_mae as build_model
from config import get_config
from config import update_config
from einops import rearrange
from datasets import get_val_transforms
import os
from PIL import Image
import paddle
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
def get_args():
    parser = argparse.ArgumentParser('MAE visualization reconstruction script', add_help=False)
    parser.add_argument('-cfg', type=str, help='config path')
    parser.add_argument('-img_path', type=str, help='input image path')
    parser.add_argument('-dataset', type=str, help='dataset type') 
    parser.add_argument('-save_path', type=str, help='save image path')
    parser.add_argument('-model_path', type=str, help='checkpoint path of model')
    parser.add_argument('-batch_size', default=1, type=int,
                        help='images input size for backbone')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size for backbone')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--imagenet_default_mean_and_std', default=True, action='store_true')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='ratio of the visual tokens/patches need be masked')
    # Model parameters
    parser.add_argument('--model', default='MAE', type=str, metavar='MODEL',
                        help='Name of model to vis')
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    
    return parser.parse_args()

def to_pil_image(pic, mode='RGB'):
    """Convert a tensor or an ndarray to PIL Image.


    Args:
        pic (Tensor or numpy.ndarray): Image to be converted to PIL Image.
        mode (`PIL.Image mode`_): color space and pixel depth of input data (optional).

    .. _PIL.Image mode: https://pillow.readthedocs.io/en/latest/handbook/concepts.html#concept-modes

    Returns:
        PIL Image: Image converted to PIL Image.
    """
    if not(isinstance(pic, paddle.Tensor) or isinstance(pic, np.ndarray)):
        raise TypeError('pic should be Tensor or ndarray. Got {}.'.format(type(pic)))

    elif isinstance(pic, paddle.Tensor):
        if pic.ndimension() not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndimension()))

        elif pic.ndimension() == 2:
            # if 2D image, add channel dimension (CHW)
            pic = pic.unsqueeze(0)

    elif isinstance(pic, np.ndarray):
        if pic.ndim not in {2, 3}:
            raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

        elif pic.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            pic = np.expand_dims(pic, 2)

    npimg = pic
    #if isinstance(pic, np.ndarray) and mode != 'F':
    #    pic = pic.mul(255).byte()
    if isinstance(pic, paddle.Tensor):
        npimg = np.transpose(pic.numpy(), (1, 2, 0))

    if not isinstance(npimg, np.ndarray):
        raise TypeError('Input pic must be a paddle.Tensor or NumPy ndarray, ' +
                        'not {}'.format(type(npimg)))

    if npimg.shape[2] == 1:
        expected_mode = None
        npimg = npimg[:, :, 0]
        if npimg.dtype == np.uint8:
            expected_mode = 'L'
        elif npimg.dtype == np.int16:
            expected_mode = 'I;16'
        elif npimg.dtype == np.int32:
            expected_mode = 'I'
        elif npimg.dtype == np.float32:
            expected_mode = 'F'
        if mode is not None and mode != expected_mode:
            raise ValueError("Incorrect mode ({}) supplied for input type {}. Should be {}"
                             .format(mode, np.dtype, expected_mode))
        mode = expected_mode

    elif npimg.shape[2] == 2:
        permitted_2_channel_modes = ['LA']
        if mode is not None and mode not in permitted_2_channel_modes:
            raise ValueError("Only modes {} are supported for 2D inputs".format(permitted_2_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'LA'

    elif npimg.shape[2] == 4:
        permitted_4_channel_modes = ['RGBA', 'CMYK', 'RGBX']
        if mode is not None and mode not in permitted_4_channel_modes:
            raise ValueError("Only modes {} are supported for 4D inputs".format(permitted_4_channel_modes))

        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGBA'
    else:
        permitted_3_channel_modes = ['RGB', 'YCbCr', 'HSV']
        if mode is not None and mode not in permitted_3_channel_modes:
            raise ValueError("Only modes {} are supported for 3D inputs".format(permitted_3_channel_modes))
        if mode is None and npimg.dtype == np.uint8:
            mode = 'RGB'

    if mode is None:
        raise TypeError('Input type {} is not supported'.format(npimg.dtype))

    return Image.fromarray(npimg, mode=mode)

args =get_args()
# get default config
config = get_config()
# update config by arguments
#config = update_config(config, args)

def get_model(args):
    print(f"Creating model: {args.model}")
    model = build_model(config)
    return model


def main(args):
    print(args)


    model = get_model(args)
    patch_size = 16
    print("Patch size = %s" % str(patch_size))
    args.window_size = (args.input_size // patch_size, args.input_size // patch_size)
    args.patch_size = patch_size

    checkpoint = paddle.load(args.model_path)
    model.set_dict(checkpoint)
    model.eval()

    with open(args.img_path, 'rb') as f:
        img = Image.open(f)
        img.convert('RGB')
        print("img path:", args.img_path)

    transforms = get_val_transforms(config)
    img= transforms(img)
    masked_position_generator = RandomMaskingGenerator(config.MAE.WINDOW_SIZE, config.MAE.MASK_RADIO)
    bool_masked_pos =masked_position_generator()
    with paddle.no_grad():
        img = img[None, :]
        bool_masked_pos = bool_masked_pos[None, :]
        bool_masked_pos = bool_masked_pos.astype(np.bool).reshape([-1,196])
        bool_masked_pos = paddle.to_tensor(bool_masked_pos)
        #print(img.shape,bool_masked_pos.shape)
        outputs = model(img, bool_masked_pos)

        #save original img
        mean = paddle.to_tensor([0.5,0.5,0.5])[None, :, None, None]
        std = paddle.to_tensor([0.5, 0.5, 0.5])[None, :, None, None]
        ori_img = (img * std + mean).numpy()  # in [0, 1]

        img =to_pil_image((ori_img[0, :]*255).astype(np.int8).transpose(1,2,0))
        img.save(f"{args.save_path}/ori_img.jpg")

        img_squeeze = rearrange(ori_img, 'b c (h p1) (w p2) -> b (h w) (p1 p2) c', p1=patch_size, p2=patch_size)
        img_norm = (img_squeeze - np.sqrt(img_squeeze.mean(axis=-2,keepdims=True) / img_squeeze.var(axis=-2,  keepdims=True)) + 1e-6)
        img_patch = rearrange(img_norm, 'b n p c -> b n (p c)')
        img_patch[bool_masked_pos.numpy()] = outputs

        #make mask
        mask = paddle.ones_like(paddle.to_tensor(img_patch)).numpy()
        mask[bool_masked_pos.numpy()] = 0
        mask = rearrange(mask, 'b n (p c) -> b n p c', c=3)
        mask = rearrange(mask, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)

        #save reconstruction img
        rec_img = rearrange(img_patch, 'b n (p c) -> b n p c', c=3)
        rec_img = rec_img * (np.sqrt(img_squeeze.var(axis=-2,  keepdims=True)) + 1e-6) + img_squeeze.mean(axis=-2,keepdims =True)
        rec_img = rearrange(rec_img, 'b (h w) (p1 p2) c -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size, h=14, w=14)
        img = to_pil_image((rec_img[0, :].clip(0,0.996)*255).astype(np.int8).transpose(1,2,0))
        img.save(f"{args.save_path}/rec_img.jpg")

        #save random mask img
        img_mask = rec_img * mask

        img = to_pil_image((img_mask[0, :]*255).astype(np.int8).transpose(1,2,0))
        img.save(f"{args.save_path}/mask_img.jpg")

if __name__ == '__main__':
    opts = get_args()
    main(opts)



