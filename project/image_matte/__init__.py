"""Image/Video Segment Package."""  # coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2021, 2022(18588220928@163.com) All Rights Reserved.
# ***
# ***    File Author: Dell, 2021年 12月 14日 星期二 00:22:28 CST
# ***
# ************************************************************************************/
#

__version__ = "1.0.0"

import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.nn import functional as F

import todos
from . import isnetdis, vitmatte

import pdb


def get_netdis_model():
    """Create model."""

    model = isnetdis.ISNetDIS()
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_matte.torch"):
    #     model.save("output/image_netdis.torch")

    return model, device


def get_vitmat_model():
    """Create model."""

    model = vitmatte.ViTMatte()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")

    # # make sure model good for C/C++
    # model = torch.jit.script(model)
    # # https://github.com/pytorch/pytorch/issues/52286
    # torch._C._jit_set_profiling_executor(False)
    # # C++ Reference
    # # torch::jit::getProfilingMode() = false;                                                                                                             
    # # torch::jit::setTensorExprFuserEnabled(false);

    # todos.data.mkdir("output")
    # if not os.path.exists("output/image_matte.torch"):
    #     model.save("output/image_vitmat.torch")

    return model, device

def get_vitmat_trace_model():
    """Create model."""

    model = vitmatte.ViTMatte()

    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")

    return model, device


def image_matte_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    device = todos.model.get_device()

    model1, device1 = get_netdis_model()
    model2, device2 = get_vitmat_model()

    model1.to(device)
    model2.to(device)
    print(f"Running models on {device} ...")

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_rgba_tensor(filename)

        # Resize input for big input image
        B, C, H, W = input_tensor.size()
        s = min(min(model1.MAX_H / H, model1.MAX_W / W), 1.0)
        SH, SW = int(s * H), int(s * W)
        print(f"Input image size: {SH}x{SW} ...")

        input_tensor = F.interpolate(input_tensor, size=(SH, SW), mode="bilinear", align_corners=False)

        trimap = input_tensor[:, 3:4, :, :]
        # Unkown area exist ?
        fg_area = (trimap >= 0.9).to(torch.float32).sum().item()
        bg_area = (trimap <= 0.1).to(torch.float32).sum().item()
        B2, C2, H2, W2 = trimap.size()
        unkown_area_ratio = 1.0  - (fg_area + bg_area)/(B2 * H2 *W2)

        # print(f"{filename}: unkown_area_ratio = {unkown_area_ratio} ...")
        if unkown_area_ratio < 0.05:
            print("Running model netdis ...")
            # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
            # orig_tensor = input_tensor.clone().detach()
            with torch.no_grad():
                input_tensor = model1(input_tensor[:, 0:3, :, :].to(device))
        else:
            print("Running model vitmat ...")

            # need set trimap in (0.1, 0.9) erea as 0.5
            mask1 = (trimap < 0.9)
            mask2 = (trimap > 0.1)
            mask = torch.logical_and(mask1, mask2)
            trimap[mask] = 0.5
            input_tensor[:, 3:4, :, :] = trimap

        with torch.no_grad():
            predict_tensor = model2(input_tensor.to(device))

        output_file = f"{output_dir}/{os.path.basename(filename)}"

        input_tensor[:, 3:4, :, :] = 1.0

        output_file = output_file.replace(".jpg", ".png") # could save RGBA format to jpg
        todos.data.save_tensor([input_tensor, predict_tensor], output_file)
    todos.model.reset_device()
