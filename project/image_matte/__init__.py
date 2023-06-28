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

import todos
from . import isnetdis, vitmatte

import pdb


class ImageMatte(nn.Module):
    def __init__(self):
        super(ImageMatte, self).__init__()
        # from VitMatte model limition
        # self.MAX_H = 1024
        # self.MAX_W = 1024
        # self.MAX_TIMES = 32

        # self.isnet_dis = todos.model.ResizePadModel(isnetdis.ISNetDIS())
        self.isnet_dis = isnetdis.ISNetDIS()
        self.vit_matte = todos.model.ResizePadModel(vitmatte.ViTMatte())

    def forward(self, x):
        B, C, H, W = x.size()

        images = x[:, 0:3, :, :]
        trimap = x[:, 3:4, :, :]

        # Unkown area exist ?
        fg_area = (trimap >= 0.9).to(torch.float32).sum().item()
        bg_area = (trimap <= 0.1).to(torch.float32).sum().item()
        unkown_area_ratio = 1.0  - (fg_area + bg_area)/(B * H *W)

        # print("area: ", 1.0 * B * H * W)
        # print("fg_area: ", fg_area)
        # print("bg_area: ", bg_area)
        # print("unkown_area_ratio: ", unkown_area_ratio)

        if unkown_area_ratio < 0.05:
            # unkown seems not exist, Create trimap by ISNetDIS
            x = self.isnet_dis(images)

        # Create matte by VitMatte
        return self.vit_matte(x)


def get_tvm_model():
    """
    TVM model base on torch.jit.trace
    """

    model = ImageMatte()
    # model = todos.model.ResizePadModel(model)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()
    print(f"Running tvm model model on {device} ...")

    return model, device


def get_matte_model():
    """Create model."""

    model = ImageMatte()
    # model = todos.model.ResizePadModel(model)
    device = todos.model.get_device()
    model = model.to(device)
    model.eval()

    print(f"Running model on {device} ...")
    # print(model)
    
    model = torch.jit.script(model)
    todos.data.mkdir("output")
    if not os.path.exists("output/image_matte.torch"):
        model.save("output/image_matte.torch")

    return model, device


def image_matte_predict(input_files, output_dir):
    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    model, device = get_matte_model()

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_rgba_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        # orig_tensor = input_tensor.clone().detach()
        predict_tensor = todos.model.forward(model, device, input_tensor)
        output_file = f"{output_dir}/{os.path.basename(filename)}"

        input_tensor[:, 3:4, :, :] = 1.0

        todos.data.save_tensor([input_tensor, predict_tensor], output_file)
    todos.model.reset_device()
