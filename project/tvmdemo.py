# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright Dell 2020-2022(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2020年 12月 28日 星期一 14:29:37 CST
# ***
# ************************************************************************************/
#

import pdb
import os
import time
from tqdm import tqdm

import torch
import todos
import image_matte

# Input shape: [1, 3, 1024, 1024]
# Test performance ...
#   Base:  {'mean': 35.491605879506096, 'median': 35.493094951380044, 'std': 0.021703701093924613}
#   TVM :  {'mean': 46.998761660652235, 'median': 48.91824670485221, 'std': 3.943442977615667}

def compile():
    model, device = image_matte.get_tvm_model()
    SO_B, SO_C, SO_H, SO_W = 1, 3, model.MAX_H, model.MAX_W

    todos.data.mkdir("output")
    if not os.path.exists("output/image_matte.so"):
        input = torch.randn(SO_B, SO_C, SO_H, SO_W)
        todos.tvmod.compile(model, device, input, "output/image_matte.so")
    todos.model.reset_device()


def predict(input_files, output_dir):
    model, device = image_matte.get_tvm_model()
    SO_B, SO_C, SO_H, SO_W = 1, 3, model.MAX_H, model.MAX_W

    # Create directory to store result
    todos.data.mkdir(output_dir)

    # load model
    tvm_model = todos.tvmod.load("output/image_matte.so", str(device))

    # load files
    image_filenames = todos.data.load_files(input_files)

    # start predict
    mean_time = 0
    progress_bar = tqdm(total=len(image_filenames))
    for filename in image_filenames:
        progress_bar.update(1)

        # orig input
        input_tensor = todos.data.load_tensor(filename)
        # pytorch recommand clone.detach instead of torch.Tensor(input_tensor)
        # orig_tensor = input_tensor.clone().detach()

        B, C, H, W = input_tensor.shape
        input_tensor = todos.data.resize_tensor(input_tensor, SO_H, SO_W)

        start_time = time.time()
        predict_tensor = todos.tvmod.forward(tvm_model, input_tensor)
        torch.cuda.synchronize()
        mean_time += time.time() - start_time

        output_file = f"{output_dir}/{os.path.basename(filename)}"
        predict_tensor = todos.data.resize_tensor(predict_tensor, H, W)
        todos.data.save_tensor([predict_tensor], output_file)

    mean_time = mean_time / len(image_filenames)

    print(f"Mean spend {mean_time:0.4f} seconds")
    os.system("nvidia-smi | grep python")

    todos.model.reset_device()


if __name__ == "__main__":
    compile()
    predict("images/*.png", "output/so")
