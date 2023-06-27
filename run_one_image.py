"""
It is used to run the model on one image (and its corresponding trimap).

For default, run:
python run_one_image.py \
    --model vitmatte-s \
    --checkpoint-dir path/to/checkpoint
It will be saved in the directory ``./demo``.

If you want to run on your own image, run:
python run_one_image.py \
    --model vitmatte-s(or vitmatte-b) \
    --checkpoint-dir <your checkpoint directory> \
    --image-dir <your image directory> \
    --trimap-dir <your trimap directory> \
    --output-dir <your output directory> \
    --device <your device>
"""
import os
from PIL import Image
import numpy as np

from os.path import join as opj
from torchvision.transforms import functional as TF
from detectron2.engine import default_argument_parser
from detectron2.config import LazyConfig, instantiate
from detectron2.checkpoint import DetectionCheckpointer

import torch.nn.functional as F

import torch
import pdb

def infer_one_image(model, input, save_dir=None):
    """
    Infer the alpha matte of one image.
    Input:
        model: the trained model
        image: the input image
        trimap: the input trimap
    """
    B, C, H, W = input['image'].size()
    if (H > 1440 and W > 1440):
        input['image'] = F.interpolate(input['image'], scale_factor=0.5, mode='bilinear', align_corners=False)        
        input['trimap'] = F.interpolate(input['trimap'], scale_factor=0.5, mode='bilinear', align_corners=False)        

    # (Pdb) input['image']
    # tensor([[[[0.7686, 0.7569, 0.7333,  ..., 0.4078, 0.4235, 0.4549],
    #           [0.7216, 0.7137, 0.7059,  ..., 0.4118, 0.4275, 0.4588],
    #           [0.6745, 0.6745, 0.6745,  ..., 0.4196, 0.4314, 0.4627],
    #           ...,
    #           [0.9882, 0.9882, 0.9882,  ..., 0.6745, 0.7059, 0.7412],
    #           [0.9922, 0.9922, 0.9922,  ..., 0.6667, 0.7059, 0.7412],
    #           [0.9882, 0.9882, 0.9922,  ..., 0.6745, 0.7098, 0.7451]],

    #          [[0.7765, 0.7647, 0.7412,  ..., 0.4157, 0.4314, 0.4627],
    #           [0.7255, 0.7216, 0.7098,  ..., 0.4196, 0.4353, 0.4667],
    #           [0.6706, 0.6784, 0.6745,  ..., 0.4353, 0.4471, 0.4784],
    #           ...,
    #           [0.9961, 0.9961, 0.9922,  ..., 0.6510, 0.6863, 0.7216],
    #           [0.9961, 0.9961, 0.9961,  ..., 0.6353, 0.6784, 0.7176],
    #           [0.9922, 0.9922, 0.9961,  ..., 0.6510, 0.6902, 0.7294]],

    #          [[0.7255, 0.7137, 0.6863,  ..., 0.3961, 0.4118, 0.4431],
    #           [0.6706, 0.6706, 0.6549,  ..., 0.4000, 0.4157, 0.4471],
    #           [0.6118, 0.6196, 0.6157,  ..., 0.4118, 0.4275, 0.4588],
    #           ...,
    #           [0.9725, 0.9765, 0.9804,  ..., 0.5961, 0.6275, 0.6588],
    #           [0.9725, 0.9765, 0.9804,  ..., 0.6000, 0.6314, 0.6667],
    #           [0.9686, 0.9686, 0.9765,  ..., 0.6235, 0.6510, 0.6824]]]])

    # (Pdb) input['trimap']
    # tensor([[[[1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
    #           [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
    #           [1.0000, 1.0000, 1.0000,  ..., 1.0000, 1.0000, 1.0000],
    #           ...,
    #           [0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020],
    #           [0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020],
    #           [0.5020, 0.5020, 0.5020,  ..., 0.5020, 0.5020, 0.5020]]]])

    model.eval()
    with torch.no_grad():
        output = model(input)['phas'].flatten(0, 2)

    # output.size() -- [643, 960]
    # input['image'].size() -- [1, 3, 643, 960]

    # output = TF.to_pil_image(output)

    # (Pdb) output
    # tensor([[9.9959e-01, 9.9999e-01, 9.9998e-01,  ..., 9.9985e-01, 9.9983e-01,
    #          9.9953e-01],
    #         [9.9982e-01, 1.0000e+00, 1.0000e+00,  ..., 9.9990e-01, 9.9990e-01,
    #          9.9990e-01],
    #         [9.9986e-01, 1.0000e+00, 1.0000e+00,  ..., 9.9995e-01, 9.9994e-01,
    #          9.9992e-01],
    #         ...,
    #         [1.1288e-04, 9.7448e-05, 1.3852e-04,  ..., 1.0425e-07, 8.6882e-08,
    #          2.6031e-08],
    #         [8.7881e-05, 5.1824e-05, 5.2086e-05,  ..., 2.1662e-07, 1.3334e-07,
    #          7.8075e-08],
    #         [3.8731e-04, 2.5726e-04, 2.7778e-04,  ..., 1.5292e-05, 1.2437e-05,
    #          5.8554e-06]], device='cuda:0')


    B, C, H, W = input['image'].size()
    blend_mask = torch.zeros(B, 4, H, W)
    blend_mask[:, 0:3, :, :] = input['image'].cpu()
    blend_mask[0:1, 3:4, :, :] = output.cpu()
    blend_mask = blend_mask.squeeze(0).permute(1, 2, 0)
    blend_mask = (blend_mask.numpy() * 255).astype(np.uint8)
    output = Image.fromarray(blend_mask)

    output.save(opj(save_dir))

    return None

def init_model(model, checkpoint, device):
    """
    Initialize the model.
    Input:
        config: the config file of the model
        checkpoint: the checkpoint of the model
    """
    assert model in ['vitmatte-s', 'vitmatte-b']
    if model == 'vitmatte-s':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)
    elif model == 'vitmatte-b':
        config = 'configs/common/model.py'
        cfg = LazyConfig.load(config)
        cfg.model.backbone.embed_dim = 768
        cfg.model.backbone.num_heads = 12
        cfg.model.decoder.in_chans = 768
        model = instantiate(cfg.model)
        model.to('cuda')
        model.eval()
        DetectionCheckpointer(model).load(checkpoint)

    # model -- ViTMatte(...)

    return model

def get_data(image_dir, trimap_dir):
    """
    Get the data of one image.
    Input:
        image_dir: the directory of the image
        trimap_dir: the directory of the trimap
    """
    image = Image.open(image_dir).convert('RGB')
    image = TF.to_tensor(image).unsqueeze(0) # [1, 3, 1440, 1920]
    trimap = Image.open(trimap_dir).convert('L')
    trimap = TF.to_tensor(trimap).unsqueeze(0) # [1, 1, 1440, 1920]

    # import todos
    # output_tensor = torch.cat((image, trimap), dim = 1)
    # output_file = f"output/{os.path.basename(image_dir)}"
    # todos.data.save_tensor([output_tensor], output_file)
    return {
        'image': image,
        'trimap': trimap
    }

if __name__ == '__main__':
    #add argument we need:
    parser = default_argument_parser()
    parser.add_argument('--model', type=str, default='vitmatte-s')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/ViTMatte_S_DIS.pth')
    parser.add_argument('--image-dir', type=str, default='demo/retriever_rgb.png')
    parser.add_argument('--trimap-dir', type=str, default='demo/retriever_trimap.png')
    parser.add_argument('--output-dir', type=str, default='demo/result.png')
    parser.add_argument('--device', type=str, default='cuda')

    args = parser.parse_args()

    input = get_data(args.image_dir, args.trimap_dir)
    print('Initializing model...Please wait...')
    model = init_model(args.model, args.checkpoint_dir, args.device)

    print('Model initialized. Start inferencing...')
    alpha = infer_one_image(model, input, args.output_dir)
    print('Inferencing finished.')