import torch
import torch.nn as nn
import pdb


class ViTMatte(nn.Module):
    def __init__(self,
                 *,
                 backbone,
                 criterion,
                 pixel_mean,
                 pixel_std,
                 input_format,
                 size_divisibility,
                 decoder,
                 ):
        super(ViTMatte, self).__init__()
        # input_format = 'RGB'
        # pixel_mean = [0.485, 0.456, 0.406]
        # pixel_std = [0.229, 0.22399999999999998, 0.225]
        # size_divisibility = 32

        self.backbone = backbone # ViT(...)
        self.criterion = criterion
        self.input_format = input_format
        self.size_divisibility = size_divisibility
        self.decoder = decoder # Detail_Capture(...)
        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        # batched_inputs.keys() -- dict_keys(['image', 'trimap'])
        images, targets, H, W = self.preprocess_inputs(batched_inputs)
        # images.size() -- [1, 4, 672, 992]
        # targets -- {'phas': None}

        features = self.backbone(images) # features.size() -- [1, 384, 42, 62]
        outputs = self.decoder(features, images)  
        # outputs.keys() -- dict_keys(['phas'])
        # outputs['phas'].size() -- [1, 1, 672, 992]

        if self.training:
            assert targets is not None
            trimap = images[:, 3:4]
            sample_map = torch.zeros_like(trimap)
            sample_map[trimap==0.5] = 1
            losses = self.criterion(sample_map ,outputs, targets)               
            return losses
        else:
            outputs['phas'] = outputs['phas'][:,:,:H,:W]
            return outputs

    def preprocess_inputs(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = batched_inputs["image"].to(self.device)
        trimap = batched_inputs['trimap'].to(self.device)
        # trimap = torch.ones_like(images)[:, 0:1, :, :] # xxxx8888
        # trimap[trimap == 1.0] = 0.8

        images = (images - self.pixel_mean) / self.pixel_std

        if 'fg' in batched_inputs.keys():
            trimap[trimap < 85] = 0
            trimap[trimap >= 170] = 1
            trimap[trimap >= 85] = 0.5

        images = torch.cat((images, trimap), dim=1)
        
        B, C, H, W = images.shape
        if images.shape[-1]%32!=0 or images.shape[-2]%32!=0:
            new_H = (32-images.shape[-2]%32) + H
            new_W = (32-images.shape[-1]%32) + W
            new_images = torch.zeros((images.shape[0], images.shape[1], new_H, new_W)).to(self.device)
            new_images[:,:,:H,:W] = images[:,:,:,:]
            del images
            images = new_images

        if "alpha" in batched_inputs:
            phas = batched_inputs["alpha"].to(self.device)
        else:
            phas = None

        return images, dict(phas=phas), H, W