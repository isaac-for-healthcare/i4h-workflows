import torch
from PIL import Image
import numpy as np
from timeit import default_timer as timer
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch_tensorrt
import cv2
import os
import torch
import torch.nn as nn
import functools


def define_generator(input_nc, output_nc, ngf, use_dropout=False):
    """Create a generator

    Parameters:
        input_nc (int) -- the number of channels in input images
        output_nc (int) -- the number of channels in output images
        ngf (int) -- the number of filters in the last conv layer

    Returns a generator
    """
    norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)

    return net

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            # add one more conv layer
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)


def initialize_generator(use_trt=True):
    if use_trt:
        weights = "/localhome/local-vennw/code/gan_training_data/checkpoints/tcia_rt_to_us.ts"
        generator = torch.jit.load(weights)
    else:
        weights = "/localhome/local-vennw/code/gan_training_data/checkpoints/500_net_G.pth"
        generator = define_generator(
            input_nc=1,
            output_nc=1,
            ngf=64,
        )

        generator.cuda()
        generator.eval()

        state_dict = torch.load(weights)
        generator.load_state_dict(state_dict)

    return generator

def __pad_to_square(img):
    ow, oh = img.size
    # do center pad to square
    if ow > oh:
        pad_h = (ow - oh) // 2
        img = F.pad(img, (0, pad_h, 0, pad_h), padding_mode='constant', fill=0)
    else:
        pad_w = (oh - ow) // 2
        img = F.pad(img, (pad_w, 0, pad_w, 0), padding_mode='constant', fill=0)

    return img

def initialize_transforms():
    img_transforms = transforms.Compose(
        [
            transforms.Lambda(lambda img: __pad_to_square(img)),
            transforms.Resize((256, 256), transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ],
    )

    return img_transforms

def generate_us(test_image, generator):
    with torch.no_grad():
        output = generator(test_image)

        # post processing
        output = output.squeeze(0)
        output = output.permute(1, 2, 0)
        output = (output + 1) / 2.0 * 255.0

        output = output.cpu().float().numpy().astype(np.uint8)

        # resize to 224
        output = cv2.resize(output, (224, 224))

        # to rgb
        output = np.stack([output, output, output], axis=-1)

    return output
