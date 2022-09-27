import os
from collections import OrderedDict

import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn


class VGGCaffePreTrained(pl.LightningModule):
    cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256,
           'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']

    def __init__(self, weights_path: str = os.path.dirname(os.path.abspath(__file__))+'/../models/vgg19.npy',
                 output_index: int = 26) -> None:
        super().__init__()
        try:
            data_dict: dict = np.load(weights_path, encoding='latin1', allow_pickle=True).item()
            self.features = self.make_layers(self.cfg, data_dict)
            del data_dict
        except FileNotFoundError as e:
            print("weights_path:", weights_path,
                  'does not exits!, if you want to training must download pretrained weights')
        self.output_index = output_index
        self.vgg_normalize = None

    def _process(self, x):
        # NOTE 图像范围为[-1~1]，先denormalize到0-1再归一化
        rgb = (x * 0.5 + 0.5) * 255  # to 255
        bgr = rgb[:, [2, 1, 0], :, :]  # rgb to bgr
        return self.vgg_normalize(bgr)  # vgg norm

    def setup(self, device: torch.device):
        mean: torch.Tensor = torch.tensor([103.939, 116.779, 123.68], device=device)
        mean = mean[None, :, None, None]
        self.vgg_normalize = lambda x: x - mean
        self.freeze()

    def _forward_impl(self, x):
        x = self._process(x)
        # NOTE get output without relu activation
        x = self.features[:self.output_index](x)
        return x

    def forward(self, x):
        return self._forward_impl(x)

    def train(self, mode: bool):
        return super().train(False)

    def state_dict(self, destination, prefix, keep_vars):
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        return destination

    @staticmethod
    def get_conv_filter(data_dict, name):
        return data_dict[name][0]

    @staticmethod
    def get_bias(data_dict, name):
        return data_dict[name][1]

    @staticmethod
    def get_fc_weight(data_dict, name):
        return data_dict[name][0]

    def make_layers(self, cfg, data_dict, batch_norm=False) -> nn.Sequential:
        layers = []
        in_channels = 3
        block = 1
        number = 1
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                block += 1
                number = 1
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                with torch.no_grad():
                    """ set value """
                    weight = torch.FloatTensor(self.get_conv_filter(data_dict, f'conv{block}_{number}'))
                    weight = weight.permute((3, 2, 0, 1))
                    bias = torch.FloatTensor(self.get_bias(data_dict, f'conv{block}_{number}'))
                    conv2d.weight.set_(weight)
                    conv2d.bias.set_(bias)
                number += 1
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v

        return nn.Sequential(*layers)
