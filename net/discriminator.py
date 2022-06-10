import torch.nn as nn
from tools.customeLayers import ConvSN


class Discriminator(nn.Module):
    def __init__(self, in_ch, out_ch, n_dis, sn):
        super(Discriminator, self).__init__()
        channel = in_ch // 2

        self.conv1 = ConvSN(3, channel, kernel_size=3, stride=1, padding=1, sn=sn, use_bias=False)
        self.lrelu1 = nn.LeakyReLU(0.2, inplace=True)

        self.listLayer = []
        for i in range(n_dis):
            self.listLayer.append(ConvSN(channel, channel*2, kernel_size=3, stride=2, padding=1, sn=sn, use_bias=False))
            self.listLayer.append(nn.LeakyReLU(0.2, inplace=True))

            self.listLayer.append(ConvSN(channel*2, channel*4, kernel_size=3, stride=1, padding=1, sn=sn, use_bias=False))
            self.listLayer.append(nn.GroupNorm(num_groups=1, num_channels=channel*4, affine=True))
            self.listLayer.append(nn.LeakyReLU(0.2, inplace=True))

            channel = channel * 4

        self.conv2 = ConvSN(channel, channel*2, kernel_size=3, stride=1, padding=1, sn=sn, use_bias=False)
        self.layernorm = nn.GroupNorm(num_groups=1, num_channels=channel*2, affine=True)
        self.lrelu2 = nn.LeakyReLU(0.2, inplace=True)

        self.conv3 = ConvSN(channel*2, out_ch, kernel_size=3, stride=1, padding=1, sn=sn, use_bias=False)

    def forward(self, input):
        out = self.conv1(input)
        out = self.lrelu1(out)
        for layer in self.listLayer:
            out = layer(out)
        out = self.conv2(out)
        out = self.layernorm(out)
        out = self.lrelu2(out)
        out = self.conv3(out)
        return out