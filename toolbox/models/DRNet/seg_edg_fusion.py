import torch.nn as nn
import torch

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class ER(nn.Module):
    def __init__(self, in_channel):
        super(ER, self).__init__()

        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 1, 1, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv2_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 4, 4, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))
        self.conv3_1 = nn.Sequential(nn.Conv2d(in_channel, in_channel, 3, 1, 8, 8, bias=False),
                                     nn.BatchNorm2d(in_channel), nn.LeakyReLU(0.1, inplace=True))

        self.b_1 = BasicConv2d(in_channel * 3, in_channel, kernel_size=3, padding=1)
        self.conv_res = BasicConv2d(in_channel,in_channel,kernel_size=1,padding=0)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        # x = x*self.sig(y).expand_as(x)
        # x = x * y.expand_as(x)
        buffer_1 = []
        buffer_1.append(self.conv1_1(x))
        buffer_1.append(self.conv2_1(x))
        buffer_1.append(self.conv3_1(x))
        buffer_1 = self.b_1(torch.cat(buffer_1, 1))
        out = self.relu(buffer_1+self.conv_res(x))

        return out