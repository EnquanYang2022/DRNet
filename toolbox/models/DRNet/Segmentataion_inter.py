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



class EDGModule(nn.Module):  #边缘模块的生成  此处channel=64
    def __init__(self, dims):
        super(EDGModule, self).__init__()
        self.relu = nn.ReLU(True)


        self.conv_upsample1 = BasicConv2d(dims[0], 64, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(dims[1], 64, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(dims[2], 64, 3, padding=1)

        self.upsample = nn.Upsample(scale_factor=2)
        self.upsample4 = nn.Upsample(scale_factor=4)
        self.conv_concat2 = BasicConv2d(2 * 64, 64, 3, padding=1)
        self.conv_concat3 = BasicConv2d(2 * 64, 64, 3, padding=1)

        self.conv5 = nn.Conv2d(64, 1, 1)
        # self.sig = nn.Sigmoid()

    def forward(self, x1, x2, x3):  #
        up_x1 = self.conv_upsample1(x1)
        x2 = self.upsample(x2)
        conv_x2 = self.conv_upsample2(x2)

        cat_x2 = self.conv_concat2(torch.cat((up_x1, conv_x2), 1))  #2*channel

        x3 = self.upsample4(x3)
        up_cat_x3 = self.conv_upsample3(x3)

        cat_x4 = self.conv_concat3(torch.cat((cat_x2, up_cat_x3), 1))  #4*channel
        x = self.conv5(cat_x4)   # channel:1  1/4
        # x = self.sig(x)
        return x