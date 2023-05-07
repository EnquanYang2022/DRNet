
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 23:05:23 2020

@author: zhang
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from toolbox.models.DRNet.uniformer import uniformer_small
from toolbox.models.DRNet.Segmentataion_inter import EDGModule
# from toolbox.models.DRNet.seg_edg_fusion import ER
from toolbox.models.DRNet.GCN import EAGCN
class BasicConv2d(nn.Module):
    def __init__(self,in_channel,out_channel,kernel_size,stride=1,padding=0,dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel,out_channel,kernel_size=kernel_size,stride=stride,padding=padding,dilation=dilation,bias=False)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1)

    def forward(self, x):
        residual = self.conv1x1(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = residual + out
        out = self.relu(out)
        return out

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.conv(x)
        return out

# def downsample():
#     return nn.MaxPool2d(kernel_size=2, stride=2)

class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x

# class GCN(nn.Module):
#     def __init__(self, channel,size):
#         super(GCN, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#
#         self.relu = nn.ReLU(inplace=True)
#         self.para = torch.nn.Parameter(torch.ones((1 ,channel ,size[0] , size[1]), dtype = torch.float32))
#         self.adj = torch.nn.Parameter(torch.ones((channel ,channel), dtype = torch.float32))
#
#     def forward(self, x):
#         device = torch.device('cuda')
#         # y = torch.nn.functional.relu(self.adj)
#         b, c, H, W = x.size()
#         fea_matrix = x.view(b ,c , H *W)
#         c_adj = self.avg_pool(x).view(b ,c)
#
#         m = torch.ones((b ,c ,H ,W), dtype = torch.float32,device=device)
#
#         for i in range(0 ,b):
#
#             t1 = c_adj[i].unsqueeze(0).cuda()
#             t2 = t1.t()
#             c_adj_s = torch.abs(torch.abs(torch.sigmoid(t1 -t2 ) -0.5 ) -0.5 ) *2  #0-1
#             c_adj_s = (c_adj_s.t() + c_adj_s ) /2  #为了产生对称矩阵
#             #mm 是矩阵相乘;mul是对应位置相乘  某些情况下torch.spmm与torch.mm()一样的
#             #初始 self.adj*c_adj_s = c_adj_s:因为self.adj初始是1.  mm(self.adj*c_adj_s,fea_matrix[i]):AH,最后相当于AHW
#             output0 = torch.mul(torch.mm(self.adj *c_adj_s ,fea_matrix[i]).view(1 ,c ,H ,W) ,self.para)
#
#             m[i] = output0
#
#         output = torch.nn.functional.relu(m)
#         # output = torch.nn.functional.relu(m)
#         return output

class EEblock(nn.Module):
    def __init__(self, channel):
        super(EEblock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.sconv13 = nn.Conv2d(channel ,channel, kernel_size=(1 ,3), padding=(0 ,1))
        self.sconv31 = nn.Conv2d(channel ,channel, kernel_size=(3 ,1), padding=(1 ,0))
        # self.GCN_layer = GCN(channel,size)


    def forward(self, y, x):
        # y = torch.nn.functional.relu(self.adj)
        b, c, H, W = x.size()

        x1 = self.sconv13(x)
        x2 = self.sconv31(x)

        y1 = self.sconv13(y)
        y2 = self.sconv31(y)

        map_y13 = torch.sigmoid(self.avg_pool(y1).view(b ,c ,1 ,1))
        map_y31 = torch.sigmoid(self.avg_pool(y2).view(b ,c ,1 ,1))

        k = x1 *map_y31 + x2 *map_y13 + x
        # k = self.GCN_layer(x+y)

        return k

class DEDCGCNEE(nn.Module):
    def __init__(self,n_classes,dims=[64, 128, 320, 512]):
        super(DEDCGCNEE, self).__init__()
        self.uniformer_rgb = uniformer_small(in_chans=3,pretrained=True)
        self.uniformer_depth = uniformer_small(in_chans=3,pretrained=True)

        self.upsample = nn.Upsample(scale_factor=4)
        self.n_classes = n_classes



        # self.EEblock1 = EEblock(channel=dims[0])
        # self.EEblock2 = EEblock(channel=dims[1])
        # self.EEblock3 = EEblock(channel=dims[2])
        # self.EEblock4 = EEblock(channel=dims[3])

        self.Up4 = up_conv(dims[3],dims[2])
        self.Up_conv4 = Decoder(2*dims[2] ,dims[2])

        self.Up3 = up_conv(dims[2] ,dims[1])
        self.Up_conv3 = Decoder(2*dims[1] ,dims[1])

        self.Up2 = up_conv(self.n_classes, self.n_classes)

        # self.Up2 = up_conv(self.n_classes ,self.n_classes)
        self.Up1 = up_conv(self.n_classes ,self.n_classes)
        self.Up0 = up_conv(self.n_classes, self.n_classes)
        # self.Up_conv2 = Decoder(2*dims[0] ,dims[0])

        self.edge = EDGModule(dims)  #1 1/4
        self.gcnreason = EAGCN(self.n_classes,1,(60,60))



        self.fconv = nn.Conv2d(dims[1] ,self.n_classes, kernel_size=1, padding=0)


    def forward(self, rgb, depth):
        # depth = depth[:,:1,:,:]

        rgb_s1, rgb_s2, rgb_s3, rgb_s4 = self.uniformer_rgb(rgb)


        depth_s1, depth_s2, depth_s3, depth_s4 = self.uniformer_depth(depth)

        # m1 = self.EEblock1(depth_s1, rgb_s1)
        # m2 = self.EEblock2(depth_s2, rgb_s2)
        # m3 = self.EEblock3(depth_s3, rgb_s3)
        # m4 = self.EEblock4(depth_s4, rgb_s4)

        m1 = rgb_s1+depth_s1
        m2 = rgb_s2+depth_s2
        m3 = rgb_s3+depth_s3
        m4 = rgb_s4+depth_s4

        edge_feat = self.edge(m1, m2, m3)  # 边缘模块,即Bs的生成,用前三层的输入 1* 1/4* 1/4  120*160

        edge_r = F.interpolate(edge_feat, size=(60, 60), mode='bilinear', align_corners=True)  # 要变为1/8
        #




        # edg = self.edg(m1, m2, m3)  # 1/16 dims[2]
        # _,_,h,w = rgb_s4.shape
        # edg_feat = F.interpolate(edg,size=(h,w),mode='bilinear',align_corners=True)


        d4 = self.Up4(m4)

        l4 = torch.cat((m3, d4), dim=1)
        d4 = self.Up_conv4(l4)

        d3 = self.Up3(d4)

        l3 = torch.cat((m2, d3), dim=1)
        d3 = self.Up_conv3(l3)

        seg = self.fconv(d3)

        out = self.gcnreason(seg, edge_r)
        # out=seg * edge_r + seg
        # out=seg
        # d2 = self.Up2(d3)
        # #
        # l2 = torch.cat((m1, d2), dim=1)
        # d2 = self.Up_conv2(l2)
        # seg = self.fconv(d3)


        # out = seg*edge_r+seg

        out = self.Up2(out)
        out = self.Up1(out)
        out = self.Up0(out)
        edge = self.upsample(edge_feat)
        #d2 1/4 dims[0]
        # d2 = self.upsample(d2)

        # edg = F.interpolate(edg,scale_factor=16,mode='bilinear',align_corners=True)
        # d4_o = self.d4_o(d4)
        # d3_o = self.d3_o(d3)
        return out,edge
if __name__ == '__main__':
    x = torch.randn(1, 3, 480, 640)
    y = torch.randn(1, 3, 480, 640)
    net = DEDCGCNEE(n_classes=41)
    # print(list(net.parameters())[0])
    # print(net.named_parameters())
    net1 = net(x, y)

    # from torchsummary import summary
    # model = BBSNet(n_class=41)
    # model = model.cuda()
    # summary(model, input_size=[(3, 480, 640),(3,480,640)],batch_size=6)
    from toolbox.models.BBSnetmodel.FLOP import CalParams

    CalParams(net, x, y)
    print(sum(p.numel() for p in net.parameters()) / 1000000.0)