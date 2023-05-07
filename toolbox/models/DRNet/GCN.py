import torch
import torch.nn as nn
class GCN(nn.Module):
    def __init__(self, num_state, num_node):  # num_in:41 num_node: HW
        super(GCN, self).__init__()
        self.num_state = num_state
        self.num_node = num_node
        # self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1)

    def forward(self, seg, aj):  # seg:n,41,h,w   aj:n,hw,hw
        n, c, h, w = seg.size()
        seg = seg.view(n, self.num_state, -1).contiguous()  # n,41,HW
        seg_similar = torch.bmm(seg, aj)  # aj:N,HW,HW   得到 N,41,HW
        out = self.relu(self.conv2(seg_similar))
        output = out + seg  # N,41,HW

        return output


class EAGCN(nn.Module):
    def __init__(self, num_in, plane_mid, mids, normalize=False):  # num_in=41, plane_mid=1, mids=(60,80)
        super(EAGCN, self).__init__()
        self.num_in = num_in  # 41
        self.mids = mids
        self.normalize = normalize
        self.num_s = int(plane_mid)  # 1
        self.num_n = (mids[0]) * (mids[1])
        self.maxpool_c = nn.AdaptiveAvgPool2d(output_size=(1))
        self.conv_s1 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s11 = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_s2 = nn.Conv2d(1, 1, kernel_size=1)
        self.conv_s3 = nn.Conv2d(1, 1, kernel_size=1)
        self.mlp = nn.Linear(num_in, self.num_s)
        self.fc = nn.Conv2d(num_in, self.num_s, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.downsample = nn.AdaptiveAvgPool2d(output_size=(mids[0], mids[1]))

        self.gcn = GCN(num_state=num_in, num_node=self.num_n)  # N,41,H W  num_in 是输入通道个数  num_n 是总的像素个数,即图的节点个数
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1)
        self.blocker = nn.BatchNorm2d(num_in)

    def forward(self, seg_ori, edge_ori):
        seg = seg_ori
        edge = edge_ori
        n, c, h, w = seg.size()  # n,41,1/8,1/8

        # 通道建模
        seg_s = self.conv_s1(seg)  # num_in -> num_s  :41->1   n,1,1/8,1/8
        theta_T = seg_s.view(n, self.num_s, -1).contiguous()  # n,1,HW
        theta = seg_s.view(n, -1, self.num_s).contiguous()  # n,HW,1
        channel_att = torch.relu(self.mlp(self.maxpool_c(seg).squeeze(3).squeeze(2))).view(n, self.num_s, -1)  # n,1,1
        diag_channel_att = torch.bmm(channel_att, channel_att.view(n, -1, self.num_s))  # n,1,1   bmm:矩阵相乘  相当于缩放A的度矩阵D

        similarity_c = torch.bmm(theta, diag_channel_att)  # n,HW,1
        similarity_c = self.softmax(torch.bmm(similarity_c, theta_T))  # N, HW,HW

        # 空间建模
        seg_c = self.conv_s11(seg)  # 2->1
        sigma = seg_c.view(n, self.num_s, -1).contiguous()  # N,1,HW
        sigma_T = seg_c.view(n, -1, self.num_s).contiguous()
        sigma_out = torch.bmm(sigma_T, sigma)  # HW,HW

        # 图中2右半边的操作   如果不需要边界信息的输入, 下面这一部分不需要, similarity 直接换为 softmax(sigma_out)
        edge_m = seg * edge  # 这里边缘通道为1

        maxpool_s, _ = torch.max(seg, dim=1)  # b,H,W  取每个通道的最大值
        edge_m_pool, _ = torch.max(edge_m, dim=1)

        seg_ss = self.conv_s2(maxpool_s.unsqueeze(1)).view(n, 1, -1)  # n,1,h,w->n,1,hw
        edge_mm = self.conv_s3(edge_m_pool.unsqueeze(1)).view(n, -1, 1)  # n,1,h,w->n,hw,1

        diag_spatial_att = torch.bmm(edge_mm, seg_ss) * sigma_out  # hw*hw
        similarity_s = self.softmax(diag_spatial_att)
        # similarity_s = self.softmax(sigma_out)

        # 最后的相加
        similarity = similarity_c + similarity_s  # 相当于输入图卷积的邻接矩阵

        seg_gcn = self.gcn(seg, similarity).view(n, self.num_in, self.mids[0], self.mids[1])  # N,41,H,W

        ext_up_seg_gcn = seg_gcn + seg_ori
        return ext_up_seg_gcn
if __name__ == '__main__':
    x = torch.randn(2 ,3 ,60 ,80)
    y = torch.randn(2 ,3 ,60 ,80)  #边缘
    net = EAGCN(3,1,(60,80))  #带边界信息的输入
    out = net(x,y)
    print(out.shape)