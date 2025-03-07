import sys

from models.utils import fps
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import math

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation as gather_points


class cross_transformer(nn.Module):

    def __init__(self, d_model=256, d_model_out=256, nhead=4, dim_feedforward=1024, dropout=0.0):
        super().__init__()
        self.multihead_attn1 = nn.MultiheadAttention(d_model_out, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear11 = nn.Linear(d_model_out, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear12 = nn.Linear(dim_feedforward, d_model_out)

        self.norm12 = nn.LayerNorm(d_model_out)
        self.norm13 = nn.LayerNorm(d_model_out)

        self.dropout12 = nn.Dropout(dropout)
        self.dropout13 = nn.Dropout(dropout)

        self.activation1 = torch.nn.GELU()

        self.input_proj = nn.Conv1d(d_model, d_model_out, kernel_size=1)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    # 原始的transformer
    def forward(self, src1, src2, if_act=False):
        """
        Args:
            src1: (B, C, N1)
            src2: (B, C, N2), N2 > N1
        """
        src1 = self.input_proj(src1)  # (B, C', N1)
        src2 = self.input_proj(src2)  # (B, C', N2)

        b, c, _ = src1.shape

        src1 = src1.reshape(b, c, -1).permute(2, 0, 1)  # (N1, B, C')
        src2 = src2.reshape(b, c, -1).permute(2, 0, 1)  # (N2, B, C')

        src1 = self.norm13(src1)
        src2 = self.norm13(src2)

        src12 = self.multihead_attn1(query=src1,
                                     key=src2,
                                     value=src2)[0]  # (N1, B, C')


        src1 = src1 + self.dropout12(src12)  # (N1, B, C')
        src1 = self.norm12(src1)             # (N1, B, C')

        src12 = self.linear12(self.dropout1(self.activation1(self.linear11(src1))))  # (N1, B, C')
        src1 = src1 + self.dropout13(src12)  # (N1, B, C')


        src1 = src1.permute(1, 2, 0)  # (B, C', N1)

        return src1


class PCT_refine(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(PCT_refine, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = cross_transformer(channel * 2, 512)
        self.sa2 = cross_transformer(512, 512)
        self.sa3 = cross_transformer(512, channel * ratio)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)

        self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)


    def forward(self, x, coarse,feat_g):
        """
        Args:
            x: unused
            coarse: (B, 3, N), N=512 or 
            feat_g: (B, 512, 1)
        """
        batch_size, _, N = coarse.size()

        y = self.conv_x1(self.relu(self.conv_x(coarse)))              # B, 128, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))         # B, 128, 1
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)  # B, 256, N

        y1 = self.sa1(y0, y0)  # B, 512, N
        y2 = self.sa2(y1, y1)  # B, 512, N
        y3 = self.sa3(y2, y2)  # B, 128 x ratio, N
        y3 = self.conv_ps(y3).reshape(batch_size, -1, N * self.ratio)  # (B, 128, N x ratio)

        y_up = y.repeat(1, 1, self.ratio)     # (B, 128, N x ratio)
        y_cat = torch.cat([y3, y_up], dim=1)  # (B, 256, N x ratio)
        y4 = self.conv_delta(y_cat)           # (B, 128, N x ratio)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + coarse.repeat(1, 1, self.ratio)  # (B, 3, N x ratio)

        return x, y3  # (B, 3, N x ratio), (B, 128, N x ratio)


class PCT_encoder(nn.Module):
    def __init__(self, channel=64):
        super(PCT_encoder, self).__init__()
        self.channel = channel
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        self.conv2 = nn.Conv1d(64, channel, kernel_size=1)

        self.sa1 = cross_transformer(channel,channel)
        self.sa1_1 = cross_transformer(channel*2,channel*2)
        self.sa2 = cross_transformer((channel)*2,channel*2)
        self.sa2_1 = cross_transformer((channel)*4,channel*4)
        self.sa3 = cross_transformer((channel)*4,channel*4)
        self.sa3_1 = cross_transformer((channel)*8,channel*8)

        self.relu = nn.GELU()


        self.sa0_d = cross_transformer(channel*8,channel*4)
        self.sa1_d = cross_transformer(channel*4,channel*2)
        self.sa2_d = cross_transformer(channel*2,channel*8)

        self.sa0_c = cross_transformer(channel*8,channel*8)
        self.sa1_c = cross_transformer(channel*4, channel*4)
        self.sa2_c = cross_transformer(channel*2, channel*2)


        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)
        self.conv_out1 = nn.Conv1d(channel*4, 64, kernel_size=1)
        self.ps = nn.ConvTranspose1d(channel*8, channel, 128, bias=True)
        self.ps_refuse = nn.Conv1d(channel, channel*8, kernel_size=1)
        self.ps_adj = nn.Conv1d(channel*8, channel*8, kernel_size=1)


    def forward(self, points):
        """
        Args:
            points: (B, 3, N), N = 2048
        """
        batch_size, _, N = points.size()

        x = self.relu(self.conv1(points))  # B, 64, 2048
        x0 = self.conv2(x)                 # B, 64, 2048

        # GDP
        idx_0 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 4)
        x_g0 = gather_points(x0, idx_0)        # B, 64, 512
        points = gather_points(points, idx_0)  # B,  3, 512
        x1 = self.sa1(x_g0, x0).contiguous()   # B, 64, 512
        x1 = torch.cat([x_g0, x1], dim=1)      # B, 128, 512
        # SFA
        x1 = self.sa1_1(x1, x1).contiguous()    # ~ B, 128, 512

        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 8)
        x_g1 = gather_points(x1, idx_1)        # B, 128, 256
        points = gather_points(points, idx_1)  # B,   3, 256
        x2 = self.sa2(x_g1, x1).contiguous()   # B, 128, 256
        x2 = torch.cat([x_g1, x2], dim=1)      # B, 256, 256
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()   # ~ B, 256, 256

        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), N // 16)
        x_g2 = gather_points(x2, idx_2)        # B, 256, 128
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()   # B, 256, 128
        x3 = torch.cat([x_g2, x3], dim=1)      # B, 512, 128
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()    # ~ B, 512, 128
        
        # seed generator
        # maxpooling
        x_g = F.adaptive_max_pool1d(x3, 1).view(batch_size, -1).unsqueeze(-1)  # B, 512, 1
        x = self.relu(self.ps_adj(x_g))  # B, 512, 1
        x = self.relu(self.ps(x))        # B, 64,  128
        x = self.relu(self.ps_refuse(x)) # ~ B, 512, 128
        # SFA
        x0_c = self.sa0_c(x, x3)  # ~ B, 512, 128
        x0_d = self.sa0_d(x0_c, x0_c)  # B, 256, 128
        x1_c = self.sa1_c(x0_d, x2)  # ~ B, 256, 128
        x1_d = self.sa1_d(x1_c, x1_c)  # B, 256, 128
        x2_c = self.sa2_c(x1_d, x1)  # ~ B, 128, 128
        x2_d = self.sa2_d(x2_c, x2_c).reshape(batch_size, self.channel * 4, N // 8)  # B, 256, 256

        fine = self.conv_out(self.relu(self.conv_out1(x2_d)))  # B, 3, 256

        return x_g, fine  # (B, 512, 1), (B, 3, 256)


class Model(nn.Module):
    def __init__(self, dataset):
        super(Model, self).__init__()
        if dataset == 'pcn':
            step1 = 4
            step2 = 8
        elif dataset == 'c3d':
            step1 = 1
            step2 = 4
        else:
            ValueError('dataset is not exist')

        self.encoder = PCT_encoder()

        self.refine = PCT_refine(ratio=step1)
        self.refine1 = PCT_refine(ratio=step2)


    def forward(self, x):
        """
        Args:
            x: (B, 2048, 3), input partial point cloud
        """
        x = x.transpose(1, 2).contiguous()
        feat_g, coarse = self.encoder(x)  # (B, 512, 1), (B, 3, 256)

        new_x = torch.cat([x, coarse],dim=2)  # (B, 3, 2048+256)
        # new_x = gather_points(new_x, furthest_point_sample(new_x.transpose(1, 2).contiguous(), 512))  # (B, 3, 512)
        new_x = fps(new_x.transpose(1, 2).contiguous()).transpose(1, 2).contiguous()

        fine, feat_fine = self.refine(None, new_x, feat_g)         # (B, 3, 2048), (B, 128, 2048)
        fine1, feat_fine1 = self.refine1(feat_fine, fine, feat_g)  # (B, 3, 16384), (B, 128, 16384)

        coarse = coarse.transpose(1, 2).contiguous()  # (B, 256, 3)
        fine = fine.transpose(1, 2).contiguous()      # (B, 2048, 3)
        fine1 = fine1.transpose(1, 2).contiguous()    # (B, 16384, 3)

        return coarse, fine, fine1


if __name__ == '__main__':
    model = Model('pcn').cuda()
    x = torch.randn(8, 2048, 3).cuda()
    coarse, fine, fine1 = model(x)
    print(coarse.shape, fine.shape, fine1.shape)
