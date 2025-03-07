import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data

from pytorch3d.ops import knn_gather, knn_points
from models.utils import MLP_CONV, MLP_Res, fps
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class TransformerBlock(nn.Module):
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


class Linear_ResBlock(nn.Module):
    def __init__(self, input_size=1024, output_size=256):
        super(Linear_ResBlock, self).__init__()
        self.conv1 = nn.Linear(input_size, input_size)
        self.conv2 = nn.Linear(input_size, output_size)
        self.conv_res = nn.Linear(input_size, output_size)

        self.af = nn.ReLU(inplace=True)

    def forward(self, feature):
        return self.conv2(self.af(self.conv1(self.af(feature)))) + self.conv_res(feature)


class AutoEncoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=256):
        super(AutoEncoder, self).__init__()

        self.relu = nn.GELU()

        self.conv1 = nn.Conv1d(3, 16, 1)
        self.mlp_geo = nn.Conv1d(6, 32, [1, 1])
        self.mlp_feat = nn.Conv1d(32, 32, [1, 1])

        # # pointattn
        # self.conv1 = nn.Conv1d(3, 64, kernel_size=1)
        # self.conv2 = nn.Conv1d(64, 64, kernel_size=1)

        self.sa1 = TransformerBlock(64, 64)
        self.sa1_1 = TransformerBlock(128, 128)
        self.sa2 = TransformerBlock(128, 128)
        self.sa2_1 = TransformerBlock(256, 256)
        self.sa3 = TransformerBlock(256, 256)
        self.sa3_1 = TransformerBlock(512, dim_feat)

        self.ps = nn.ConvTranspose1d(dim_feat, 128, num_pc, bias=True)
        self.mlp_1 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=dim_feat + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.GELU(),
            nn.Conv1d(64, 3, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, N, 3), N = 2048
        """
        # point-wise feature extraction
        x_ = x.transpose(1, 2).contiguous()  # (B, 3, N)
        feature = self.relu(self.conv1(x_))  # (B, 16, N)

        _, idx_xyz, knn_xyz = knn_points(x, x, K=16+1, return_nn=True, return_sorted=False)
        repeat_xyz = x_.unsqueeze(3).repeat(1, 1, 1, 16+1)
        feat_geo = knn_xyz.permute(0, 3, 1, 2) - repeat_xyz
        feat_geo = torch.cat([repeat_xyz, feat_geo], dim=1)
        feat_geo = self.relu(self.mlp_geo(feat_geo))

        feat = feature.transpose(1, 2).contiguous()  # (B, N, 32)
        _, idx_feat, knn_feat = knn_points(feat, feat, K=16+1, return_nn=True, return_sorted=False)
        repeat_feat = feature.unsqueeze(3).repeat(1, 1, 1, 16+1)
        feat_feat = knn_feat.permute(0, 3, 1, 2) - repeat_feat
        feat_feat = torch.cat([repeat_feat, feat_feat], dim=1)  # (B, 64, 512, 17)
        feat_feat = self.relu(self.mlp_feat(feat_feat))  # (B, 64, 512, 17)

        x0 = torch.max(torch.cat([feat_geo, feat_feat], dim=1), dim=3)[0]  # (B, 64, 2048)

        # x0 = self.relu(self.conv1(x_))
        # x0 = self.conv2(x0)

        # Encoder
        idx_0 = furthest_point_sample(x, 512)
        x_g0 = gather_operation(x0, idx_0)     # B, 64, 512
        points = gather_operation(x_, idx_0)   # B,  3, 512
        x1 = self.sa1(x_g0, x0).contiguous()   # B, 64, 512
        x1 = torch.cat([x_g0, x1], dim=1)      # B, 128, 512
        x1 = self.sa1_1(x1, x1).contiguous()   # B, 128, 512

        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), 256)
        x_g1 = gather_operation(x1, idx_1)        # B, 128, 256
        points = gather_operation(points, idx_1)  # B,   3, 256
        x2 = self.sa2(x_g1, x1).contiguous()      # B, 128, 256
        x2 = torch.cat([x_g1, x2], dim=1)         # B, 256, 256
        x2 = self.sa2_1(x2, x2).contiguous()      # B, 256, 256

        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), 128)
        x_g2 = gather_operation(x2, idx_2)        # B, 256, 128
        x3 = self.sa3(x_g2, x2).contiguous()      # B, 256, 128
        x3 = torch.cat([x_g2, x3], dim=1)         # B, 512, 128
        x3 = self.sa3_1(x3,x3).contiguous()       # B, 512, 128

        feat = torch.max(x3, dim=2, keepdim=True)[0]  # (B, 512, 1)

        # TODO: Variational mu and sigma are learned from global feat.

        x1 = self.ps(feat)  # (b, 128, num_pc)
        x1 = self.mlp_1(torch.cat([x1, feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        completion = self.mlp_4(x3)  # (b, 3, num_pc)
        
        return feat, completion.transpose(1, 2).contiguous()
        

class Denoiser(nn.Module):
    def __init__(self, num_down=1024, k=16):
        super().__init__()

        self.num_down = num_down
        self.k = k

        self.convs1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
        self.conv_feat = MLP_CONV(in_channel=512, layer_dims=[256, 128])
        self.convs = MLP_CONV(in_channel=128+128, layer_dims=[256, 128])

        self.convs2 = nn.Sequential(
            nn.Conv2d(9, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 64, [1, 1]),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 128, [1, 1])
        )

        self.conv_query = nn.Conv2d(256, 256, [1, 1])
        self.conv_key = nn.Conv2d(256, 256, [1, 1])
    
    def forward(self, x, global_feat):
        """
        Args:
            x: (B, N, 3)
        """
        x_ = x.transpose(1, 2).contiguous()  # (B, 3, 2048)
        f = self.convs1(x_)  # (B, 128, 2048)
        g_feat = self.conv_feat(global_feat)  # (B, 128, 1)
        f = self.convs(torch.cat([f, g_feat.repeat(1, 1, f.shape[2])], dim=1))  # (B, 128, 2048)
        
        down_x = fps(x, self.num_down)  # (B, 512, 3)

        _, idx_knn, knn_x = knn_points(down_x, x, K=self.k+1, return_nn=True, return_sorted=False)  # (B, 512, 16, 3)
        knn_f = knn_gather(f.permute(0, 2, 1), idx_knn)  # (B, 512, k+1, 128)
        
        repeat_x = down_x.unsqueeze(2).expand_as(knn_x)  # (B, 512, k+1, 3)
        dec = repeat_x - knn_x
        # distance = torch.sum(dec ** 2.0, dim=-1, keepdim=True) ** 0.5
        r = torch.cat([repeat_x, knn_x, dec], dim=-1)
        r = self.convs2(r.permute(0, 3, 1, 2))

        feat = torch.cat([knn_f.permute(0, 3, 1, 2), r], dim=1)  # (B, 256, 512, k+1)
        
        # attention weight
        q = self.conv_query(feat[:, :, :, :1])  # (B, 256, N, 1)
        q = q.repeat(1, 1, 1, 16)  # (B, 256, N, k)
        k = self.conv_key(feat[:, :, :, 1:])    # (B, 256, N, k)

        weight = torch.softmax(torch.sum(q * k, dim=1), dim=-1)  # (B, N, 16)
        knn_x = knn_x[:, :, 1:, :]  # (B, N, 16, 3)
        new_x = torch.sum(weight.unsqueeze(3).expand_as(knn_x) * knn_x, dim=2)

        return down_x, new_x


class UpSample(nn.Module):
    def __init__(self, channel=128, ratio=1):
        super(UpSample, self).__init__()
        self.ratio = ratio
        self.conv_1 = nn.Conv1d(256, channel, kernel_size=1)
        self.conv_11 = nn.Conv1d(512, 256, kernel_size=1)
        # self.conv_x = nn.Conv1d(3, 64, kernel_size=1)

        self.sa1 = TransformerBlock(channel * 2, 512)
        self.sa2 = TransformerBlock(512, 512)
        self.sa3 = TransformerBlock(512, 512)

        self.relu = nn.GELU()

        self.conv_out = nn.Conv1d(64, 3, kernel_size=1)

        self.channel = channel

        self.conv_delta = nn.Conv1d(channel * 2, channel * 1, kernel_size=1)
        # self.conv_delta = nn.Conv1d(512 + 128, channel * 1, kernel_size=1)
        # self.conv_ps = nn.Conv1d(channel * ratio, channel * ratio, kernel_size=1)
        self.ps = nn.ConvTranspose1d(512, 128, ratio, ratio, bias=False)
        self.up = nn.Upsample(scale_factor=ratio)

        # self.conv_x1 = nn.Conv1d(64, channel, kernel_size=1)

        self.conv_out1 = nn.Conv1d(channel, 64, kernel_size=1)

        # # PU-Transformer PUHead
        self.mlp = nn.Conv1d(3, 32, 1)
        self.k = 16 + 1
        self.mlp_geo = nn.Conv2d(6, 64, [1, 1])
        self.mlp_feat = nn.Conv2d(64, 64, [1, 1])

    def forward(self, x, coarse, feat_g):
        """
        Args:
            x: unused
            coarse: (B, 3, N), N=512 or 
            feat_g: (B, 512, 1)
        """
        batch_size, _, N = coarse.size()

        # PU-Transformer
        feature = self.relu(self.mlp(coarse))  # (B, 32, N)
        
        xyz = coarse.transpose(1, 2).contiguous()  # (B, N, 3)
        _, idx_xyz, knn_xyz = knn_points(xyz, xyz, K=self.k, return_nn=True, return_sorted=False)
        repeat_xyz = coarse.unsqueeze(3).repeat(1, 1, 1, self.k)
        feat_geo = knn_xyz.permute(0, 3, 1, 2) - repeat_xyz  # (B, 3, N, k)
        feat_geo = torch.cat([repeat_xyz, feat_geo], dim=1)  # (B, 6, N, k)
        feat_geo = self.relu(self.mlp_geo(feat_geo))

        feat = feature.transpose(1, 2).contiguous() # (B, N, 32)
        _, idx_feat, knn_feat = knn_points(feat, feat, K=self.k, return_nn=True, return_sorted=False)
        repeat_feat = feature.unsqueeze(3).repeat(1, 1, 1, self.k)
        feat_feat = knn_feat.permute(0, 3, 1, 2) - repeat_feat
        feat_feat = torch.cat([repeat_feat, feat_feat], dim=1)  # (B, 64, N, k)
        feat_feat = self.relu(self.mlp_feat(feat_feat))

        y = torch.max(torch.cat([feat_geo, feat_feat], dim=1), dim=3)[0]  # (B, 128, N)

        # y = self.conv_x1(self.relu(self.conv_x(coarse)))            # B, 128, N
        feat_g = self.conv_1(self.relu(self.conv_11(feat_g)))         # B, 128, 1
        y0 = torch.cat([y, feat_g.repeat(1, 1, y.shape[-1])], dim=1)  # B, 256, N

        y1 = self.sa1(y0, y0)  # B, 512, N
        y2 = self.sa2(y1, y1)  # B, 512, N
        y3 = self.sa3(y2, y2)  # B, 512, N
        y3 = self.ps(y3)  # B, 128, N x r
        # y3_ = self.ps(y3)  # B, 128, N x r

        y_up = self.up(y)  # (B, 128, N x ratio)
        # y_up = self.up(y3)  # (B, 128, N x ratio)
        y_cat = torch.cat([y3, y_up], dim=1)  # (B, 256, N x ratio)
        y4 = self.conv_delta(y_cat)           # (B, 128, N x ratio)

        x = self.conv_out(self.relu(self.conv_out1(y4))) + self.up(coarse)

        return x, y3  # (B, 3, N x ratio), (B, 128, N x ratio)


class Model(nn.Module):
    def __init__(self, num_pc=256, num_down=256, k=16, ratios=[4, 8]):
        super().__init__()

        self.ae = AutoEncoder(num_pc=num_pc)
        self.denoiser = Denoiser(num_down=num_down, k=k)
        # self.upsample0 = UpSample(ratio=1)
        self.upsample1 = UpSample(ratio=ratios[0])
        self.upsample2 = UpSample(ratio=ratios[1])

    def forward(self, x):
        """
        Args:
            x: (B, 2048, 3), input partial point cloud
        """
        feat_g, coarse = self.ae(x)    # (B, 512, 1), (B, num_pc, 3)

        fusion = torch.cat([coarse, x], dim=1)
        down_fusion, coarse_ = self.denoiser(fusion, feat_g)

        x_ = fps(torch.cat([coarse_, x], dim=1), 512)

        # x_ = fps(fusion, 512)
        # refine0, feat_fine0 = self.upsample0(None, x_.transpose(1, 2).contiguous(), feat_g)

        refine1, feat_fine1 = self.upsample1(None, x_.transpose(1, 2).contiguous(), feat_g)         # (B, 3, 2048),  (B, 128, 2048)
        refine2, feat_fine2 = self.upsample2(None, refine1, feat_g)                                 # (B, 3, 16384), (B, 128, 16384)

        return coarse, fusion, down_fusion, coarse_, x_, refine1.transpose(1, 2).contiguous(), refine2.transpose(1, 2).contiguous()
        # return coarse, x_, refine0.transpose(1, 2).contiguous(), refine1.transpose(1, 2).contiguous(), refine2.transpose(1, 2).contiguous()

if __name__ == '__main__':
    model = Model().cuda()
    x = torch.randn(16, 2048, 3).cuda()
    coarse, fusion, down_fusion, denoise, new_in, refine1, refine2 = model(x)
    print(coarse.shape)
    print(fusion.shape)
    print(down_fusion.shape)
    print(denoise.shape)
    print(new_in.shape)
    print(refine1.shape)
    print(refine2.shape)
