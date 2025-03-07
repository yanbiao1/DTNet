import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from pytorch3d.ops import knn_gather, knn_points
from models.utils import MLP_CONV, MLP_Res, fps, index_points
from models.pointattn import cross_transformer, PCT_refine
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class AutoEncoder(nn.Module):
    def __init__(self, dim_feat=512, num_pc=512):
        super(AutoEncoder, self).__init__()

        # self.first_conv = nn.Sequential(
        #     nn.Conv1d(3, 128, 1),
        #     nn.BatchNorm1d(128),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(128, 256, 1)
        # )

        # self.second_conv = nn.Sequential(
        #     nn.Conv1d(512, 512, 1),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(inplace=True),
        #     nn.Conv1d(512, dim_feat, 1)
        # )
        self.relu = nn.GELU()

        self.conv1 = nn.Conv1d(3, 16, 1)
        self.mlp_geo = nn.Conv1d(6, 32, [1, 1])
        self.mlp_feat = nn.Conv1d(32, 32, [1, 1])

        self.sa1 = cross_transformer(64, 64)
        self.sa1_1 = cross_transformer(128, 128)
        self.sa2 = cross_transformer(128, 128)
        self.sa2_1 = cross_transformer(256, 256)
        self.sa3 = cross_transformer(256, 256)
        self.sa3_1 = cross_transformer(512, 512)

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
        # x = self.first_conv(x)  # (B, 256, N)
        # feat1 = torch.max(x, dim=2, keepdim=True)[0]  # (B, 256, 1)
        # x = torch.cat([x, feat1.repeat(1, 1, x.size(2))], dim=1)  # (B, 512, N)
        # x = self.second_conv(x)  # (B, latent_dim, N)
        # feat = torch.max(x, dim=2, keepdim=True)[0]  # (B, 512, 1)
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

        # GDP
        idx_0 = furthest_point_sample(x, 512)
        x_g0 = gather_operation(x0, idx_0)    # B, 64, 512
        points = gather_operation(x_, idx_0)  # B,  3, 512
        x1 = self.sa1(x_g0, x0).contiguous()  # B, 64, 512
        x1 = torch.cat([x_g0, x1], dim=1)      # B, 128, 512
        # SFA
        x1 = self.sa1_1(x1, x1).contiguous()    # B, 128, 512

        # GDP
        idx_1 = furthest_point_sample(points.transpose(1, 2).contiguous(), 256)
        x_g1 = gather_operation(x1, idx_1)        # B, 128, 256
        points = gather_operation(points, idx_1)  # B,   3, 256
        x2 = self.sa2(x_g1, x1).contiguous()   # B, 128, 256
        x2 = torch.cat([x_g1, x2], dim=1)      # B, 256, 256
        # SFA
        x2 = self.sa2_1(x2, x2).contiguous()   # B, 256, 256

        # GDP
        idx_2 = furthest_point_sample(points.transpose(1, 2).contiguous(), 128)
        x_g2 = gather_operation(x2, idx_2)        # B, 256, 128
        # points = gather_points(points, idx_2)
        x3 = self.sa3(x_g2, x2).contiguous()   # B, 256, 128
        x3 = torch.cat([x_g2, x3], dim=1)      # B, 512, 128
        # SFA
        x3 = self.sa3_1(x3,x3).contiguous()    # B, 512, 128

        feat = torch.max(x3, dim=2, keepdim=True)[0]  # (B, 512, 1)

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
        # k = self.conv_key(feat)
        k = self.conv_key(feat[:, :, :, 1:])    # (B, 256, N, k)

        # weight = torch.softmax(torch.sum(q * k, dim=1), dim=-1)  # (B, N, k)

        # new_x = torch.sum(weight.unsqueeze(3).expand_as(knn_x) * knn_x, dim=2)

        weight = torch.softmax(torch.sum(q * k, dim=1), dim=-1)  # (B, N, 16)
        knn_x = knn_x[:, :, 1:, :]  # (B, N, 16, 3)
        new_x = torch.sum(weight.unsqueeze(3).expand_as(knn_x) * knn_x, dim=2)

        return down_x, new_x


# class Denoiser2(nn.Module):
#     def __init__(self, num_down=1024, k=16):
#         super().__init__()

#         self.num_down = num_down
#         self.k = k

#         self.convs1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
#         self.conv_feat = MLP_CONV(in_channel=512, layer_dims=[256, 128])
#         self.convs = MLP_CONV(in_channel=128+128, layer_dims=[256, 256])

#         self.conv_query = nn.Conv2d(256, 256, [1, 1])
#         self.conv_key = nn.Conv2d(256, 256, [1, 1])

#         self.pos_mlp = nn.Sequential(
#             nn.Conv2d(3, 64, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 256, 1)
#         )

#         self.attn_mlp = nn.Sequential(
#             nn.Conv2d(256, 1024, [1, 1]),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.Conv2d(1024, 256, [1, 1]),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, [1, 1])
#         )
    
#     def forward(self, x, global_feat):
#         """
#         Args:
#             x: (B, N, 3)
#         """
#         # feature extraction
#         x_ = x.transpose(1, 2).contiguous()  # (B, 3, 2048)
#         f = self.convs1(x_)  # (B, 128, 2048)
#         g_feat = self.conv_feat(global_feat)  # (B, 128, 1)
#         f = self.convs(torch.cat([f, g_feat.repeat(1, 1, f.shape[2])], dim=1))  # (B, 256, 2048)
        
#         # fps
#         down_x = fps(x, self.num_down)  # (B, 512, 3)

#         # knn for down_x from x
#         _, idx_knn, knn_x = knn_points(down_x, x, K=self.k+1, return_nn=True, return_sorted=False)  # (B, 512, k+1, 3)
#         knn_f = knn_gather(f.permute(0, 2, 1), idx_knn)  # (B, 512, k+1, 256)
#         feat = knn_f.permute(0, 3, 1, 2)  # (B, 256, 512, k+1)
        
#         dec = down_x.unsqueeze(2) - knn_x
#         pos_emb = self.pos_mlp(dec[:, :, 1:, :].permute(0, 3, 1, 2))  # (B, 256, 512, k)

#         # attention weight
#         q = self.conv_query(feat[:, :, :, :1])  # (B, 256, N, 1)
#         k = self.conv_key(feat[:, :, :, 1:])     # (B, 256, N, k)

#         attention = torch.softmax(self.attn_mlp(q - k + pos_emb), dim=-1)  # (B, 1, N, k)

#         new_x = torch.sum(attention.permute(0, 2, 3, 1) * knn_x[:, :, 1:, :], dim=2)  # (B, 512, 3)

#         return down_x, new_x


# class Denoiser3(nn.Module):
#     def __init__(self, num_down=1024, k=16):
#         super().__init__()

#         self.num_down = num_down
#         self.k = k

#         self.convs1 = MLP_CONV(in_channel=3, layer_dims=[64, 128])
#         self.conv_feat = MLP_CONV(in_channel=512, layer_dims=[256, 128])
#         self.convs = MLP_CONV(in_channel=128+128, layer_dims=[256, 128])

#         self.convs2 = nn.Sequential(
#             nn.Conv2d(9, 64, [1, 1]),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 64, [1, 1]),
#             nn.BatchNorm2d(64),
#             nn.LeakyReLU(inplace=True),
#             nn.Conv2d(64, 128, [1, 1])
#         )

#         self.conv_query = nn.Conv2d(256, 256, [1, 1])
#         self.conv_key = nn.Conv2d(256, 256, [1, 1])

#         self.pos_mlp = nn.Sequential(
#             nn.Conv2d(3, 64, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.Conv2d(64, 256, 1)
#         )

#         self.attn_mlp = nn.Sequential(
#             nn.Conv2d(256, 1024, [1, 1]),
#             nn.BatchNorm2d(1024),
#             nn.ReLU(),
#             nn.Conv2d(1024, 256, [1, 1]),
#             nn.BatchNorm2d(256),
#             nn.ReLU(),
#             nn.Conv2d(256, 1, [1, 1])
#         )
    
#     def forward(self, x, global_feat):
#         """
#         Args:
#             x: (B, N, 3)
#         """
#         # feature extraction
#         x_ = x.transpose(1, 2).contiguous()  # (B, 3, 2048)
#         f = self.convs1(x_)  # (B, 128, 2048)
#         g_feat = self.conv_feat(global_feat)  # (B, 128, 1)
#         f = self.convs(torch.cat([f, g_feat.repeat(1, 1, f.shape[2])], dim=1))  # (B, 256, 2048)
        
#         # fps
#         down_x = fps(x, self.num_down)  # (B, 512, 3)

#         # knn for down_x from x
#         _, idx_knn, knn_x = knn_points(down_x, x, K=self.k+1, return_nn=True, return_sorted=False)  # (B, 512, k+1, 3)
#         knn_f = knn_gather(f.permute(0, 2, 1), idx_knn)  # (B, 512, k+1, 256)
#         # feat = knn_f.permute(0, 3, 1, 2)  # (B, 256, 512, k+1)

#         repeat_x = down_x.unsqueeze(2).expand_as(knn_x)  # (B, 512, k+1, 3)
#         dec = repeat_x - knn_x
#         # distance = torch.sum(dec ** 2.0, dim=-1, keepdim=True) ** 0.5
#         r = torch.cat([repeat_x, knn_x, dec], dim=-1)
#         r = self.convs2(r.permute(0, 3, 1, 2))

#         feat = torch.cat([knn_f.permute(0, 3, 1, 2), r], dim=1)  # (B, 256, 512, k+1)
        
#         dec = down_x.unsqueeze(2) - knn_x
#         pos_emb = self.pos_mlp(dec[:, :, 1:, :].permute(0, 3, 1, 2))  # (B, 256, 512, k)

#         # attention weight
#         q = self.conv_query(feat[:, :, :, :1])  # (B, 256, N, 1)
#         k = self.conv_key(feat[:, :, :, 1:])    # (B, 256, N, k)

#         attention = torch.softmax(self.attn_mlp(q - k + pos_emb), dim=-1)  # (B, 1, N, k)

#         new_x = torch.sum(attention.permute(0, 2, 3, 1) * knn_x[:, :, 1:, :], dim=2)  # (B, 512, 3)

#         return down_x, new_x


class Model4(nn.Module):
    def __init__(self, num_pc=1024, num_down=1024, k=16):
        super(Model4, self).__init__()

        self.encoder = AutoEncoder(num_pc=num_pc)
        self.denoiser = Denoiser(num_down=num_down, k=k)

    def forward(self, x):
        """
        Args:
            x: (B, 2048, 3), input partial point cloud
        """
        # x_ = x.transpose(1, 2).contiguous()  # (B, 3, 1024)
        feat_g, coarse = self.encoder(x)    # (B, 512, 1), (B, 1024, 3)

        # fusion = torch.cat([coarse, fps(x, 1024)], dim=1)  # (B, 2048, 3)
        fusion = torch.cat([coarse, x], dim=1)  # (B, 2048, 3)
        down_fusion, coarse_ = self.denoiser(fusion, feat_g)

        return coarse, feat_g, fusion, down_fusion, coarse_


class Upsample(nn.Module):
    def __init__(self, ratios=[4, 8]):
        super().__init__()

        self.refine1 = PCT_refine(ratio=ratios[0])
        self.refine2 = PCT_refine(ratio=ratios[1])

    def forward(self, feat_g, x):
        fine, feat_fine = self.refine1(None, x, feat_g)         # (B, 3, 2048), (B, 128, 2048)
        fine1, feat_fine1 = self.refine2(None, fine, feat_g)    # (B, 3, 16384), (B, 128, 16384)

        return fine.transpose(1, 2).contiguous(), fine1.transpose(1, 2).contiguous()


class CompletionModel(nn.Module):
    def __init__(self, num_pc, num_down, ratios=[4, 8]):
        super().__init__()

        self.coarse_denoiser = Model4(num_pc=num_pc, num_down=num_down, k=16)
        self.model = Upsample(ratios=ratios)
    
    def forward(self, x):
        coarse, feat_g, fusion, down_fusion, coarse_ = self.coarse_denoiser(x)
        in_cloud__ = fps(torch.cat([x, coarse_], dim=1), 512)
        refine1, refine2 = self.model(feat_g, in_cloud__.transpose(1, 2).contiguous())
        return coarse, fusion, down_fusion, coarse_, in_cloud__, refine1, refine2


if __name__ == '__main__':
    model = CompletionModel(1024, 1024, [4, 4]).cuda()
    x = torch.randn(4, 2048, 3).cuda()
    coarse, fusion, down_fusion, coarse_, in_cloud__, refine1, refine2 = model(x)
    print(coarse.shape)
    print(fusion.shape)
    print(down_fusion.shape)
    print(coarse_.shape)
    print(in_cloud__.shape)
    print(refine1.shape)
    print(refine2.shape)
