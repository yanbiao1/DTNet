import sys
sys.path.append('.')

from typing import Iterable

import torch
import torch.nn as nn

from torch import Tensor
from pytorch3d.ops import knn_points, knn_gather

from models.utils import MLP, PointNet_SA_Module_KNN, Transformer, FullyConnected, MLP_CONV, MLP_Res, fps, get_mesh, get_sample_points


class CoarseAutoEncoder(nn.Module):
    """
    Given a partial point cloud, this module will regress a coarse complete point cloud.
    """
    
    def __init__(self, feat_dim=512, num_coarse=256):
        super().__init__()

        self.sa_module_1 = PointNet_SA_Module_KNN(512, 16, 3, [64, 128], group_all=False, if_bn=False, if_idx=True)
        self.transformer_1 = Transformer(128, dim=64)
        self.sa_module_2 = PointNet_SA_Module_KNN(128, 16, 128, [128, 256], group_all=False, if_bn=False, if_idx=True)
        self.transformer_2 = Transformer(256, dim=64)
        self.sa_module_3 = PointNet_SA_Module_KNN(None, None, 256, [512, feat_dim], group_all=True, if_bn=False)
    
        self.ps = nn.ConvTranspose1d(feat_dim, 128, num_coarse, bias=True)
        self.mlp_1 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_2 = MLP_Res(in_dim=128, hidden_dim=64, out_dim=128)
        self.mlp_3 = MLP_Res(in_dim=feat_dim + 128, hidden_dim=128, out_dim=128)
        self.mlp_4 = nn.Sequential(
            nn.Conv1d(128, 64, 1),
            nn.ReLU(),
            nn.Conv1d(64, 3, 1)
        )

    def forward(self, x: Tensor):
        """
        Args:
            x: (B, N, 3), partial point cloud
        
        Returns:
            global_feat: (B, feat_dim, 1)
            coarse_pc: (B, num_coarse, 3)
        """
        x = x.transpose(1, 2).contiguous()

        l0_xyz = x
        l0_points = x

        l1_xyz, l1_points, idx1 = self.sa_module_1(l0_xyz, l0_points)  # (B, 3, 512), (B, 128, 512)
        l1_points = self.transformer_1(l1_points, l1_xyz)
        l2_xyz, l2_points, idx2 = self.sa_module_2(l1_xyz, l1_points)  # (B, 3, 128), (B, 256, 512)
        l2_points = self.transformer_2(l2_points, l2_xyz)
        l3_xyz, l3_points = self.sa_module_3(l2_xyz, l2_points)  # (B, 3, 1), (B, feat_dim, 1)

        global_feat = l3_points

        x1 = self.ps(global_feat)  # (b, 128, 256)
        x1 = self.mlp_1(torch.cat([x1, global_feat.repeat((1, 1, x1.size(2)))], 1))
        x2 = self.mlp_2(x1)
        x3 = self.mlp_3(torch.cat([x2, global_feat.repeat((1, 1, x2.size(2)))], 1))  # (b, 128, 256)
        coarse_pc = self.mlp_4(x3)  # (b, 3, 256)
        
        return global_feat, coarse_pc.transpose(1, 2).contiguous()


class FeatureExtraction(nn.Module):

    def __init__(self, idim: int, odim: int, k: int, growth_width: int, is_dynamic: bool):
        super(FeatureExtraction, self).__init__()
        assert (odim % growth_width == 0)

        self.k = k
        self.is_dynamic = is_dynamic
        self.num_conv = (odim // growth_width)

        idim = idim * 3
        self.convs = nn.ModuleList()

        conv_first = nn.Sequential(*[
            nn.Conv2d(idim, growth_width, kernel_size=[1, 1]),
            nn.BatchNorm2d(growth_width),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv_first)

        for i in range(self.num_conv - 1):
            in_channel = growth_width * (i + 1) + idim

            conv = nn.Sequential(*[
                nn.Conv2d(in_channel, growth_width, kernel_size=[1, 1], bias=True),
                nn.BatchNorm2d(growth_width),
                nn.LeakyReLU(0.05, inplace=True),
            ])
            self.convs.append(conv)

        self.conv_out = nn.Conv2d(growth_width * self.num_conv + idim, odim, kernel_size=[1, 1], bias=True)

    def derive_edge_feat(self, x: Tensor, knn_idx: Tensor or None):
        """
        x: [B, N, C]
        """
        if knn_idx is None and self.is_dynamic:
            _, knn_idx, _ = knn_points(x, x, K=self.k, return_nn=False, return_sorted=False)  # [B, N, K]
        knn_feat = knn_gather(x, knn_idx)                         # [B, N, K, C]
        x_tiled = torch.unsqueeze(x, dim=-2).expand_as(knn_feat)  # [B, N, K, C]
        
        return torch.cat([x_tiled, knn_feat, knn_feat - x_tiled], dim=-1)  # [B, N, K, C * 3]

    def forward(self, x: Tensor, knn_idx: Tensor or None=None, is_pooling=True):
        f = self.derive_edge_feat(x, knn_idx)  # [B, N, K, C]
        f = f.permute(0, 3, 1, 2)              # [B, C, N, K]

        for i in range(self.num_conv):
            _f = self.convs[i](f)
            f = torch.cat([f, _f], dim=1)

        f = self.conv_out(f)  # [B, C, N, K]

        if is_pooling:
            f, _ = torch.max(f, dim=-1, keepdim=False)
            return torch.transpose(f, 1, 2)   # [B, N, C]
        else:
            return f


class GPool(nn.Module):

    def __init__(self, n, dim, use_mlp=False, mlp_activation='relu'):
        super().__init__()
        self.use_mlp = use_mlp
        if use_mlp:
            self.pre = nn.Sequential(
                FullyConnected(dim, dim // 2, bias=True, activation=mlp_activation),
                FullyConnected(dim // 2, dim // 4, bias=True, activation=mlp_activation),
            )
            self.p = nn.Linear(dim // 4, 1, bias=True)
        else:
            self.p = nn.Linear(dim, 1, bias=True)
        self.n = n

    def forward(self, pos, x):
        """
        Args:
            pos: (B, N, 3)
            x: (B, N, C)
        """
        batchsize = x.size(0)
        if self.n < 1:
            k = int(x.size(1) * self.n)
        else:
            k = self.n

        if self.use_mlp:
            y = self.pre(x)
        else:
            y = x

        y = (self.p(y) / torch.norm(self.p.weight, p='fro')).squeeze(-1)  # B * N

        top_idx = torch.argsort(y, dim=1, descending=True)[:, 0:k]  # B * k
        y = torch.gather(y, dim=1, index=top_idx)  # B * k
        y = torch.sigmoid(y)

        pos = torch.gather(pos, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, 3))  # B * k * 3
        x = torch.gather(x, dim=1, index=top_idx.unsqueeze(-1).expand(batchsize, k, x.size(-1)))  # B * k * Fin
        x = x * y.unsqueeze(-1).expand_as(x)  # B * k * Fin, weighted feature

        return pos, x


class DownsampleModule(nn.Module):
    def __init__(self, feature_dim, ratio=0.5, use_mlp=False, activation='relu', pre_filter=True):
        super().__init__()
        self.pre_filter = pre_filter
        self.pool = GPool(ratio, dim=feature_dim, use_mlp=use_mlp, mlp_activation=activation)
        self.mlp = nn.Sequential(
            FullyConnected(feature_dim, feature_dim // 2, activation=activation),
            FullyConnected(feature_dim // 2, feature_dim // 4, activation=activation),
            FullyConnected(feature_dim // 4, 3, activation=None)
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        pos, x = self.pool(pos, x)
        if self.pre_filter:
            pos = pos + self.mlp(x)
        return pos, x


# class UpsampleModule(nn.Module):
#     def __init__(self, up_ratio):
#         super().__init__()

#         self.mlp_ps = MLP_CONV(in_channel=128, layer_dims=[64, 32])
#         self.ps = nn.ConvTranspose1d(32, 128, up_ratio, up_ratio, bias=False)   # point-wise splitting

#         self.up_sampler = nn.Upsample(scale_factor=up_ratio)
#         self.mlp_delta_feature = MLP_Res(in_dim=256, hidden_dim=128, out_dim=128)

#         self.mlp_delta = MLP_CONV(in_channel=128, layer_dims=[64, 3])
    
#     def forward(self, x, feat):
#         """
#         Args:
#             x: (B, N, 3)
#             feat: (B, N, C)
        
#         Returns:
#             new_x: (B, N x up_ratio, 3)
#         """
#         x = x.transpose(1, 2).contiguous()
#         feat = feat.transpose(1, 2).contiguous()

#         feat_ = self.mlp_ps(feat)
#         feat_ = self.ps(feat_)  # (B, 128, N * up_ratio)
        
#         feat_up = self.up_sampler(feat)
#         feat_cur = self.mlp_delta_feature(torch.cat([feat_, feat_up], 1))
        
#         delta = self.mlp_delta(torch.relu(feat_cur))
#         new_x = self.up_sampler(x) + delta
        
#         return new_x.transpose(1, 2).contiguous()


class UpsampleModule(nn.Module):

    def __init__(self, feature_dim, mesh_dim=1, mesh_steps=2, use_random_mesh=False, activation='relu'):
        super().__init__()
        self.mesh_dim = mesh_dim
        self.mesh_steps = mesh_steps
        self.mesh = nn.Parameter(get_mesh(dim=mesh_dim, steps=mesh_steps), requires_grad=False)    # Regular mesh
        self.use_random_mesh = use_random_mesh
        if use_random_mesh:
            self.ratio = mesh_steps
            print('[INFO] Using random mesh.')
        else:
            self.ratio = mesh_steps ** mesh_dim


        self.folding = nn.Sequential(
            FullyConnected(feature_dim+mesh_dim, 128, bias=True, activation=activation),
            FullyConnected(128, 128, bias=True, activation=activation),
            FullyConnected(128, 64, bias=True, activation=activation),
            FullyConnected(64, 3, bias=True, activation=None),
        )

    def forward(self, pos, x):
        """
        :param  pos:    (B, N, 3)
        :param  x:      (B, N, d)
        :return (B, rN, d)
        """
        batchsize, n_pts, _ = x.size()
        x_tiled = x.repeat(1, self.ratio, 1)

        if self.use_random_mesh:
            mesh = get_sample_points(dim=self.mesh_dim, samples=self.mesh_steps, num_points=n_pts, num_batch=batchsize).to(device=x.device)
            x_expanded = torch.cat([x_tiled, mesh], dim=-1)   # (B, rN, d+d_mesh)
        else:        
            mesh_tiled = self.mesh.unsqueeze(-1).repeat(1, 1, n_pts).transpose(1, 2).reshape(1, -1, self.mesh_dim).repeat(batchsize, 1, 1)
            x_expanded = torch.cat([x_tiled, mesh_tiled], dim=-1)   # (B, rN, d+d_mesh)

        residual = self.folding(x_expanded) # (B, rN, 3)

        upsampled = pos.repeat(1, self.ratio, 1) + residual
        return upsampled
        

class DenoisedUpsampling(nn.Module):
    def __init__(self, down_ratio=0.5, up_ratio=4, k=16):
        super().__init__()

        self.encoder = FeatureExtraction(3, 128, k, 32, is_dynamic=True)
        self.mlp1 = nn.Sequential(nn.Conv1d(512, 256, 1), nn.GELU(), nn.Conv1d(256, 128, 1))
        self.mlp2 = MLP(256, [256, 128])
        self.downsample = DownsampleModule(128, down_ratio)
        self.upsample = UpsampleModule(128, 1, up_ratio)
    
    def forward(self, x, global_feat):
        """
        Args:
            x: (B, N, 3), coarse complete point cloud
        """
        # feature extraction
        feat = self.encoder(x)  # (B, N, 128)

        # differentiable pooling
        global_feat = self.mlp1(global_feat)  # (B, 128, 1)
        feat = self.mlp2(torch.cat([feat, global_feat.transpose(1, 2).repeat(1, feat.shape[1], 1)], dim=2))
        new_x, new_feat = self.downsample(x, feat)

        # upsampling
        up_new_x = self.upsample(new_x, new_feat)

        return new_x, up_new_x


class Model(nn.Module):
    def __init__(self, feat_dim: int, num_coarse: int, down_ratios: Iterable, up_ratios: Iterable):
        super().__init__()

        self.coarse_ae = CoarseAutoEncoder(feat_dim, num_coarse)

        self.denoise_upsample1 = DenoisedUpsampling(down_ratios[0], up_ratios[0], 16)
        self.denoise_upsample2 = DenoisedUpsampling(down_ratios[1], up_ratios[1], 16)
    
    def forward(self, x):
        """
        Args:
            x: (B, N, 3), coarse complete point cloud
        """
        global_feat, coarse_pc = self.coarse_ae(x)

        fusion_pc = fps(torch.cat([x, coarse_pc], dim=1), 1024)

        # denoise and upsampling
        denoised1, upsampled1 = self.denoise_upsample1(fusion_pc, global_feat)
        denoised2, upsampled2 = self.denoise_upsample2(upsampled1, global_feat)

        return coarse_pc, fusion_pc, denoised1, upsampled1, denoised2, upsampled2


if __name__ == '__main__':
    model = Model(512, 512, [0.5, 0.5], [4, 16]).cuda()
    x = torch.randn(8, 2048, 3).cuda()
    coarse_pc, fusion_pc, denoised1, upsampled1, denoised2, upsampled2 = model(x)
    print('coarse:', coarse_pc.shape)
    print('fusion:', fusion_pc.shape)
    print('denoised1:', denoised1.shape)
    print('upsample1:', upsampled1.shape)
    print('denoised2:', denoised2.shape)
    print('upsample2:', upsampled2.shape)
