import math

import torch
import torch.nn as nn
import numpy as np


tree_arch = {}
tree_arch[2] = [32, 64]
tree_arch[4] = [4, 8, 8, 8]
tree_arch[6] = [2, 4, 4, 4, 4, 4]
tree_arch[8] = [2, 2, 2, 2, 2, 4, 4, 4]


def get_arch(nlevels, npts):
    logmult = int(math.log2(npts / 2048))
    assert 2048 * (2 ** (logmult)) == npts, "Number of points is %d, expected 2048x(2^n)" % (npts)
    arch = tree_arch[nlevels]
    while logmult > 0:
        last_min_pos = np.where(arch == np.min(arch))[0][-1]
        arch[last_min_pos] *= 2
        logmult -= 1
    return arch


# print(get_arch(8, 16384))


class TopNet(nn.Module):
    """
    "TopNet: Structural Point Cloud Decoder"
    (https://openaccess.thecvf.com/content_CVPR_2019/papers/Tchapmi_TopNet_Structural_Point_Cloud_Decoder_CVPR_2019_paper.pdf)
    
    Attributes:
        node_dim: dimension of every node feature except for leaf node.
        latent_dim: dimension of latent code.
        nlevels: number of level in decoder.
        num_pred: number of points in output point cloud.
    """
    def __init__(self, node_dim=8, latent_dim=1024, nlevels=6, num_pred=2048):
        super().__init__()
        self.node_dim = node_dim
        self.latent_dim = latent_dim
        self.nlevels = nlevels
        self.num_pred = num_pred

        self.tarch = get_arch(self.nlevels, self.num_pred)     # [2, 2, 4, 4, 4, 4, 4, 4]
        self.IN_CHANNEL = self.latent_dim + self.node_dim      # 1032
        self.OUT_CHANNEL = self.node_dim                       # 8

        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )

        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, self.latent_dim, 1)
        )

        self.root_layer = nn.Sequential(
            nn.Linear(self.latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(64, self.node_dim * int(self.tarch[0])),
            nn.Tanh()
        )
        self.leaf_layer = self.get_tree_layer(self.IN_CHANNEL, 3, int(self.tarch[-1]))
        self.feature_layers = nn.ModuleList([self.get_tree_layer(self.IN_CHANNEL, self.OUT_CHANNEL, int(self.tarch[d]) ) for d in range(1, self.nlevels - 1)])
    
    @staticmethod
    def get_tree_layer(in_channel, out_channel, node):
        return nn.Sequential(
            nn.Conv1d(in_channel, in_channel // 2, 1),
            nn.BatchNorm1d(in_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 2, in_channel // 4, 1),
            nn.BatchNorm1d(in_channel // 4),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 4, in_channel // 8, 1),
            nn.BatchNorm1d(in_channel // 8),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channel // 8, out_channel * node, 1),
        )

    def forward(self, xyz):
        B, N ,_ = xyz.shape
        
        # encoder
        feature = self.first_conv(xyz.transpose(2, 1))                           # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]              # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)                                      # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]             # (B, 1024)
        
        # decoder
        level10 = self.root_layer(feature_global).reshape(-1, self.node_dim, int(self.tarch[0])) # (B, 8, 2)
        outs = [level10,]
        for i in range(1, self.nlevels):
            last_level = outs[-1]
            expand_feature = feature_global.unsqueeze(2).expand(-1, -1, last_level.shape[2])         # (B, 1024, n_temp1)
            if i == self.nlevels - 1:
                layer_feature = self.leaf_layer(torch.cat([expand_feature, last_level], dim=1)).reshape(B, 3 ,-1)  # (B, 3, num_pred)
            else:
                layer_feature = self.feature_layers[i - 1](torch.cat([expand_feature, last_level], dim=1)).reshape(B, self.node_dim, -1)  # (B, 8, n_temp2)
            outs.append(nn.Tanh()(layer_feature))
        
        return outs[-1].transpose(1, 2).contiguous()


if __name__ == '__main__':
    x = torch.randn(32, 2048, 3)
    topnet = TopNet()
    y = topnet(x)
    print(y.shape)
