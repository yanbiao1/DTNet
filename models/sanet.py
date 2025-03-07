import sys
sys.path.append('.')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pointnet2_ops.pointnet2_modules import PointnetSAModule


def MLP(channels):
    return nn.Sequential(*[
        nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
        for i in range(1, len(channels))
    ])


class Attention(torch.nn.Module):
    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(Attention, self).__init__()
        self.M_h = NN_h
        self.M_l = NN_l
        self.M_g = NN_g
        self.M_f = NN_f


class SelfAttention(Attention):
    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SelfAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p):
        """
        Args:
            p: (B, M, C)
        """
        h = self.M_h(p).transpose(-1, -2)            # (B, M, H) -> (B, H, M)
        l = self.M_l(p)                              # (B, M, H)
        g = self.M_g(p)                              # (B, M, C)
        mm = torch.matmul(l, h)                      # (B, M, M)
        attn_weights = F.softmax(mm, dim=-1)         # (B, M, M)
        atten_appllied = torch.bmm(attn_weights, g)  # (B, M, C)
        if self.M_f is not None:
            return self.M_f(p + atten_appllied)      # (B, M, C')
        else:
            return p + atten_appllied                # (B, M, C)


class SkipAttention(Attention):
    def __init__(self, NN_h, NN_l, NN_g, NN_f=None):
        super(SkipAttention, self).__init__(NN_h, NN_l, NN_g, NN_f)

    def forward(self, p, r):
        """
        Args:
            p: (B, M, 1, C)
            r: (B, 1, N, C')
        """
        h = self.M_h(p).expand(-1, -1, r.size(2), -1).unsqueeze(-2)  # (B, M, 1, H) -> (B, M, N, H) -> (B, M, N, 1, H)
        l = self.M_l(r).expand(-1, h.size(1), -1, -1).unsqueeze(-1)  # (B, 1, N, H) -> (B, M, N, H) -> (B, M, N, H, 1)
        g = self.M_g(r).squeeze()                                    # (B, 1, N, C) -> (B, N, C)
        mm = torch.matmul(h, l).squeeze()                            # (B, M, N)
        attn_weights = F.softmax(mm, dim=-1)                         # (B, M, N)
        atten_appllied = torch.bmm(attn_weights, g)                  # (B, M, C)
        if self.M_f is not None:
            return self.M_f(p.squeeze() + atten_appllied)            # (B, M, C) -> (B, 512 or 256)
        else:
            return p.squeeze() + atten_appllied                      # (B, M, C)


class FoldingBlock(torch.nn.Module):
    def __init__(self, input_shape, output_shape, attentions, NN_up, NN_down):
        super(FoldingBlock, self).__init__()

        self.in_shape = input_shape
        self.out_shape = output_shape

        self.self_attn1 = SelfAttention(*attentions)
        self.upmlp1 = MLP(NN_up)
        
        self.M_down = MLP(NN_down)

        self.self_attn2 = SelfAttention(*attentions)
        self.M_up2 = MLP(NN_up)

    def forward(self, p, m, n):
        p1 = self.up(p, m, n)        # (B, N_i, C')

        p2 = self.down(p1)           # (B, N_i+1, C)
        p_delta = p - p2
        
        p2 = self.up(p_delta, m, n)  # (B, N_i, C')
        
        return p1 + p2

    def up(self, p, m, n):
        """
        Args:
            p: (B, N_i+1, C) ... (B, 64, 512), (B, 256, 256), (B, 512, 128)
        """
        p = p.repeat(1, int(self.out_shape / self.in_shape), 1).contiguous()  # ??? should repeat directly (B, N_i, C)
        points = SANet.sample_2D(m, n)  # (mxn, 2) ... (256, 2), (512, 2), (2048, 2)
        p_2d = torch.cat((p, points.unsqueeze(0).expand(p.size(0), -1, -1)), -1)  # (B, N_i, C+2)
        p_2d = self.upmlp1(p_2d)                                                    # (B, N_i, C)
        p = torch.cat([p, p_2d], -1)                                              # (B, N_i, 2C)

        return self.self_attn1(p)                                                 # (B, N_i, C')

    def down(self, p):
        return self.M_down(p.view(-1, self.in_shape, int(self.out_shape / self.in_shape) * p.size(2)))  # (B, N_i+1, in_shape / out_shape * C')


class SANet(nn.Module):

    meshgrid = [[-0.3, 0.3, 46], [-0.3, 0.3, 46]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])

    points = torch.tensor(np.meshgrid(x, y), dtype=torch.float32).cuda()  # (2, 46, 46)

    def __init__(self):
        super().__init__()

        self.pointnet2_sa1 = PointnetSAModule(mlp=[3, 64, 64, 128],
                                              npoint=512,
                                              radius=0.2,
                                              nsample=32,
                                              bn=False,
                                              use_xyz=True)
        self.pointnet2_sa2 = PointnetSAModule(mlp=[128, 128, 128, 256],
                                              npoint=256,
                                              radius=0.4,
                                              nsample=32,
                                              bn=False,
                                              use_xyz=True)
        self.pointnet2_gsa = PointnetSAModule(mlp=[256, 256, 512, 512],
                                              npoint=None,
                                              radius=None,
                                              bn=False,
                                              use_xyz=True)

        # self.pointnet2_knn_sa1 = PointNet_SA_Module_KNN(512, 64, 3, [64, 64, 128], if_bn=False, use_xyz=True)
        # self.pointnet2_knn_sa2 = PointNet_SA_Module_KNN(256, 64, 128, [128, 128, 256], if_bn=False, use_xyz=True)
        # self.pointnet2_knn_sa3 = PointNet_SA_Module_KNN(None, None, 256, [256, 512, 512], if_bn=False, group_all=True, use_xyz=True)

        self.skip_attn1 = SkipAttention(MLP([512 + 2, 128]), MLP([256, 128]), MLP([256, 512 + 2]), MLP([512 + 2, 512]))
        self.skip_attn2 = SkipAttention(MLP([256, 64]), MLP([128, 64]), MLP([128, 256]), MLP([256, 256]))

        self.folding1 = FoldingBlock(64, 256, [MLP([512 + 512, 256]), MLP([512 + 512, 256]), MLP([512 + 512, 512 + 512]),
                                               MLP([512 + 512, 512, 256])], [512 + 2, 512], [1024, 512])

        self.folding2 = FoldingBlock(256, 512, [MLP([256 + 256, 64]), MLP([256 + 256, 64]), MLP([256 + 256, 256 + 256]),
                                                MLP([256 + 256, 256, 128])], [256 + 2, 256], [256, 256])
        self.folding3 = FoldingBlock(512, 2048, [MLP([128 + 128, 64]), MLP([128 + 128, 64]), MLP([128 + 128, 128 + 128]),
                                                 MLP([128 + 128, 128])], [128 + 2, 128], [512, 256, 128])

        self.lin = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 3))
            
    def forward(self, x):
        """
        Args:
            x: (B, N, 3), partial point clouds
        """
        feats = self.encode(x)
        out = self.decode(feats)
        return out

    def encode(self, x):
        """
        Args:
            x: (B, N, 3), partial point clouds
        
        Returns:
            (B, 512, 128)
            (B, 256, 256)
            (B, 1, 512)
        """
        feat = x.transpose(1, 2).contiguous()
        x, feat1 = self.pointnet2_sa1(x, feat)
        x, feat2 = self.pointnet2_sa2(x, feat1)
        x, feat3 = self.pointnet2_gsa(x, feat2)

        # x_ = x.transpose(1, 2).contiguous()
        # x_, feat1 = self.pointnet2_knn_sa1(x_, x_)
        # x_, feat2 = self.pointnet2_knn_sa2(x_, feat1)
        # x_, feat3 = self.pointnet2_knn_sa3(x_, feat2)

        return feat1.transpose(1, 2).contiguous(), feat2.transpose(1, 2).contiguous(), feat3.transpose(1, 2).contiguous()
    
    def decode(self, feats):
        p = SANet.sample_2D(8, 8)    # (64, 2)
        out = feats[2]               # (B, 1, 512)
        
        out = out.view(out.size(0), 1, 1, out.size(-1)).repeat(1, 64, 1, 1)                           # (B, 1, 1, 512) -> (B, 64, 1, 512)
        out = torch.cat((out, p.view(1, p.size(0), 1, p.size(-1)).repeat(out.size(0), 1, 1, 1)), -1)  # (B, 64, 1, 512), (1, 64, 1, 2) -> (B, 64, 1, 2) ==> (B, 64, 1, 512+2)
        
        out = self.skip_attn1(out, feats[1].view(out.size(0), 1, 256, feats[1].size(-1)))  # params: (B, 64, 1, 514), (B, 1, 256, 256)   out: (B, 64, 512)
        out = self.folding1(out, 16, 16)  # (B, 256, 256)
        out = out.unsqueeze(-2)           # (B, 256, 1, 256)
        
        out = self.skip_attn2(out, feats[0].view(out.size(0), 1, 512, feats[0].size(-1)))  # params: (B, 256, 1, 256), (B, 1, 512, 128)  out: (B, 256, 256)
        out = self.folding2(out, 16, 32)  # (B, 512, 128)
        
        out = self.folding3(out, 64, 32)  # (B, 2048, 128)

        return self.lin(out)              # (B, 2048, 3)
    
    @staticmethod
    def sample_2D(m, n):
        indeces_x = np.round(np.linspace(0, 45, m)).astype(int)
        indeces_y = np.round(np.linspace(0, 45, n)).astype(int)
        x, y = np.meshgrid(indeces_x, indeces_y)
        p = SANet.points[:, x.ravel(), y.ravel()].T.contiguous()  # (mxn, 2)
        return p
