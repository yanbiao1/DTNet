import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetFPModule
from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


class STN3d(nn.Module):
    def __init__(self):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x,_ = torch.max(x, 2)
        x = x.view(-1, 1024)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1,0,0,0,1,0,0,0,1]).astype(np.float32))).view(1, 9).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class RFA(nn.Module):
    def __init__(self):
        super().__init__()

        self.num_input = 2048
        self.num_coarse = 4096
        self.num_fine = 16384
        self.grid_size = 2
        self.up_ratio = 8

        self.pointnet_sa1 = PointnetSAModule(mlp=[3, 32, 32, 64],
                                             npoint=self.num_input,
                                             radius=0.05,
                                             nsample=32,
                                             bn=False,
                                             use_xyz=True)
        self.pointnet_sa2 = PointnetSAModule(mlp=[64, 64, 64, 128],
                                             npoint=self.num_input // 2,
                                             radius=0.1,
                                             nsample=32,
                                             bn=False,
                                             use_xyz=True)
        self.pointnet_sa3 = PointnetSAModule(mlp=[128, 128, 128, 256],
                                             npoint=self.num_input // 4,
                                             radius=0.2,
                                             nsample=32,
                                             bn=False,
                                             use_xyz=True)
        self.pointnet_sa4 = PointnetSAModule(mlp=[256, 256, 256, 512],
                                             npoint=self.num_input // 8,
                                             radius=0.3,
                                             nsample=32,
                                             bn=False,
                                             use_xyz=True)
        self.pointnet_sa5 = PointnetSAModule(mlp=[512, 512, 512, 1024],
                                             npoint=self.num_input // 16,
                                             radius=0.4,
                                             nsample=32,
                                             bn=False,
                                             use_xyz=True)
        self.pointnet_gsa = PointnetSAModule(mlp=[1024, 512, 512, 1024],
                                             npoint=None,
                                             radius=None,
                                             bn=False,
                                             use_xyz=True)
        
        self.pointnet_fp1 = PointnetFPModule(mlp=[1024, 64], bn=False)
        self.pointnet_fp2 = PointnetFPModule(mlp=[1024, 64], bn=False)
        self.pointnet_fp3 = PointnetFPModule(mlp=[512, 64],  bn=False)
        self.pointnet_fp4 = PointnetFPModule(mlp=[256, 64],  bn=False)
        self.pointnet_fp5 = PointnetFPModule(mlp=[128, 64],  bn=False)

        self.transform_nets = nn.ModuleList([STN3d() for _ in range(self.up_ratio // 2)])

        self.mlps = nn.ModuleList([
            nn.Sequential(nn.Conv1d(387, 256, 1), nn.ReLU(), nn.Conv1d(256, 128, 1), nn.ReLU())
            for _ in range(self.up_ratio)
            ])

        self.shared_mlp1 = nn.Sequential(nn.Conv1d(128, 64, 1),
                                         nn.ReLU())
        self.shared_mlp2 = nn.Conv1d(64, 3, 1)
        self.score_net = nn.Sequential(nn.Conv1d(64, 16, 1),
                                       nn.ReLU(),
                                       nn.Conv1d(16, 8, 1),
                                       nn.ReLU(),
                                       nn.Conv1d(8, 1, 1),
                                       nn.ReLU())

        self.final_mlp = nn.Sequential(
            nn.Conv1d(1093, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 512, 1),
            nn.ReLU(),
            nn.Conv1d(512, 3, 1)
        )
        
        a = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(1, self.grid_size).expand(self.grid_size, self.grid_size).reshape(1, -1)
        b = torch.linspace(-0.05, 0.05, steps=self.grid_size, dtype=torch.float).view(self.grid_size, 1).expand(self.grid_size, self.grid_size).reshape(1, -1)
        
        self.folding_seed = torch.cat([a, b], dim=0).view(1, 2, self.grid_size ** 2).cuda()  # (1, 2, 4)
    
    def forward(self, x):
        """
        Parameters
        ----------
            x: (B, N=2048, 3)
        """
        B = x.shape[0]

        l0_xyz, l0_points = x, x.transpose(1, 2).contiguous()         # (B, 2048, 3), None

        # PointNet++
        l1_xyz, l1_points = self.pointnet_sa1(l0_xyz, l0_points)      # (B, 2048, 3), (B, 64,  2048)
        l2_xyz, l2_points = self.pointnet_sa2(l1_xyz, l1_points)      # (B, 1024, 3), (B, 128, 1024)
        l3_xyz, l3_points = self.pointnet_sa3(l2_xyz, l2_points)      # (B, 512,  3), (B, 256,  512)
        l4_xyz, l4_points = self.pointnet_sa4(l3_xyz, l3_points)      # (B, 256,  3), (B, 512,  256)
        l5_xyz, l5_points = self.pointnet_sa5(l4_xyz, l4_points)      # (B, 128,  3), (B, 1024, 128)
        gl_xyz, gl_points = self.pointnet_gsa(l5_xyz, l5_points)      # None,         (B, 1024,   1)

        # Feature Interpolation
        up_gl_points = self.pointnet_fp1(l0_xyz, gl_xyz, None, gl_points)  # (8, 64, 2048)
        up_l5_points = self.pointnet_fp2(l0_xyz, l5_xyz, None, l5_points)  # (8, 64, 2048)
        up_l4_points = self.pointnet_fp3(l0_xyz, l4_xyz, None, l4_points)  # (8, 64, 2048)
        up_l3_points = self.pointnet_fp4(l0_xyz, l3_xyz, None, l3_points)  # (8, 64, 2048)
        up_l2_points = self.pointnet_fp5(l0_xyz, l2_xyz, None, l2_points)  # (8, 64, 2048)

        new_points_list = []
        for i in range(self.up_ratio):
            if i > 3:
                # T-Net
                transform_mat = self.transform_nets[i - 4](l0_points)  # (B, 3, 3)
                xyz_transformed = torch.bmm(l0_xyz, transform_mat).transpose(1, 2).contiguous()  # (B, 3, 2048)

                # resudual features for unknown parts
                concat_feat = torch.cat([up_gl_points,
                                         up_gl_points - up_l5_points,
                                         up_gl_points - up_l4_points,
                                         up_gl_points - up_l3_points,
                                         up_gl_points - up_l2_points,
                                         up_gl_points - l1_points,
                                         xyz_transformed], dim=1)
            else:
                # known features
                concat_feat = torch.cat([up_gl_points,
                                         up_l5_points,
                                         up_l4_points,
                                         up_l3_points,
                                         up_l2_points,
                                         l1_points,
                                         l0_points], dim=1)
            
            # Seperated MLP
            new_points = self.mlps[i](concat_feat)
            new_points_list.append(new_points)
        
        # Shared MLP
        coord_feat = self.shared_mlp1(torch.cat(new_points_list, dim=2))  # (B, 64, 16384)
        coord = self.shared_mlp2(coord_feat)                              # (B, 3,  16384)
        coord_flip = coord.transpose(1, 2).contiguous()                   # (B, 16384,  3)

        # FPS
        fps_idx = furthest_point_sample(coord_flip, self.num_fine // 2)   # (B, 8192)
        feat_fps = gather_operation(coord_feat, fps_idx)                  # (B, 64, 8192)
        coord_fps = gather_operation(coord, fps_idx)                      # (B, 3,  8192)

        # Score and index
        score = F.softplus(self.score_net(feat_fps)).squeeze()            # (B, 4096)
        _, indices = torch.topk(score, self.num_coarse)                   # _, (B, 4096)
        coarse_points = gather_operation(coord_fps, indices.int())        # (B, 3,  4096)
        coarse_feats = gather_operation(feat_fps, indices.int())          # (B, 64, 4096)

        # Folding
        center = coarse_points.unsqueeze(3).expand(-1, -1, -1, self.grid_size ** 2).reshape(B, -1, self.num_fine)       # (B, 3,    16384)
        coarse_feats = coarse_feats.unsqueeze(3).expand(-1, -1, -1, self.grid_size ** 2).reshape(B, -1, self.num_fine)  # (B, 64,   16384)
        point_feat = torch.cat([center, coarse_feats], dim=1)                                                           # (B, 67,   16384)
        grid_feat = self.folding_seed.unsqueeze(2).expand(B, -1, self.num_coarse, -1).reshape(B, -1, self.num_fine)     # (B, 2,    16384)
        gl_feat = gl_points.expand(-1, -1, self.num_fine)                                                               # (B, 1024, 16384)
        fine_points = center + self.final_mlp(torch.cat([grid_feat, point_feat, gl_feat], dim=1))

        return coord_flip, coarse_points.transpose(1, 2).contiguous(), fine_points.transpose(1, 2).contiguous()


if __name__ == '__main__':
    model = RFA().cuda()
    x = torch.randn(2, 2048, 3).cuda()
    coarse1, coarse2, fine = model(x)
    print(coarse1.shape, coarse2.shape, fine.shape)
