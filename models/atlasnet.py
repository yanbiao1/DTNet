import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable


class PointNetfeat(nn.Module):
    def __init__(self):
        super(PointNetfeat, self).__init__()

        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)

    def forward(self, x):
        """
        Paramrters
        ----------
            x: (B, 3, N)
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2)[0]
        
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=1024):
        super(PointGenCon, self).__init__()

        self.bottleneck_size = bottleneck_size

        self.conv1 = torch.nn.Conv1d(self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(self.bottleneck_size, int(self.bottleneck_size / 2), 1)
        self.conv3 = torch.nn.Conv1d(int(self.bottleneck_size / 2), int(self.bottleneck_size / 4), 1)
        self.conv4 = torch.nn.Conv1d(int(self.bottleneck_size / 4), 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 2))
        self.bn3 = torch.nn.BatchNorm1d(int(self.bottleneck_size / 4))

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        
        return x


class AtlasNet(nn.Module):
    """
    "AtlasNet: A Papier-Mâché Approach to Learning 3D Surface Generation"
    (https://arxiv.org/pdf/1802.05384)

    Attributes:
        num_points:   2500
        n_primitives: 4
    """
    def __init__(self, num_points=2500, bottleneck_size=1024, n_primitives=4):
        super(AtlasNet, self).__init__()

        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.npoints_per = self.num_points // self.n_primitives

        self.grid = self.gen_regular_grid1()  # (32x32, 2)

        self.encoder = nn.Sequential(
            PointNetfeat(),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )

        self.decoder = nn.ModuleList([PointGenCon(bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])

    def forward(self, x, train=True):
        B, _, _ = x.shape

        x = self.encoder(x.transpose(1, 2))  # (B, 1024)

        outs = []
        for i in range(0, self.n_primitives):
            if train:
                rand_grid = Variable(torch.cuda.FloatTensor(B, 2, self.num_points // self.n_primitives))  # (B, 2, 1024)
                rand_grid.data.uniform_(0, 1)
            else:
                rand_grid = Variable(torch.cuda.FloatTensor(self.grid))          # (npoints_per, 2)
                rand_grid = rand_grid.unsqueeze(0).transpose(1, 2)               # (1, 2, npoints_per)
                rand_grid = rand_grid.expand(B, -1, -1)                          # (B, 2, npoints_per)
            y = x.unsqueeze(2).expand(-1, -1, self.npoints_per)                  # (B, latent_dim, npoints_per)
            y = torch.cat((rand_grid, y), 1)
            outs.append(self.decoder[i](y))
        
        return torch.cat(outs, 2).transpose(1, 2).contiguous()                   # (B, N, 3)
    
    def gen_regular_grid1(self):
        up_ratio = self.num_points // self.n_primitives
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted+1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break
        
        grid_x = np.linspace(0.0, 1.0, num_x)
        grid_y = np.linspace(0.0, 1.0, num_y)

        x, y = np.meshgrid(grid_x, grid_y)
        grid = np.reshape(np.stack([x, y], axis=-1), [-1, 2])
        return grid
    
    def gen_regular_grid2(self):
        grain = int(np.sqrt(self.num_points / self.n_primitives))
        grain = grain * 1.0
        n = ((grain + 1) * (grain + 1) * self.n_primitives)
        if n < self.num_points:
            grain += 1
        vertices = []
        for i in range(0, int(grain + 1)):
                for j in range(0, int(grain + 1)):
                    vertices.append([i / grain, j / grain])
        return vertices


if __name__ == '__main__':
    model = AtlasNet(16384, 1024, 4).cuda()
    x = torch.randn(8, 3, 2048).cuda()
    y = model(x, False)
    print(y.shape)
