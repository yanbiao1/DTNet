import sys
sys.path.append('.')

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F

from torch.autograd import Variable

import extensions.MDS.MDS_module as MDS_module
import extensions.expansion_penalty.expansion_penalty_module as expansion
from extensions.emd.emd_module import emdModule


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
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)
        x = x.view(-1, 1024)
        return x


class PointGenCon(nn.Module):
    def __init__(self, bottleneck_size=8192):
        self.bottleneck_size = bottleneck_size
        super(PointGenCon, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size, 1)
        self.conv2 = torch.nn.Conv1d(
            self.bottleneck_size, self.bottleneck_size//2, 1)
        self.conv3 = torch.nn.Conv1d(
            self.bottleneck_size//2, self.bottleneck_size//4, 1)
        self.conv4 = torch.nn.Conv1d(self.bottleneck_size//4, 3, 1)

        self.th = nn.Tanh()
        self.bn1 = torch.nn.BatchNorm1d(self.bottleneck_size)
        self.bn2 = torch.nn.BatchNorm1d(self.bottleneck_size//2)
        self.bn3 = torch.nn.BatchNorm1d(self.bottleneck_size//4)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.th(self.conv4(x))
        return x


class PointNetRes(nn.Module):
    def __init__(self):
        super(PointNetRes, self).__init__()
        self.conv1 = torch.nn.Conv1d(4, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1088, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 256, 1)
        self.conv6 = torch.nn.Conv1d(256, 128, 1)
        self.conv7 = torch.nn.Conv1d(128, 3, 1)

        self.bn1 = torch.nn.BatchNorm1d(64)
        self.bn2 = torch.nn.BatchNorm1d(128)
        self.bn3 = torch.nn.BatchNorm1d(1024)
        self.bn4 = torch.nn.BatchNorm1d(512)
        self.bn5 = torch.nn.BatchNorm1d(256)
        self.bn6 = torch.nn.BatchNorm1d(128)
        self.bn7 = torch.nn.BatchNorm1d(3)
        self.th = nn.Tanh()

    def forward(self, x):
        npoints = x.size()[2]
        x = F.relu(self.bn1(self.conv1(x)))
        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x, _ = torch.max(x, 2)  # (B, 1024)
        x = x.view(-1, 1024)
        x = x.view(-1, 1024, 1).repeat(1, 1, npoints)  # (B, 1024, N)
        x = torch.cat([x, pointfeat], 1)               # (B, 1088, N)
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.th(self.conv7(x))
        return x


class MSN(nn.Module):
    def __init__(self, num_points=8192, bottleneck_size=1024, n_primitives=16):
        super(MSN, self).__init__()
        self.num_points = num_points
        self.bottleneck_size = bottleneck_size
        self.n_primitives = n_primitives
        self.encoder = nn.Sequential(
            PointNetfeat(),
            nn.Linear(1024, self.bottleneck_size),
            nn.BatchNorm1d(self.bottleneck_size),
            nn.ReLU()
        )
        self.decoder = nn.ModuleList([PointGenCon(
            bottleneck_size=2 + self.bottleneck_size) for i in range(0, self.n_primitives)])
        self.res = PointNetRes()
        self.expansion = expansion.expansionPenaltyModule()

        self.emd = emdModule()

    def forward(self, x):
        """
        Parameters
        ----------
            x: (B, N, 3)
        """
        x = x.transpose(1, 2)
        partial = x
        x = self.encoder(x)  # (B, 1024)
        outs = []
        for i in range(0, self.n_primitives):
            rand_grid = Variable(torch.cuda.FloatTensor(x.size(0), 2, self.num_points // self.n_primitives))
            rand_grid.data.uniform_(0, 1)
            y = x.unsqueeze(2).expand(x.size(0), x.size(1), rand_grid.size(2)).contiguous()
            y = torch.cat((rand_grid, y), 1).contiguous()
            outs.append(self.decoder[i](y))

        outs = torch.cat(outs, 2).contiguous()
        out1 = outs.transpose(1, 2).contiguous()  # coarse output

        dist, _, mean_mst_dis = self.expansion(out1, self.num_points // self.n_primitives, 1.5)
        loss_mst = torch.mean(dist)

        id0 = torch.zeros(outs.shape[0], 1, outs.shape[2]).cuda().contiguous()
        outs = torch.cat((outs, id0), 1)
        id1 = torch.ones(partial.shape[0], 1, partial.shape[2]).cuda().contiguous()
        partial = torch.cat((partial, id1), 1)
        xx = torch.cat((outs, partial), 2)

        resampled_idx = MDS_module.minimum_density_sample(xx[:, 0:3, :].transpose(1, 2).contiguous(), out1.shape[1], mean_mst_dis)
        xx = MDS_module.gather_operation(xx, resampled_idx)
        delta = self.res(xx)
        xx = xx[:, 0:3, :]
        out2 = (xx + delta).transpose(2, 1).contiguous()  # (B, N_out, 3)

        return out1, out2, loss_mst
    
    def train_iter(self, x, gt):
        out1, out2, loss_mst = self.forward(x)
        
        dist, _ = self.emd(out1, gt, 0.005, 50)
        emd1 = torch.sqrt(dist).mean(1).mean()

        dist, _ = self.emd(out2, gt, 0.005, 50)
        emd2 = torch.sqrt(dist).mean(1).mean()

        total_loss = emd1 + emd2 + loss_mst * 0.1

        return loss_mst, emd1, emd2, total_loss
    
    def eval_iter(self, x, gt):
        out1, out2, _ = self.forward(x)
        dist, _ = self.emd(out2, gt, 0.004, 3000)
        emd2 = torch.sqrt(dist).mean(1)
        return out1, out2, emd2.mean()


if __name__ == '__main__':
    x = torch.randn(8, 2048, 3).cuda()
    model = MSN(num_points=8192, bottleneck_size=1024).cuda()
    out1, out2, loss_mst = model(x)
    print('out1:', out1.shape)
    print('out2:', out2.shape)
    print('loss_mst:', loss_mst.shape)
