import sys
sys.path.append('.')

import math

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import pointnet2_ops.pointnet2_utils as pn2

from pointnet2_ops.pointnet2_modules import PointnetSAModuleMSG


def symmetric_sample(points, num):
    """
    Parameters
    ----------
        points: (B, N, 3)
        num: int, equals 256 in default.
    
    Returns
    -------
        input_fps: (B, 2xnum, 3)
    """
    p1_idx = pn2.furthest_point_sample(points, num)  # (B, num)
    input_fps = pn2.gather_operation(points.transpose(1, 2).contiguous(), p1_idx).transpose(1, 2).contiguous()  # (B, num, 3)
    x = torch.unsqueeze(input_fps[:, :, 0], dim=2)
    y = torch.unsqueeze(input_fps[:, :, 1], dim=2)
    z = torch.unsqueeze(-input_fps[:, :, 2], dim=2)
    input_fps_flip = torch.cat([x, y, z], dim=2)
    input_fps = torch.cat([input_fps, input_fps_flip], dim=1)  # (B, 2xnum, 3)
    return input_fps


class MLP(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('fc_%d' % (i+1), nn.Linear(num_channels, dims[i+1]))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())
   
    def forward(self, features):
        return self.model(features)


class MLPConv(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.model = nn.Sequential()
        for i, num_channels in enumerate(dims[:-1]):
            self.model.add_module('conv1d_%d' % (i+1), nn.Conv1d(num_channels, dims[i+1], kernel_size=1))
            if i != len(dims) - 2:
                self.model.add_module('relu_%d' % (i+1), nn.ReLU())

    def forward(self, inputs):
        return self.model(inputs)


class ContractExpandOperation(nn.Module):
    """
    # !!! Cannot understand.
    """
    
    def __init__(self, num_input_channels, up_ratio):
        super().__init__()
        self.up_ratio = up_ratio
        # PyTorch default padding is 'VALID'
        # !!! rmb to add in L2 loss for conv2d weights
        self.conv2d_1 = nn.Conv2d(num_input_channels, 64, kernel_size=(1, self.up_ratio), stride=(1, 1))
        self.conv2d_2 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_3 = nn.Conv2d(64, 64, kernel_size=(1, 1), stride=(1, 1))

    def forward(self, inputs):
        """
        Parameters
        ----------
            inputs: (B, 64, N)
        
        Returns
        -------
            net: (B, 64, N)
        """
        net = inputs.view(inputs.shape[0], inputs.shape[1], self.up_ratio, -1)  # (32, 64, 2, 1024)
        net = net.permute(0, 1, 3, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_1(net))  # (32, 64, 1024, 1)
        net = F.relu(self.conv2d_2(net))  # (32, 128, 1024, 1)
        net = net.permute(0, 2, 3, 1).contiguous()  # (32, 1024, 1, 128)
        net = net.view(net.shape[0], -1, self.up_ratio, 64)  # (32, 1024, 2, 64)
        net = net.permute(0, 3, 1, 2).contiguous()  # (32, 64, 1024, 2)
        net = F.relu(self.conv2d_3(net)) # (32, 64, 1024, 2)
        net = net.view(net.shape[0], 64, -1)  # (32, 64, 2048)
        return net


class Encoder(nn.Module):
    def __init__(self, embed_size=1024):
        super().__init__()

        self.conv1 = MLPConv([3, 128, 256])
        self.conv2 = MLPConv([512, 512, embed_size])

    def forward(self, inputs):
        '''
        Parameters
        ----------
            inputs: (B, 3, N)
        
        Returns
        -------
            features: (B, embed_size)
        '''
        features = self.conv1(inputs)                                          # [32, 256, 2048]
        
        features_global, _ = torch.max(features, 2, keepdim=True)              # [32, 256, 1]
        features_global_tiled = features_global.repeat(1, 1, inputs.shape[2])  # [32, 256, 2048]
        features = torch.cat([features, features_global_tiled], dim=1)         # [32, 512, 2048]
        
        features = self.conv2(features)                                        # [32, 1024, 2048]
        features, _ = torch.max(features, 2)                                   # [32, 1024]
        
        return features


class Decoder(nn.Module):
    def __init__(self, step_ratio=16, num_extract=512):
        super().__init__()

        self.step_ratio = step_ratio
        self.num_extract = num_extract

        self.coarse_mlp = MLP([1024, 1024, 1024, 512 * 3])

        self.mean_fc = nn.Linear(1024, 128)
        self.up_branch_mlp_conv_mf = MLPConv([1157, 128, 64])
        self.up_branch_mlp_conv_nomf = MLPConv([1029, 128, 64])
        self.contract_expand = ContractExpandOperation(64, 2)
        self.fine_mlp_conv = MLPConv([64, 512, 512, 3])

    def forward(self, code, inputs, mean_feature=None):
        '''
        Parameters
        ----------
            code: B * C
            inputs: B * C * N
            step_ratio: int
            num_extract: int
            mean_feature: B * C

        Returns
        -------
            coarse(B * N * C)
            fine(B, N, C)
        '''
        coarse = torch.tanh(self.coarse_mlp(code))    # (B, 1536)
        coarse = coarse.view(-1, 512, 3)              # (B, 512, 3)
        coarse = coarse.transpose(2, 1).contiguous()  # (B, 3, 512)

        # FPS + Mirror
        inputs_new = inputs.transpose(2, 1).contiguous()                     # (B, 2048, 3)
        input_fps = symmetric_sample(inputs_new, int(self.num_extract / 2))  # (B, 512,  3)
        input_fps = input_fps.transpose(2, 1).contiguous()                   # (B, 3,  512)
        
        level0 = torch.cat([input_fps, coarse], 2)                           # (B, 3, 1024)
        
        if self.num_extract > 512:
            level0_flipped = level0.transpose(2, 1).contiguous()
            level0 = pn2.gather_operation(level0, pn2.furthest_point_sample(level0_flipped, 1024))

        for i in range(int(math.log2(self.step_ratio))):  # step_ration is the upsample ration for 1024 points
            num_fine = 2 ** (i + 1) * 1024
            grid = self.gen_grid_up(2 ** (i + 1)).cuda().contiguous()      # (2, 2), (2, 4), (2, 8), (2, 16)
            grid = torch.unsqueeze(grid, 0)                                # (1, 2, 2)
            grid_feat = grid.repeat(level0.shape[0], 1, 1024)              # (B, 2, 2048)
            point_feat = torch.unsqueeze(level0, 3).repeat(1, 1, 1, 2)     # (B, 3, 1024, 2)
            point_feat = point_feat.view(-1, 3, num_fine)                  # (B, 3, 2048)
            global_feat = torch.unsqueeze(code, 2).repeat(1, 1, num_fine)  # (B, 1024, 2048)

            if mean_feature is not None:
                mean_feature_use = F.relu(self.mean_fc(mean_feature))                            # (B, 128)
                mean_feature_use = torch.unsqueeze(mean_feature_use, 2).repeat(1, 1, num_fine)   # (B, 128, 2048)
                feat = torch.cat([grid_feat, point_feat, global_feat, mean_feature_use], dim=1)  # (B, 1157, 2048)
                feat1 = F.relu(self.up_branch_mlp_conv_mf(feat))                                 # (B, 64, 2048)
            else:
                feat = torch.cat([grid_feat, point_feat, global_feat], dim=1)                    # (B, 1029)
                feat1 = F.relu(self.up_branch_mlp_conv_nomf(feat))                               # (B, 64, 2048)

            feat2 = self.contract_expand(feat1)  # (B, 64, 2048)
            feat = feat1 + feat2                 # (B, 64, 2048)

            fine = self.fine_mlp_conv(feat) + point_feat  # (B, 3, 2048)
            level0 = fine

        return coarse.transpose(1, 2).contiguous(), fine.transpose(1, 2).contiguous()

    def gen_grid_up(self, up_ratio, grid_size=0.2):
        sqrted = int(math.sqrt(up_ratio)) + 1
        for i in range(1, sqrted + 1).__reversed__():
            if (up_ratio % i) == 0:
                num_x = i
                num_y = up_ratio // i
                break

        grid_x = torch.linspace(-grid_size, grid_size, steps=num_x)
        grid_y = torch.linspace(-grid_size, grid_size, steps=num_y)

        x, y = torch.meshgrid(grid_x, grid_y)  # x, y shape: (2, 1)
        grid = torch.stack([x, y], dim=-1).view(-1, 2).transpose(0, 1).contiguous()
        return grid


class Generator(nn.Module):
    def __init__(self, embed_size=1024, step_ratio=16, num_extract=512):
        super(Generator, self).__init__()

        self.encoder = Encoder(embed_size=embed_size)
        self.decoder = Decoder(step_ratio=step_ratio, num_extract=num_extract)
    
    def forward(self, x, mean_feature=None):
        """
        Parameters
        ----------
            x: (B, N, 3)
            mean_feature: (B, 1024)
        
        Returns
        -------
            coarse: (B, 512, 3)
            fine: (B, 16384, 3)
        """
        x = x.transpose(1, 2).contiguous()
        code = self.encoder(x)
        coarse, fine = self.decoder(code, x, mean_feature)
        
        return coarse, fine


class Discriminator(nn.Module):
    def __init__(self, num_points=256, divide_ratio=2):
        super(Discriminator, self).__init__()
        self.pointnet_sa_module = PointnetSAModuleMSG(npoint=num_points, radii=[0.1, 0.2, 0.4], nsamples=[16, 32, 128],
                                                      mlps=[[3,  32 // divide_ratio, 32 // divide_ratio, 64 // divide_ratio],
                                                       [3,  64 // divide_ratio, 64 // divide_ratio, 128 // divide_ratio],
                                                       [3, 64 // divide_ratio, 96 // divide_ratio, 128 // divide_ratio]])
        self.patch_mlp_conv = MLPConv([(64//divide_ratio + 128 // divide_ratio + 128 // divide_ratio), 1])

    def forward(self, xyz):
        _, l1_points = self.pointnet_sa_module(xyz, features=x.transpose(1, 2).contiguous())
        patch_values = self.patch_mlp_conv(l1_points)

        return patch_values


if __name__ == '__main__':
    # ===== network for PCN dataset =====
    # generator = Generator().cuda()
    # x = torch.randn(2, 2048, 3).cuda()
    # coarse, fine = generator(x)
    # print(coarse.shape, fine.shape)

    # discriminator = Discriminator().cuda()
    # y = discriminator(fine)
    # print(y.shape)

    # === network for Completion3D dataset ===
    generator = Generator(1024, 2, 512).cuda()
    x = torch.randn(2, 2048, 3).cuda()
    coarse, fine = generator(x)
    print(coarse.shape, fine.shape)

    discriminator = Discriminator().cuda()
    y = discriminator(fine)
    print(y.shape)
