import random

import torch

from models.model4 import AutoEncoder
from dataset import ShapeNet_v3
from loss import l1_cd, l1_cd_batch
from visualization import plot_pcd_one_view
from models.utils import fps
from torch.utils.data.dataloader import DataLoader


dataset = ShapeNet_v3('/home/scut/workspace/liuqing/dataset/PCN', 'test', 'all')
dataloader = DataLoader(dataset, 32, True)

model = AutoEncoder(512, 256).cuda()
model.load_state_dict(torch.load('log/shapenet/coarse_256/all/checkpoints/best_l1_cd.pth'))
model.eval()


def mirror(in_cloud, c, theta):
    mirror_c = torch.cat([c[:, :, :1], c[:, :, 1:2], -c[:, :, 2:]], dim=2)
    in_cloud = fps(in_cloud, 1024)
    mirror_in_cloud = torch.cat([in_cloud[:, :, :1], in_cloud[:, :, 1:2], -in_cloud[:, :, 2:]], dim=2)
    return mirror_c, torch.cat([in_cloud, torch.where((l1_cd_batch(c, mirror_c) < theta).reshape(c.shape[0], 1, 1), mirror_in_cloud, in_cloud)], dim=1)


for p, c in dataloader:
    p, c = p.cuda(), c.cuda()
    _, pred = model(p.transpose(1, 2).contiguous())
    mirror_pred = torch.cat([pred[:, :, :1], pred[:, :, 1:2], -pred[:, :, 2:]], dim=2)
    print(l1_cd_batch(pred, mirror_pred))
    break
