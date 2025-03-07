import argparse
import os
import random
import torch
import open3d as o3d
import numpy as np

from pathlib import Path
from models import Model
from visualization import plot_pcd_one_view


def read_point_cloud(filename):
    suffix = Path(filename).suffix
    if suffix in ['.ply', '.pcd', '.obj']:
        pc = o3d.io.read_point_cloud(filename)
        points = np.array(pc.points)
    elif suffix == '.npy':
        points = np.load(filename)
    else:
        raise Exception('不支持的文件格式')
    return np.asarray(points, dtype=np.float32)


def write_point_cloud(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc)


parser = argparse.ArgumentParser('Point Cloud Completion')
parser.add_argument('--partial_path', type=str, help='The path of partial point cloud')
parser.add_argument('--output_file', type=str, help='The path of completion point cloud')
parser.add_argument('--output_image', type=str, help='The image of visualization')
params = parser.parse_args()

model = Model(num_pc=256, num_down=256, ratios=[4, 8]).cuda()
model.load_state_dict(torch.load('checkpoints/pretrained.pth'))
model.eval()

# 点云补全
points = read_point_cloud(params.partial_path)
p = torch.from_numpy(points).cuda().unsqueeze(0)
with torch.no_grad():
    coarse, _, _, denoiser, _, refine1, pred = model(p)
    pred = pred.squeeze(0).detach().cpu().numpy()

write_point_cloud(params.output_file, points)
plot_pcd_one_view(params.output_image, [points, pred], ['Partial', 'Completion'])

