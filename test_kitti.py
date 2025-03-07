import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from tqdm import tqdm
from dataset import ShapeNetCars, KITTI
from models import Model, PCN, GRNet, PointAttn, SnowflakeNet, FBNet
from models.utils import fps
from loss import l1_cd, l2_cd, l1_dcd, l1_cd_batch
from visualization import plot_pcd_one_view


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def fd(input_pc, output_pc):
    return l1_dcd(input_pc, output_pc)


def mmd(output_pc, shapenet_cars, batch_size):
    output_pc = output_pc.repeat(batch_size, 1, 1)
    cds = list()
    for car_pcs in shapenet_cars:
        car_pcs = car_pcs.cuda()
        cds.append(torch.min(l1_cd_batch(output_pc, car_pcs)).item())
    return np.min(cds)


def test(params):
    if params.plot or params.save:
        make_dir(params.result_dir)
    if params.plot:
        image_dir = os.path.join(params.result_dir, 'image')
        make_dir(image_dir)
    if params.save:
        output_dir = os.path.join(params.result_dir, 'output')
        make_dir(output_dir)

    # customized
    # model = PCN().cuda()
    # model = GRNet().cuda()
    # model = SnowflakeNet(up_factors=[4, 8]).cuda()
    # model = PointAttn('pcn').cuda()
    # model = FBNet().cuda()
    model = Model(256, 256, ratios=[4, 8]).cuda()
    model.load_state_dict(torch.load(params.ckpt_path))
    model.eval()

    kitti = KITTI('/home/scut/workspace/liuqing/dataset/KITTI')
    kitti_dataloader = DataLoader(kitti, batch_size=1, shuffle=False, num_workers=params.num_workers)

    shapenetcars = ShapeNetCars('/home/scut/workspace/liuqing/dataset/PCN')
    shapenetcars_dataloader = DataLoader(shapenetcars, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

    if params.fd:
        fds = list()
    if params.mmd:
        mmds = list()

    with torch.no_grad():
        for model_ids, pcds, p, p_ in kitti_dataloader:
            p = p.cuda()
            p_ = p_.cuda()

            pcds = model(p_)
            pred = pcds[-1]

            if params.fd:
                fds.append(fd(p, pred).item())
            if params.mmd:
                mmds.append(mmd(pred, shapenetcars_dataloader, params.batch_size))
            
            if params.plot:
                    plot_pcd_one_view(os.path.join(image_dir, '{}.png'.format(model_ids[0])), 
                                    [p[0].detach().cpu().numpy(),
                                    # coarse[0].detach().cpu().numpy(),
                                    pred[0].detach().cpu().numpy()],
                                    ['Input', 'Pred'],
                                    xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            if params.save:
                export_ply(os.path.join(output_dir, '{}.ply'.format(model_ids[0])), pred[0].detach().cpu().numpy())

    if params.fd:
        print('FD: {:.6f}'.format(np.mean(fds)))
    if params.mmd:
        print('MMD: {:.6f}'.format(np.mean(mmds)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing Point Cloud Completion on KITTI')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--num_workers', type=int, default=2, help='num_workers for data loader')
    parser.add_argument('--batch_size', type=int, default=64)
    
    parser.add_argument('--plot', action='store_true', help='Visualize by matplotlib')
    parser.add_argument('--save', action='store_true', help='Saving test result')

    parser.add_argument('--fd', action='store_true', help='Test L1 Chamfer Distance')
    parser.add_argument('--mmd', action='store_true', help='Test L2 Chamfer Distance')

    params = parser.parse_args()

    test(params)
