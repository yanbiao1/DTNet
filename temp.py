from dataset import ShapeNet, KITTI, CRN
from visualization import plot_pcd_one_view
import numpy as np
import open3d as o3d
import torch
from models.utils import fps
from models import FBNet

# dataset = KITTI()
# model = FBNet().cuda()
# model.load_state_dict(torch.load('log/kitti/FBNet/all/checkpoints/best_val_l1_cd.pth'))

# for model_id, points, partial, partial_ in dataset:
#     if model_id == 'frame_6_car_3':
#         p = partial_.unsqueeze(0).cuda()
#         preds = model(p)
#         pred = preds[-1]
#         plot_pcd_one_view('temp.png', [partial.numpy(), pred[0].detach().cpu().numpy()], titles=['input', 'gt'], comment='partial')
#         pc = o3d.geometry.PointCloud()
#         pc.points = o3d.utility.Vector3dVector(pred[0].detach().cpu().numpy())
#         o3d.io.write_point_cloud('temp.ply', pc, write_ascii=True)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


dataset = ShapeNet('/home/scut/workspace/liuqing/dataset/PCN', 'test',  'chair')
partial, complete = dataset[24]
export_ply('partial.ply', partial.numpy())
export_ply('complete.ply', complete.numpy())
