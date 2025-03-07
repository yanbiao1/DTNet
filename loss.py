import torch
import open3d as o3d

from torch import nn, Tensor
from torch.nn import functional as F

from extensions.emd.emd_module import emdFunction
from extensions.chamfer_distance.dist_chamfer_3D import ChamferDistance
from extensions.earth_movers_distance.emd import EarthMoverDistance


CD = ChamferDistance()
EMD = EarthMoverDistance()


def l1_cd(pcs1, pcs2):
    d1, d2, _, _ = CD(pcs1, pcs2)
    d1 = torch.mean(torch.sqrt(d1))
    d2 = torch.mean(torch.sqrt(d2))
    return (d1 + d2) / 2


def l1_dcd(pcd1, pcd2):
    d1, _, _, _ = CD(pcd1, pcd2)
    d1 = torch.mean(torch.sqrt(d1))
    return d1


def l1_cd_batch(pcs1, pcs2):
    d1, d2, _, _ = CD(pcs1, pcs2)
    d1 = torch.mean(torch.sqrt(d1), dim=1)
    d2 = torch.mean(torch.sqrt(d2), dim=1)
    return (d1 + d2) / 2


def l2_cd(pcs1, pcs2):
    d1, d2, _, _ = CD(pcs1, pcs2)
    return torch.mean(d1) + torch.mean(d2)


def l2_dcd(pcd1, pcd2):
    d1, d2, _, _ = CD(pcd1, pcd2)
    d1 = torch.mean(d1)
    return d1


def emd1(pcs1, pcs2):
    """
    EMD Loss.

    Args:
        pcs1 (torch.Tensor): (b, N, 3)
        pcs2 (torch.Tensor): (b, N, 3)
    """
    dists = EMD(pcs1, pcs2)
    return torch.mean(dists) / pcs1.shape[1]


def emd2(pcs1, pcs2, eps=0.005, iters=50):
    """
    Args:
        pcs1: prediction, (B, N, 3)
        pcs2: gt, (B, N, 3)
              N must be div by 1024
    """
    dist, _ = emdFunction.apply(pcs1, pcs2, eps, iters)
    return torch.mean(torch.sqrt(dist))


def f_score(pred, gt, th=0.01):
    """
    References: https://github.com/lmb-freiburg/what3d/blob/master/util.py

    Args:
        pred (np.ndarray): (N1, 3)
        gt   (np.ndarray): (N2, 3)
        th   (float): a distance threshhold
    """
    pred = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pred))
    gt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(gt))

    dist1 = pred.compute_point_cloud_distance(gt)
    dist2 = gt.compute_point_cloud_distance(pred)

    recall = float(sum(d < th for d in dist2)) / float(len(dist2))
    precision = float(sum(d < th for d in dist1)) / float(len(dist1))
    return 2 * recall * precision / (recall + precision) if recall + precision else 0
