import math
import random

import h5py
import numpy as np
import open3d as o3d
import torch
import transforms3d

from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation


def normalize_pc(pc):
    """
    Normalize point cloud into unit bounding box [-1.0, 1.0]^3.

    Parameters
    ----------
        pc: np.ndarray with size of (N, 3)
    """
    min_bound = np.amin(pc, axis=0)
    max_bound = np.amax(pc, axis=0)
    box_size = max_bound - min_bound
    scale_factor = 2.0 / np.amax(box_size)
    pc = pc - min_bound
    pc = pc * scale_factor
    curr_center = (np.amax(pc, axis=0) + np.amin(pc, axis=0)) / 2.0
    pc = pc - curr_center
    return pc


def normalize_pairs(partial, complete):
    """
    Normalize partial point cloud and corresponding complete point cloud
    into unit bounding box [-1.0, 1.0]^3.

    Parameters
    ----------
        partial:  np.ndarray with size of (N, 3)
        complete: np.ndarray with size of (M, 3)
    
    Returns
    -------
        partial
        complete
    """
    min_bound = np.amin(complete, axis=0)
    max_bound = np.amax(complete, axis=0)
    box_size = max_bound - min_bound
    scale_factor = 1.0 / np.amax(box_size)  # modify 2.0 to 1.0 means normalize into [-0.5, 0.5]

    partial = partial - min_bound
    complete = complete - min_bound

    partial = partial * scale_factor
    complete = complete * scale_factor
    
    curr_center = (np.amax(complete, axis=0) + np.amin(complete, axis=0)) / 2.0
    partial = partial - curr_center
    complete = complete - curr_center
    
    return partial, complete


def sphere_normalize_pc(pc):
    """
    Normalize point cloud into unit sphere whose center and radius is (0, 0) and 1 respectively.

    Parameters
    ----------
        pc: np.ndarray with size of (N, 3)
    """
    center = (np.amax(pc, axis=0) + np.amin(pc, axis=0)) / 2.0
    # pc = pc - np.mean(pc, axis=0)
    pc = pc - center
    dist = np.max(np.sqrt(np.sum(pc ** 2.0, axis=1)))
    pc = pc / dist

    return pc


def sphere_normalize_pairs(partial, complete):
    """
    Normalize point cloud into unit sphere whose center and radius is (0, 0) and 0.5 respectively.

    Parameters
    ----------
        partial:  np.ndarray with size of (N, 3)
        complete: np.ndarray with size of (M, 3)
    
    Returns
    -------
        partial
        complete
    """
    # center = np.mean(complete, axis=0)
    center = (np.amax(complete, axis=0) + np.amin(complete, axis=0)) / 2.0
    complete = complete - center
    partial = partial - center

    dist = np.max(np.sqrt(np.sum(complete ** 2.0, axis=1)))
    complete = complete / (2 * dist)
    partial = partial / (2 * dist)

    return partial, complete


def read_point_cloud(path):
    with h5py.File(path, 'r') as f:
        pc = np.asarray(f['data'])
    return pc


def write_point_cloud(path, pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.io.write_point_cloud(path, point_cloud, write_ascii=True)


def augment_cloud(Ps, args):
    """
    Augmentation on XYZ and jittering of everything
    """
    M = transforms3d.zooms.zfdir2mat(1)
    if args.pc_augm_scale > 1:
        s = random.uniform(1/args.pc_augm_scale, 1)
        M = np.dot(transforms3d.zooms.zfdir2mat(s), M)
    if args.pc_augm_rot:
        angle = random.uniform(0, 2*math.pi)
        M = np.dot(transforms3d.axangles.axangle2mat([0,1,0], angle), M) # y=upright assumption
    if args.pc_augm_mirror_prob > 0: # mirroring x&z, not y
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [1,0,0]), M)
        if random.random() < args.pc_augm_mirror_prob/2:
            M = np.dot(transforms3d.zooms.zfdir2mat(-1, [0,0,1]), M)
    result = []
    for P in Ps:
        P[:,:3] = np.dot(P[:,:3], M.T)
        result.append(P)
    return result


def fps(x : torch.Tensor, n : int):
    """
    Args:
        x: (B, N, 3)
        n: int
    
    Returns:
        (B, n, 3)
    """
    index = furthest_point_sample(x, n)
    fps_x = gather_operation(x.transpose(1, 2).contiguous(), index)
    return fps_x


def random_sample(pc, n):
    idx = np.random.permutation(pc.shape[0])
    if idx.shape[0] < n:
        idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
    return pc[idx[:n]]


def random_mirror(pcs, rnd_value):
    trfm_mat = transforms3d.zooms.zfdir2mat(1)
    trfm_mat_x = np.dot(transforms3d.zooms.zfdir2mat(-1, [1, 0, 0]), trfm_mat)
    trfm_mat_z = np.dot(transforms3d.zooms.zfdir2mat(-1, [0, 0, 1]), trfm_mat)

    if rnd_value <= 0.25:
        trfm_mat = np.dot(trfm_mat_x, trfm_mat)
        trfm_mat = np.dot(trfm_mat_z, trfm_mat)
    elif rnd_value > 0.25 and rnd_value <= 0.5:    # lgtm [py/redundant-comparison]
        trfm_mat = np.dot(trfm_mat_x, trfm_mat)
    elif rnd_value > 0.5 and rnd_value <= 0.75:
        trfm_mat = np.dot(trfm_mat_z, trfm_mat)

    
    for pc in pcs:
        pc[:, :3] = np.dot(pc[:, :3], trfm_mat.T)
    
    return pcs


def scale_points(pcs, scale=None):
    if scale is None:
        # scale = np.random.randint(85, 95) * 0.01
        scale = np.random.uniform(1/1.6, 1)
    
    for pc in pcs:
        pc[:, :3] = pc[:, :3] * scale
    
    return pcs
