import sys
sys.path.append('.')

import os
import random

import numpy as np
import torch
import open3d as o3d

from torch.utils.data import Dataset

from dataset.util import random_sample


class KITTI(Dataset):

    def __init__(self, dataroot='/home/scut/workspace/liuqing/dataset/KITTI'):
        super().__init__()

        self.dataroot = dataroot
        self.filenames = os.listdir(os.path.join(dataroot, 'cars'))
    
    def __getitem__(self, index):
        model_id = self.filenames[index][:-4]
        pcd = o3d.io.read_point_cloud(os.path.join(self.dataroot, 'cars', '{}.pcd'.format(model_id)))
        pcd = np.array(pcd.points)
        bbox = np.loadtxt(os.path.join(self.dataroot, 'bboxes', '{}.txt'.format(model_id)))

        center = (bbox.min(0) + bbox.max(0)) / 2
        bbox -= center
        yaw = np.arctan2(bbox[3, 1] - bbox[0, 1], bbox[3, 0] - bbox[0, 0])
        rotation = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                            [np.sin(yaw), np.cos(yaw), 0],
                            [0, 0, 1]])
        bbox = np.dot(bbox, rotation)
        scale = bbox[3, 0] - bbox[0, 0]
        bbox /= scale

        partial = np.dot(pcd - center, rotation) / scale
        partial = np.dot(partial, [[1, 0, 0], [0, 0, 1], [0, 1, 0]])

        partial = np.asarray(partial, dtype=np.float32)
        partial_ = random_sample(partial, 2048)

        return model_id, pcd, torch.from_numpy(partial), torch.from_numpy(partial_)
    
    def __len__(self):
        return len(self.filenames)


if __name__ == '__main__':
    kitti = KITTI()
    index = random.randint(0, len(kitti) - 1)
    model_id, pcd, partial, partial_ = kitti[index]
    print(model_id)
    print(pcd)
    print(partial)
    print(partial_)