import os
import random
import sys
sys.path.append('.')

import h5py
import numpy as np
import torch

from torch.utils.data import Dataset

from visualization import plot_pcd_one_view
from dataset.util import sphere_normalize_pairs, random_mirror, random_sample


class MVP(Dataset):
    """
    MVP dataset. Once you call the __getitem__, it will return a partial scan and corresponding
    ground truth.
    """
    
    def __init__(self, data_root, category, split, npoints, normalize=False, aug=False):
        super().__init__()

        self.cat2labels = {'plane':   0,
                           'dresser': 1,
                           'car':     2,
                           'chair':   3,
                           'lamp':    4,
                           'sofa':    5,
                           'table':   6,
                           'boat':    7}

        assert split in ['train', 'test'], "illegal spliting, only can be 'train' or 'test'"
        assert npoints in [2048, 4096, 8192, 16384], "illegal npoints, only can be 2048, 4096, 8192 or 16384"

        self.data_root = data_root
        self.category = category
        self.split = split
        self.npoints = npoints
        self.normalize = normalize
        self.aug = aug

        self.partials, self.completes = self._load_data()
    
    def __getitem__(self, index):
        partial = self.partials[index]
        complete = self.completes[index // 26]
        if self.normalize:
            partial, complete = sphere_normalize_pairs(partial, complete)
        if self.aug:
            partial = random_sample(partial, 2048)
            complete = random_sample(complete, self.npoints)
            partial, complete = random_mirror([partial, complete], random.random())
        return torch.from_numpy(partial), torch.from_numpy(complete)

    def __len__(self):
        return len(self.partials)

    def _load_data(self):
        partial_file = os.path.join(self.data_root, 'mvp_{}_input.h5'.format(self.split))
        complete_file = os.path.join(self.data_root, 'mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints))

        if self.category != 'all':
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])[np.array(fp['labels']) == self.cat2labels[self.category]]
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])[np.array(fc['labels']) == self.cat2labels[self.category]]
        else:
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])
        
        return partial_pcs, complete_pcs


class MVP_v2(Dataset):
    """
    MVP dataset. Once you call __getitem__, it will return ground truth and a random viewpoint partial scan.
    """
    
    def __init__(self, data_root, category, split, npoints, normalize=False):
        super().__init__()

        self.cat2labels = {'plane':   0,
                           'dresser': 1,
                           'car':     2,
                           'chair':   3,
                           'lamp':    4,
                           'sofa':    5,
                           'table':   6,
                           'boat':    7}

        assert split in ['train', 'test'], "illegal spliting, only can be 'train' or 'test'"
        assert npoints in [2048, 4096, 8192, 16384], "illegal npoints, only can be 2048, 4096, 8192 or 16384"

        self.data_root = data_root
        self.category = category
        self.split = split
        self.npoints = npoints
        self.normalize = normalize

        self.partials, self.completes = self._load_data()
    
    def __getitem__(self, index):
        partial = self.partials[random.randint(index * 26, (index + 1) * 26 - 1)]
        complete = self.completes[index]
        if self.normalize:
            sphere_normalize_pairs(partial, complete)
        return torch.from_numpy(partial), torch.from_numpy(complete)

    def __len__(self):
        return len(self.completes)

    def _load_data(self):
        partial_file = os.path.join(self.data_root, 'mvp_{}_input.h5'.format(self.split))
        complete_file = os.path.join(self.data_root, 'mvp_{}_gt_{}pts.h5'.format(self.split, self.npoints))

        if self.category != 'all':
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])[np.array(fp['labels']) == self.cat2labels[self.category]]
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])[np.array(fc['labels']) == self.cat2labels[self.category]]
        else:
            with h5py.File(partial_file, 'r') as fp:
                partial_pcs =  np.array(fp['incomplete_pcds'])
            with h5py.File(complete_file, 'r') as fc:
                complete_pcs =  np.array(fc['complete_pcds'])
        
        return partial_pcs, complete_pcs


if __name__ == '__main__':
    import open3d as o3d

    dataset = MVP('/home/scut/workspace/liuqing/dataset/MVP', category='all', split='train', npoints=16384)
    
    index = random.randint(0, len(dataset) - 1)
    incomplete, complete = dataset[index]
    incomplete = incomplete.numpy()
    complete = complete.numpy()

    print(np.min(incomplete), np.max(complete))
    print(np.min(complete), np.max(complete))

    # incomplete, complete = normalize_pairs(incomplete, complete)

    # print(np.min(incomplete), np.max(complete))
    # print(np.min(complete), np.max(complete))
    
    # p_pc = o3d.geometry.PointCloud()
    # p_pc.points = o3d.utility.Vector3dVector(incomplete)
    # o3d.io.write_point_cloud('temp/partial.ply', p_pc, write_ascii=True)

    # c_pc = o3d.geometry.PointCloud()
    # c_pc.points = o3d.utility.Vector3dVector(complete)
    # o3d.io.write_point_cloud('temp/complete.ply', c_pc, write_ascii=True)
    plot_pcd_one_view('temp/mvp.png', [incomplete, complete], ['Partial', 'Complete'])

    # dataset = MVP_v2('/root/autodl-tmp/data/MVP', category='chair', split='train', npoints=8192)
    # incomplete, complete = dataset[random.randint(0, len(dataset) - 1)]
    # incomplete = incomplete.numpy()
    # complete = complete.numpy()
    # plot_pcd_one_view('temp/temp2.png', [incomplete, complete], ['incomplete', 'complete'])