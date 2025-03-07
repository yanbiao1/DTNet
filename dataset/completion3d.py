import sys
sys.path.append('.')

import os
import random
import h5py as h5

import torch
import torch.utils.data as data
import numpy as np

import dataset.util as util
from visualization.visualization import plot_pcd_one_view


class Completion3D(data.Dataset):
    """
    Completion3D dataset. All point clouds are in bound box with range [-0.5, 0.5]^3.

    Attributes
    ----------
        data_root: data path.
        category: category of data.
        split: 'train' or 'val'. Completion3D's testing spliting has not ground truth.
    """
    
    def __init__(self, data_root, split, category='all', aug=False, scale=False):
        super().__init__()

        assert split in ['train', 'val', 'test'], "No such split data."

        self.data_root = data_root
        self.split = split
        self.category = category
        self.aug = aug
        self.scale = scale

        self.cat2id = {
            "plane"      : "02691156",
            "cabinet"    : "02933112",
            "car"        : "02958343",
            "chair"      : "03001627",
            "lamp"       : "03636649",
            "couch"      : "04256520",
            "table"      : "04379243",
            "watercraft" : "04530566",
        }

        if split != 'test':
            self.partial_pcs, self.complete_pcs = self._load_data()
        else:
            self.partial_pcs = self._load_test_data()
    
    def __getitem__(self, index):
        if self.split == 'test':
            return torch.from_numpy(self.partial_pcs[index])
        
        partial = util.random_sample(self.partial_pcs[index], 2048)
        complete = util.random_sample(self.complete_pcs[index], 2048)

        # random mirror
        if self.aug:
            partial, complete = util.random_mirror([partial, complete], random.random())
        if self.scale:
            partial, complete = util.scale_points([partial, complete])

        return torch.from_numpy(partial), torch.from_numpy(complete)
            

    def __len__(self):
        return len(self.partial_pcs)
    
    def _load_data(self):
        if self.category == 'all':
            partial_pcs, complete_pcs = list(), list()
            for category in self.cat2id:
                ps, cs = self._load_single_category_data(category)
                partial_pcs.extend(ps)
                complete_pcs.extend(cs)
            return partial_pcs, complete_pcs
        else:
            return self._load_single_category_data(self.category)
    
    def _load_test_data(self):
        test_dir = os.path.join(self.data_root, 'test', 'partial', 'all')
        
        partial_pcs = list()
        for filename in os.listdir(test_dir):
            pc = self._load_h5(os.path.join(test_dir, filename))
            partial_pcs.append(pc)
        
        return partial_pcs

    def _load_single_category_data(self, category):
        complete_dir = os.path.join(self.data_root, self.split, 'gt', self.cat2id[category])

        partial_pcs = list()
        complete_pcs = list()

        for filename in os.listdir(complete_dir):
            complete_path = os.path.join(complete_dir, filename)
            complete_pcs.append(self._load_h5(complete_path))
            partial_pcs.append(self._load_h5(complete_path.replace('gt', 'partial')))
        
        return partial_pcs, complete_pcs
    
    def _load_h5(self, filename):
        with h5.File(filename, 'r') as f:
            pc_array = np.asarray(f['data'], dtype=np.float32)
        return pc_array


if __name__ == '__main__':
    dataset = Completion3D('/media/server/new/datasets/Completion3D', 'val', 'all', aug=True)
    p, c = dataset[50]
    p = p.numpy()
    c = c.numpy()
    print(np.min(p), np.max(p))
    print(np.min(c), np.max(c))
    plot_pcd_one_view('temp/completion3d.png', [p, c], ['partial', 'complete'], cmap='jet')
