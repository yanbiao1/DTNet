import os
import random
import h5py as h5

import torch
import torch.utils.data as data
import numpy as np

import dataset.util as util


class CRN(data.Dataset):
    """
    """

    def __init__(self, dataroot, split, category='all', normalize=False, aug=False):
        super().__init__()

        assert split in ['train', 'valid', 'test'], "No such split data."

        self.cat2labels = {
            'airplane': 0,
            'plane':   0,
            'dresser': 1,
            'cabinet': 1,
            'car':     2,
            'chair':   3,
            'lamp':    4,
            'sofa':    5,
            'table':   6,
            'boat':    7,
            'vessel':  7
        }

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.normalize = normalize
        self.aug = aug

        self.partial_pcs, self.complete_pcs = self._load_data()

    def __getitem__(self, index):
        partial_pc = self.random_sample(self.partial_pcs[index], 2048)
        complete_pc = self.random_sample(self.complete_pcs[index], 2048)

        # normalize into bounding box [-0.5, 0.5] like completion3d
        if self.normalize:
            partial_pc, complete_pc = util.normalize_pairs(partial_pc, complete_pc)

        # random mirror
        if self.aug:
            partial_pc, complete_pc = util.random_mirror([partial_pc, complete_pc], random.random())

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_pcs)

    def _load_data(self):
        h5_file = os.path.join(self.dataroot, '{}_data.h5'.format(self.split))

        with h5.File(h5_file, 'r') as f:
            if self.category != 'all':
                partial_pcs = np.array(f['incomplete_pcds'])[np.array(f['labels']) == self.cat2labels[self.category]]
                complete_pcs = np.array(f['complete_pcds'])[np.array(f['labels']) == self.cat2labels[self.category]]
            else:
                partial_pcs = np.array(f['incomplete_pcds'])
                complete_pcs = np.array(f['complete_pcds'])
        
        return partial_pcs, complete_pcs
    
    def random_sample(self, pc, n):
        idx = np.random.permutation(pc.shape[0])
        if idx.shape[0] < n:
            idx = np.concatenate([idx, np.random.randint(pc.shape[0], size=n-pc.shape[0])])
        return pc[idx[:n]]
