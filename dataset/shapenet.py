import sys
sys.path.append('.')

import os
import random
import h5py

import torch
import torch.utils.data as data
import numpy as np

import dataset.util as util
from visualization import plot_pcd_one_view


class ShapeNet(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.

    TODO: data augmentation.
    """
    
    def __init__(self, dataroot, split, category, aug=False):
        assert split in ['train', 'valid', 'val', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "plane"     : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.aug = aug

        self.partial_paths, self.complete_paths = self._load_data()
    
    def __getitem__(self, index):
        if self.split == 'train':
            partial_path = self.partial_paths[index * 8 + random.randint(0, 7)]
        else:
            partial_path = self.partial_paths[index]
        complete_path = self.complete_paths[index]

        # random permute, upsampling for partial point cloud
        partial_pc = util.random_sample(util.read_point_cloud(partial_path), 2048)
        complete_pc = util.random_sample(util.read_point_cloud(complete_path), 16384)
        
        # random mirror
        if self.aug:
            partial_pc, complete_pc = util.random_mirror([partial_pc, complete_pc], random.random())

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        partial_paths, complete_paths = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            complete_paths.append(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.h5'))
            if self.split == 'train':
                for i in range(8):
                    partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.h5'.format(i)))
            else:
                partial_paths.append(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.h5'))
        
        return partial_paths, complete_paths


class ShapeNet_v2(data.Dataset):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.

    ~ All data are loaded into memory firstly.

    TODO: data augmentation.
    """
    
    def __init__(self, dataroot, split, category, aug=False):
        assert split in ['train', 'valid', 'val', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "airplane"  : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.aug = aug

        self.partial_pcs, self.complete_pcs = self._load_data()
    
    def __getitem__(self, index):
        if self.split == 'train':
            partial_pc = util.random_sample(self.partial_pcs[index * 8 + random.randint(0, 7)], 2048)
        else:
            partial_pc = util.random_sample(self.partial_pcs[index], 2048)
        complete_pc = util.random_sample(self.complete_pcs[index], 16384)
        
        if self.aug:
            partial_pc, complete_pc = util.random_mirror([partial_pc, complete_pc], random.random())

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_pcs)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        partial_pcs, complete_pcs = list(), list()

        for line in lines:
            category, model_id = line.split('/')
            complete_pcs.append(util.read_point_cloud(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.h5')))
            if self.split == 'train':
                for i in range(8):
                    partial_pcs.append(util.read_point_cloud(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '_{}.h5'.format(i))))
            else:
                partial_pcs.append(util.read_point_cloud(os.path.join(self.dataroot, self.split, 'partial', category, model_id + '.h5')))
        
        return partial_pcs, complete_pcs


class ShapeNet_v3(ShapeNet):
    def __init__(self, dataroot, split, category, aug=False):
        super().__init__(dataroot, split, category, aug)
    
    def __getitem__(self, index):
        partial_path = self.partial_paths[index]
        if self.split == 'train':
            complete_path = self.complete_paths[index // 8]
        else:
            complete_path = self.complete_paths[index]

        partial_pc = util.random_sample(util.read_point_cloud(partial_path), 2048)
        complete_pc = util.random_sample(util.read_point_cloud(complete_path), 16384)

        if self.aug:
            partial_pc, complete_pc = util.random_mirror([partial_pc, complete_pc], random.random())
        
        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)
    
    def __len__(self):
        return len(self.partial_paths)


class ShapeNet_v4(ShapeNet_v2):
    """
    ShapeNet dataset in "PCN: Point Completion Network". It contains 28974 training
    samples while each complete samples corresponds to 8 viewpoint partial scans, 800
    validation samples and 1200 testing samples.

    ~ All data are loaded into memory firstly.

    TODO: data augmentation.
    """
    
    def __init__(self, dataroot, split, category, aug=None):
        super().__init__(dataroot, split, category, aug)
    
    def __getitem__(self, index):
        partial_pc = util.random_sample(self.partial_pcs[index], 2048)
        
        if self.split == 'train':
            complete_pc = util.random_sample(self.complete_pcs[index // 8], 16384)
        else:
            complete_pc = util.random_sample(self.complete_pcs[index], 16384)

        if self.aug:
            partial_pc, complete_pc = util.random_mirror([partial_pc, complete_pc], random.random())

        return torch.from_numpy(partial_pc), torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.partial_pcs)


class ShapeNetAE:
    def __init__(self, dataroot, split, category, aug=False):
        assert split in ['train', 'valid', 'val', 'test', 'test_novel'], "split error value!"

        self.cat2id = {
            # seen categories
            "airplane"  : "02691156",  # plane
            "cabinet"   : "02933112",  # dresser
            "car"       : "02958343",
            "chair"     : "03001627",
            "lamp"      : "03636649",
            "sofa"      : "04256520",
            "table"     : "04379243",
            "vessel"    : "04530566",  # boat
            
            # alis for some seen categories
            "boat"      : "04530566",  # vessel
            "couch"     : "04256520",  # sofa
            "dresser"   : "02933112",  # cabinet
            "airplane"  : "02691156",  # airplane
            "watercraft": "04530566",  # boat

            # unseen categories
            "bus"       : "02924116",
            "bed"       : "02818832",
            "bookshelf" : "02871439",
            "bench"     : "02828884",
            "guitar"    : "03467517",
            "motorbike" : "03790512",
            "skateboard": "04225987",
            "pistol"    : "03948459",
        }

        # self.id2cat = {cat_id: cat for cat, cat_id in self.cat2id.items()}

        self.dataroot = dataroot
        self.split = split
        self.category = category
        self.aug = aug

        self.pcs = self._load_data()
    
    def __getitem__(self, index):
        pc = util.random_sample(self.pcs[index], 16384)
        if self.aug:
            pc = util.random_mirror([pc], random.random())[0]
        return torch.from_numpy(pc)

    def __len__(self):
        return len(self.pcs)

    def _load_data(self):
        with open(os.path.join(self.dataroot, '{}.list').format(self.split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        pcs = list()

        for line in lines:
            category, model_id = line.split('/')
            pcs.append(util.read_point_cloud(os.path.join(self.dataroot, self.split, 'complete', category, model_id + '.h5')))
        
        return pcs


class ShapeNetCars(data.Dataset):
    
    def __init__(self, dataroot):

        self.cat2id = {
            "car"       : "02958343",
        }

        self.category = 'car'

        self.dataroot = dataroot

        self.complete_paths = self._load_data('train')
        self.complete_paths.extend(self._load_data('valid'))
        self.complete_paths.extend(self._load_data('test'))
    
    def __getitem__(self, index):
        complete_path = self.complete_paths[index]

        complete_pc = util.random_sample(util.read_point_cloud(complete_path), 16384)

        return torch.from_numpy(complete_pc)

    def __len__(self):
        return len(self.complete_paths)

    def _load_data(self, split):
        with open(os.path.join(self.dataroot, '{}.list').format(split), 'r') as f:
            lines = f.read().splitlines()

        if self.category != 'all':
            lines = list(filter(lambda x: x.startswith(self.cat2id[self.category]), lines))
        
        complete_paths = list()

        for line in lines:
            category, model_id = line.split('/')
            complete_paths.append(os.path.join(self.dataroot, split, 'complete', category, model_id + '.h5'))
        
        return complete_paths


if __name__ == '__main__':
    # import time

    # start = time.time()
    # dataset = ShapeNet_v3('/media/server/new/datasets/PCN_h5_f4', 'train', 'all', True)
    # end = time.time()

    # print((end - start) / 60, 'minutes')
    # print("samples:", len(dataset))
    
    # # incomplete, complete = dataset[random.randint(0, len(dataset) - 1)]
    # incomplete, complete = dataset[1]

    # incomplete = incomplete.numpy()
    # complete = complete.numpy()
    # print(incomplete.shape, complete.shape)

    # print(np.min(incomplete), np.max(complete))
    # print(np.min(complete), np.max(complete))

    # plot_pcd_one_view('temp/pcn.png', [incomplete, complete], ['Partial', 'Complete'], cmap='jet', xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
    shapenetcars = ShapeNetCars('/home/scut/workspace/liuqing/dataset/PCN')
    print(len(shapenetcars))
