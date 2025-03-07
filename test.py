import os
import argparse

import numpy as np
import open3d as o3d
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from dataset import ShapeNet, Completion3D, CRN
from models import Model, FC, PCN, TopNet, GRNet, SANet, PointAttn, PMPNet, SnowflakeNet, FBNet, AblationModel
from models.utils import fps
from loss import emd1, l1_cd, l2_cd, f_score
from visualization import plot_pcd_one_view


CATEGORIES_C3D       = ['plane', 'cabinet', 'car', 'chair', 'lamp', 'couch', 'table', 'watercraft']
CATEGORIES_CRN       = ['plane', 'dresser', 'car', 'chair', 'lamp', 'sofa', 'table', 'boat']
CATEGORIES_PCN       = ['airplane', 'cabinet', 'car', 'chair', 'lamp', 'sofa', 'table', 'vessel']
CATEGORIES_PCN_NOVEL = ['bus', 'bed', 'bookshelf', 'bench', 'guitar', 'motorbike', 'skateboard', 'pistol']
CATEGORIES = {
    'pcn': CATEGORIES_PCN,
    'pcn_novel': CATEGORIES_PCN_NOVEL,
    'c3d': CATEGORIES_C3D,
    'crn': CATEGORIES_CRN
}


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def export_ply(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pc, write_ascii=True)


def test_single_category(category, model, params):
    if params.plot or params.save:
        cat_dir = os.path.join(params.result_dir, category)
        image_dir = os.path.join(cat_dir, 'image')
        output_dir = os.path.join(cat_dir, 'output')
        make_dir(cat_dir)
        make_dir(image_dir)
        make_dir(output_dir)

    lim = 0.35
    if params.dataset == 'pcn':
        test_dataset = ShapeNet('/home/scut/hdd/liuqing_scut/datasets/PCN', 'test', category)
    elif params.dataset == 'pcn_novel':
        test_dataset = ShapeNet('/home/scut/hdd/liuqing_scut/datasets/PCN', 'test_novel', category)
    elif params.dataset == 'c3d':
        test_dataset = Completion3D('/home/scut/workspace/liuqing/dataset/Completion3D', 'val', category)
        lim = 0.4
    elif params.dataset == 'crn':
        test_dataset = CRN('/home/scut/workspace/liuqing/dataset/CRN', 'test', category)
        lim = 0.4
    # print(len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=params.num_workers)

    index = 1
    if params.l1cd:
        total_l1_cd = 0.0
    if params.l2cd:
        total_l2_cd = 0.0
    if params.emd:
        total_emd = 0.0
    if params.fscore:
        total_f_score = 0.0
    with torch.no_grad():
        for p, c in test_dataloader:
            p, c = p.cuda(), c.cuda()
            # print(0)
            # ====== PCN-FC, FoldingNet, TopNet ======
            # pred = model(p)
            # ====== PCN, GRNet =======
            # coarse, pred = model(p)
            # ====== FBNet =======
            # pcds = model(p)
            # pred = pcds[-1]
            # ====== PointAttn =====
            # coarse, refine1, pred = model(p)
            # ====== PMP-Net =======
            # preds = []
            # for t in range(8):
            #     pcds, deltas = model(p)
            # pred = pcds[2]
            # preds.append(pred)
            # pred = torch.cat(preds, dim=1)
            # ===== SnowflakeNet =====
            # coarse, refine1, refine2, pred = model(p)
            # ========= Ours ==========
            coarse, fusion, down_fusion, coarse_, x_, refine1, pred = model(p)
            # coarse, pred = model(p)
            # coarse, refine1, refine2, pred = model(p)
            # coarse, refine1, pred = model(p)
            # coarse, x_, refine0, refine1, pred = model(p)
            
            if params.l1cd:
                total_l1_cd += l1_cd(pred, c).item()
            if params.l2cd:
                total_l2_cd += l2_cd(pred, c).item()
            if params.emd:
                total_emd += emd1(pred, c).item()
            if params.fscore:
                total_f_score += f_score(pred[0].detach().cpu().numpy(), c[0].detach().cpu().numpy())

            if params.plot:
                # customized
                # plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(index)),
                #                   [p[0].detach().cpu().numpy(),
                #                    coarse[0].detach().cpu().numpy(),
                #                    fusion[0].detach().cpu().numpy(),
                #                    down_fusion[0].detach().cpu().numpy(),
                #                    coarse_[0].detach().cpu().numpy(),
                #                    x_[0].detach().cpu().numpy(),
                #                    refine1[0].detach().cpu().numpy(),
                #                    pred[0].detach().cpu().numpy(),
                #                    c[0].detach().cpu().numpy()],
                #                    ['Input', 'Coarse', 'Fusion', 'Downsampled Fusion', 'Denoise', 'New Input', 'Upsample1', 'Prediction', 'Ground Truth'],
                #                    xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), comment='ours')
                pass
                # plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(index)), 
                #                   [p[0].detach().cpu().numpy(),
                #                    pred[0].detach().cpu().numpy(),
                #                    c[0].detach().cpu().numpy()],
                #                    ['Input', 'Pred', 'Ground Truth'],
                #                    xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim), comment='others')
            if params.save:
                export_ply(os.path.join(output_dir, '{:03d}.ply'.format(index)), pred[0].detach().cpu().numpy())
            index += 1
    results = {}
    if params.l1cd:
        avg_l1_cd = total_l1_cd / len(test_dataset)
        results['l1_cd'] = avg_l1_cd
    if params.l2cd:
        avg_l2_cd = total_l2_cd / len(test_dataset)
        results['l2_cd'] = avg_l2_cd
    if params.emd:
        avg_emd = total_emd / len(test_dataset)
        results['emd'] = avg_emd
    if params.fscore:
        avg_f_score = total_f_score / len(test_dataset)
        results['fscore'] = avg_f_score
    # print(results)
    return results


# def test_single_category_(category, model, params):
#     """
#     Batch size not equals to 1.
#     """
#     if params.plot:
#         cat_dir = os.path.join(params.result_dir, category)
#         image_dir = os.path.join(cat_dir, 'image')
#         output_dir = os.path.join(cat_dir, 'output')
#         make_dir(cat_dir)
#         make_dir(image_dir)
#         make_dir(output_dir)

#     lim = 0.35
#     if params.dataset == 'pcn':
#         test_dataset = ShapeNet('/home/scut/hdd/liuqing/dataset/PCN', 'test', category)
#     elif params.dataset == 'pcn_novel':
#         test_dataset = ShapeNet('/home/scut/hdd/liuqing/dataset/PCN', 'test_novel', category)
#     elif params.dataset == 'c3d':
#         test_dataset = Completion3D('/home/scut/workspace/liuqing/dataset/Completion3D', 'val', category)
#         lim = 0.6
#     elif params.dataset == 'crn':
#         test_dataset = CRN('/home/scut/workspace/liuqing/dataset/CRN', 'test', category)
#         lim = 0.5
#     test_dataloader = DataLoader(test_dataset, batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)

#     index = 1
#     if params.l1cd:
#         total_l1_cd = 0.0
#     if params.l2cd:
#         total_l2_cd = 0.0
#     if params.emd:
#         total_emd = 0.0
#     if params.fscore:
#         total_f_score = 0.0
#     with torch.no_grad():
#         num_iter = 0
#         for p, c in test_dataloader:
#             p, c = p.cuda(), c.cuda()
#             num_iter += 1

#             # customized
#             # coarse, fusion, down_fusion, coarse_, x_, refine1, pred = model(p)
#             # ====== fc, foldingnet, topnet, SANet ======
#             # pred = model(p)
#             # pred = fps(pred, 2048)
#             # ====== pcn =======
#             coarse, pred = model(p)
            
#             if params.l1cd:
#                 total_l1_cd += l1_cd(pred, c).item()
#             if params.l2cd:
#                 total_l2_cd += l2_cd(pred, c).item()
#             if params.emd:
#                 total_emd += emd1(pred, c).item()
#             if params.fscore:
#                 for i in range(len(p)):
#                     total_f_score += f_score(pred[i].detach().cpu().numpy(), c[i].detach().cpu().numpy())

#             if params.plot:
#                 for i in range(len(p)):
#                     plot_pcd_one_view(os.path.join(image_dir, '{:03d}.png'.format(index)),
#                                       [p[i].detach().cpu().numpy(),
#                                        coarse[i].detach().cpu().numpy(),
#                                     #    coarse_[i].detach().cpu().numpy(),
#                                     #    refine1[i].detach().cpu().numpy(),
#                                        pred[i].detach().cpu().numpy(),
#                                        c[i].detach().cpu().numpy()],
#                                       ['Input', 'Coarse', 'Denoise', 'Refine1', 'Output', 'Ground Truth'],
#                                       xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
#                     if params.save:
#                         export_ply(os.path.join(output_dir, '{:03d}.ply'.format(index)), pred[i].detach().cpu().numpy())
#                     index += 1
    
#     results = {}
#     if params.l1cd:
#         avg_l1_cd = total_l1_cd / num_iter
#         results['l1_cd'] = avg_l1_cd
#     if params.l2cd:
#         avg_l2_cd = total_l2_cd / num_iter
#         results['l2_cd'] = avg_l2_cd
#     if params.emd:
#         avg_emd = total_emd / num_iter
#         results['emd'] = avg_emd
#     if params.fscore:
#         avg_f_score = total_f_score / len(test_dataset)
#         results['fscore'] = avg_f_score

#     return results


def test(params):
    if params.plot or params.save:
        make_dir(params.result_dir)

    # customized
    # model = Model(256, 256, ratios=[4, 8]).cuda()
    model = Model(256, 256, ratios=[4, 8]).cuda()
    # model = AblationModel(256, 256, ratios=[4, 8]).cuda()
    
    # model = FC(1024, 16384).cuda()
    # model = FC(1024, 2048).cuda()
    # model = PCN(2048, 1024, 2).cuda()
    # model = PCN().cuda()
    # model = torch.nn.DataParallel(GRNet()).cuda()
    # model = SANet().cuda()
    # model = PMPNet().cuda()
    # model = torch.nn.DataParallel(PMPNet()).cuda()
    # model = SnowflakeNet(up_factors=[4, 8]).cuda()
    # model = SnowflakeNet(up_factors=[2, 2]).cuda()
    # model = PointAttn('c3d').cuda()
    # model = Model(ratios=[2, 2]).cuda()
    # model = FBNet().cuda()
    # model = TopNet(8, 1024, 6, 2048).cuda()
    # model = torch.nn.DataParallel(GRNet(None)).cuda()
    # model = PointAttn(params.dataset).cuda()

    model.load_state_dict(torch.load(params.ckpt_path))
    # model.load_state_dict(torch.load(params.ckpt_path)['grnet'])
    # model.load_state_dict(torch.load(params.ckpt_path)['model'])
    # model.load_state_dict(torch.load(params.ckpt_path)['net_state_dict'])
    model.eval()

    format_str1 = '\033[33m{:20s}'.format('Category')
    split_line = '\033[33m{:20s}'.format('--------')
    if params.l1cd:
        format_str1 = format_str1 + '{:20s}'.format('L1_CD(1e-3)')
        split_line = split_line + '{:20s}'.format('-----------')
    if params.l2cd:
        format_str1 = format_str1 + '{:20s}'.format('L2_CD(1e-4)')
        split_line = split_line + '{:20s}'.format('-----------')
    if params.emd:
        format_str1 = format_str1 + '{:20s}'.format('EMD(1e-2)')
        split_line = split_line + '{:20s}'.format('-----------')
    if params.fscore:
        format_str1 = format_str1 + '{:20s}'.format('FScore-0.01(%)')
        split_line = split_line + '{:20s}'.format('--------------')
    
    format_str1 = format_str1 + '\033[0m'
    split_line = split_line + '\033[0m'

    print(format_str1)
    print(split_line)

    if params.category == 'all':
        categories = CATEGORIES[params.dataset]
        
        if params.l1cd:
            l1_cds = list()
        if params.l2cd:
            l2_cds = list()
        if params.emd:
            emds = list()
        if params.fscore:
            fscores = list()

        for category in categories:
            if params.batch_size == 1:
                res = test_single_category(category, model, params)
            else:
                # res = test_single_category_(category, model, params)
                pass
            print('{:20s}'.format(category.title()), end='')
            if params.l1cd:
                print('{:<20.4f}'.format(1e3 * res['l1_cd']), end='')
                l1_cds.append(res['l1_cd'])
            if params.l2cd:
                print('{:<20.4f}'.format(1e4 * res['l2_cd']), end='')
                l2_cds.append(res['l2_cd'])
            if params.emd:
                print('{:<20.4f}'.format(1e2 * res['emd']), end='')
                emds.append(res['emd'])
            if params.fscore:
                print('{:<20.4f}'.format(1e2 * res['fscore']), end='')
                fscores.append(res['fscore'])
            print()
        
        print(split_line)

        format_str2 = '\033[32m{:20s}'.format('Average')
        if params.l1cd:
            format_str2 = format_str2 + '{:<20.4f}'.format(np.mean(l1_cds) * 1e3)
        if params.l2cd:
            format_str2 = format_str2 + '{:<20.4f}'.format(np.mean(l2_cds) * 1e4)
        if params.emd:
            format_str2 = format_str2 + '{:<20.4f}'.format(np.mean(emds) * 1e2)
        if params.fscore:
            format_str2 = format_str2 + '{:<20.4f}'.format(np.mean(fscores) * 1e2)
        format_str2 = format_str2 + '\033[0m'
        print(format_str2)
    else:
        res = test_single_category(params.category, model, params)
        print('{:20s}'.format(params.category.title()), end='')
        if params.l1cd:
            print('{:<20.4f}'.format(1e3 * res['l1_cd']), end='')
        if params.l2cd:
            print('{:<20.4f}'.format(1e4 * res['l2_cd']), end='')
        if params.emd:
            print('{:<20.4f}'.format(1e2 * res['emd']), end='')
        if params.fscore:
            print('{:<20.4f}'.format(1e2 * res['fscore']), end='')
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Testing Point Cloud Completion')
    parser.add_argument('--ckpt_path', type=str, default='checkpoints\PCN_models\ckpt-best.pth', help='The path of pretrained model')
    parser.add_argument('--dataset', type=str, default='pcn', help='Dataset')
    parser.add_argument('--result_dir', type=str, default='results', help='Results directory')
    parser.add_argument('--category', type=str, default='table', help='Category of point clouds')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for data loader')
    
    parser.add_argument('--plot', default='store_true', help='Visualize by matplotlib')
    parser.add_argument('--save', action='store_true', help='Saving test result')

    parser.add_argument('--l1cd', action='store_true', help='Test L1 Chamfer Distance')
    parser.add_argument('--l2cd', action='store_true', help='Test L2 Chamfer Distance')
    parser.add_argument('--emd', action='store_true', help='Test Earth Movers Distance')
    parser.add_argument('--fscore', action='store_true', help='Test F-Score')

    params = parser.parse_args()

    test(params)
