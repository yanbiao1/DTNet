import argparse
import os
import datetime
import random

import torch
import torch.optim as Optim

from torch.utils.data.dataloader import DataLoader
from tensorboardX import SummaryWriter

from dataset import ShapeNet_v3 as ShapeNet, Completion3D, CRN
from models import FC, FoldingNet, PCN, TopNet, GRNet, SANet, PMPNet, SnowflakeNet, PointAttn,  Model
from models.utils import fps
from visualization import plot_pcd_one_view
from loss import l2_cd, emd2, l2_dcd, l1_cd


def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def log(fd,  message):
    message = ' ==> '.join([datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), message])
    fd.write(message + '\n')
    fd.flush()
    print(message)


def prepare_logger(params):
    # prepare logger directory
    make_dir(params.log_dir)
    make_dir(os.path.join(params.log_dir, params.exp_name))

    logger_path = os.path.join(params.log_dir, params.exp_name, params.category)
    ckpt_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'checkpoints')
    epochs_dir = os.path.join(params.log_dir, params.exp_name, params.category, 'epochs')

    make_dir(logger_path)
    make_dir(ckpt_dir)
    make_dir(epochs_dir)

    logger_file = os.path.join(params.log_dir, params.exp_name, params.category, 'logger.log')
    log_fd = open(logger_file, 'a')

    log(log_fd, "Experiment: {}".format(params.exp_name))
    log(log_fd, "Logger directory: {}".format(logger_path))
    log(log_fd, str(params))

    train_writer = SummaryWriter(os.path.join(logger_path, 'train'))
    val_writer = SummaryWriter(os.path.join(logger_path, 'val'))

    return ckpt_dir, epochs_dir, log_fd, train_writer, val_writer


def train(params):
    torch.backends.cudnn.benchmark = True

    ckpt_dir, epochs_dir, log_fd, train_writer, val_writer = prepare_logger(params)

    log(log_fd, 'Loading Data...')
    train_dataset    = Completion3D('/home/scut/workspace/liuqing/dataset/Completion3D', 'train', 'all', aug=True, scale=True)
    test_dataset     = CRN('/home/scut/workspace/liuqing/dataset/CRN', 'test',  'all')
    val_dataset      = Completion3D('/home/scut/workspace/liuqing/dataset/Completion3D', 'val', 'all')
    train_dataloader = DataLoader(train_dataset, batch_size=params.batch_size, shuffle=True,  num_workers=params.num_workers)
    test_dataloader  = DataLoader(test_dataset,  batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    val_dataloader   = DataLoader(val_dataset,   batch_size=params.batch_size, shuffle=False, num_workers=params.num_workers)
    log(log_fd, "Dataset loaded!")

    # model = SnowflakeNet(up_factors=[4, 8]).cuda()
    # model = PointAttn('pcn').cuda()
    # model = FC(1024, 2048).cuda()
    # model = FoldingNet(512, 2025).cuda()
    # model = PCN(2048, 1024, 2).cuda()
    # model = TopNet(8, 1024, 6, 2048).cuda()
    # model = GRNet().cuda()
    # model = SANet().cuda()
    # model = PMPNet().cuda()
    # model = PointAttn(dataset='c3d').cuda()
    # model = SnowflakeNet(up_factors=[2, 2]).cuda()
    # model = Model(ratios=[1, 4]).cuda()
    model = Model(ratios=[2, 2]).cuda()
    

    # optimizer
    optimizer = Optim.Adam(model.parameters(), lr=params.lr)
    lr_schedual = Optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)

    step = len(train_dataloader) // params.log_frequency

    best_test_l2_cd = 1e8
    best_test_l2_epoch = -1
    best_val_l2_cd = 1e8
    best_val_l2_epoch = -1
    train_step, test_step = 0, 0

    # load pretrained model and optimizer
    if params.ckpt_path is not None:
        model.load_state_dict(torch.load(params.ckpt_path))
    
        for i in range(1, params.start_epoch):
            lr_schedual.step()

    for epoch in range(params.start_epoch, params.epochs + 1):
        # training
        model.train()

        for i, (in_cloud, gt_cloud) in enumerate(train_dataloader):
            in_cloud, gt_cloud = in_cloud.cuda(), gt_cloud.cuda()

            coarse, fusion, down_fusion, denoise, new_in, refine1, pred = model(in_cloud)

            gt1024 = fps(gt_cloud, 1024)
            gt512 = fps(gt1024, 512)
            gt256 = fps(gt512, 256)

            cdc = l1_cd(coarse,  gt256)
            cdd = l1_cd(denoise, gt256)
            cd1 = l1_cd(refine1, gt1024)
            cd2 = l1_cd(pred,    gt_cloud)

            loss = cdc + cdd + cd1 + cd2

            # back propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % step == 0:
                log(log_fd,
                    "Training Epoch [{:03d}/{:03d}] - Iteration [{:04d}/{:04d}]: "
                    "cdc = {:.6f}, cdd = {:.6f}, cd1 = {:.6f}, cd2 = {:.6f}, loss = {:.6f}".format(
                        epoch, params.epochs, i + 1, len(train_dataloader),
                        cdc.item(), cdd.item(), cd1.item(), cd2.item(), loss.item()
                        )
                    )
            
            train_step += 1
        
        lr_schedual.step()

        # evaluation
        model.eval()

        total_test_l2_cd = 0.0
        total_val_l2_cd = 0.0

        with torch.no_grad():
            # ========================================== Test Dataset ======================================
            rand_iter = random.randint(0, len(test_dataloader) - 1)  # for visualization

            for i, (in_cloud, gt_cloud) in enumerate(test_dataloader):
                in_cloud, gt_cloud = in_cloud.cuda(), gt_cloud.cuda()

                coarse, fusion, down_fusion, denoise, new_in, refine1, pred = model(in_cloud)
                total_test_l2_cd += l2_cd(pred, gt_cloud).item()

                # save into image
                # if rand_iter == i:
                #     plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                #                       [in_cloud[0].detach().cpu().numpy(),
                #                        fusion[0].detach().cpu().numpy(),
                #                        refine1[0].detach().cpu().numpy(),
                #                        refine2[0].detach().cpu().numpy(),
                #                        pred[0].detach().cpu().numpy(),
                #                        gt_cloud[0].detach().cpu().numpy()],
                #                       ['Partial', 'Coarse', 'Fusion', 'Refine1', 'Refine2', 'Pred', 'GT'],
                #                       cmap='jet', xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35))
            
            total_test_l2_cd /= len(test_dataloader)
            test_step += 1

            log(log_fd, "Test Epoch [{:03d}/{:03d}] - L2 CD = {:.6f}".format(epoch, params.epochs, total_test_l2_cd * 1e4))
        
            if total_test_l2_cd < best_test_l2_cd:
                best_test_l2_epoch = epoch
                best_test_l2_cd = total_test_l2_cd
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_test_l2_cd.pth'))
            # ===============================================================================================

            # ============================================ Val dataset ======================================
            rand_iter = random.randint(0, len(val_dataloader) - 1)  # for visualization
            for i, (in_cloud, gt_cloud) in enumerate(val_dataloader):
                in_cloud, gt_cloud = in_cloud.cuda(), gt_cloud.cuda()

                coarse, fusion, down_fusion, denoise, new_in, refine1, pred = model(in_cloud)
                total_val_l2_cd += l2_cd(pred, gt_cloud).item()

                # save into image
                if rand_iter == i:
                    plot_pcd_one_view(os.path.join(epochs_dir, 'epoch_{:03d}.png'.format(epoch)),
                                        [in_cloud[0].detach().cpu().numpy(),
                                         coarse[0].detach().cpu().numpy(),
                                         fusion[0].detach().cpu().numpy(),
                                         down_fusion[0].detach().cpu().numpy(),
                                         denoise[0].detach().cpu().numpy(),
                                         new_in[0].detach().cpu().numpy(),
                                         refine1[0].detach().cpu().numpy(),
                                         pred[0].detach().cpu().numpy(),
                                         gt_cloud[0].detach().cpu().numpy()],
                                        ['Partial', 'Coarse', 'Fusion', 'Down Fusion', 'Denoise', 'New In', 'Refine1', 'Pred', 'GT'],
                                        cmap='jet',
                                        xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), zlim=(-0.6, 0.6))
            
            total_val_l2_cd /= len(val_dataloader)

            log(log_fd, "Val  Epoch [{:03d}/{:03d}] - L2 CD = {:.6f}".format(epoch, params.epochs, total_val_l2_cd * 1e4))
        
            if total_val_l2_cd < best_val_l2_cd:
                best_val_l2_epoch = epoch
                best_val_l2_cd = total_val_l2_cd
                torch.save(model.state_dict(), os.path.join(ckpt_dir, 'best_val_l2_cd.pth'))
            # ===============================================================================================
        
        torch.save(model.state_dict(), os.path.join(ckpt_dir, 'current.pth'))
            
    log(log_fd, 'Best test l2 cd model in epoch {}, the minimum l1 cd is {:.6f}'.format(best_test_l2_epoch, best_test_l2_cd * 1e4))
    log(log_fd, 'Best val  l2 cd model in epoch {}, the minimum l2 cd is {:.6f}'.format(best_val_l2_epoch,  best_val_l2_cd * 1e4))
    log_fd.close()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Point Cloud Completion')

    parser.add_argument('--exp_name', type=str, default='C3D-PCC')
    parser.add_argument('--log_dir', type=str, default='log/c3d_', help='Logger directory')
    parser.add_argument('--ckpt_path', type=str, default=None, help='The path of pretrained model')
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--category', type=str, default='all', help='Category of point clouds')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs of training')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loader')
    parser.add_argument('--num_workers', type=int, default=8, help='num_workers for data loader')
    parser.add_argument('--device', type=str, default='cuda:0', help='device for training')
    parser.add_argument('--log_frequency', type=int, default=10, help='Logger frequency in every epoch')
    parser.add_argument('--save_frequency', type=int, default=10, help='Model saving frequency')

    params = parser.parse_args()
    
    train(params)
