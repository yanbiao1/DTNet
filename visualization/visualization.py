import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def o3d_visualize_pc(pc):
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    o3d.visualization.draw_geometries([point_cloud])


def o3d_visualize_pcs(pcs: list):
    lst = list()
    for pc in pcs:
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(pc)
        lst.append(point_cloud)
    o3d.visualization.draw_geometries(lst)


def plot_pcd_one_view(filename, pcds, titles, suptitle='', sizes=None, cmap='jet', zdir='y',
                         xlim=(-0.35, 0.35), ylim=(-0.35, 0.35), zlim=(-0.35, 0.35), comment=None):
    if sizes is None:
        sizes = [0.05 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3 * 1.4, 3 * 1.4))
    elev = 30  # 水平倾斜
    azim = -45  # 旋转
    for j, (pcd, size) in enumerate(zip(pcds, sizes)):
        if comment is None:
            color = pcd[:, 0]
        elif comment == 'partial':
            color = '#4e89c7'  # input
        elif comment == 'others':
            color = '#fc9390'  # other methods
        elif comment == 'ours':
            color = '#ff6347'  # ours
        elif comment == 'gt':
            color = 'orange'  # gt
        ax = fig.add_subplot(1, len(pcds), j + 1, projection='3d')
        ax.view_init(elev, azim)
        ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-1.0, vmax=0.5)
        ax.set_title(titles[j])
        ax.set_axis_off()
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)


def plot_pcd_three_views(filename, pcds, titles, suptitle='', sizes=None, cmap='Reds', zdir='y',
                         xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.5, 0.5)):
    if sizes is None:
        sizes = [0.5 for i in range(len(pcds))]
    fig = plt.figure(figsize=(len(pcds) * 3, 9))
    for i in range(3):
        elev = 30  # 水平倾斜
        azim = -45 + 90 * i  # 旋转
        # azim = -60
        for j, (pcd, size) in enumerate(zip(pcds, sizes)):
            color = pcd[:, 0]
            ax = fig.add_subplot(3, len(pcds), i * len(pcds) + j + 1, projection='3d')
            ax.view_init(elev, azim)
            ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], zdir=zdir, c=color, s=size, cmap=cmap, vmin=-0.6, vmax=0.5)
            ax.set_title(titles[j])
            ax.set_axis_off()
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_zlim(zlim)
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.9, wspace=0.1, hspace=0.1)
    plt.suptitle(suptitle)
    fig.savefig(filename)
    plt.close(fig)
