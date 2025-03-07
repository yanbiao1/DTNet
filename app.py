import os
import random
import torch
import open3d as o3d
import numpy as np

from flask import Flask, render_template, request, send_file, send_from_directory
from datetime import datetime
from pathlib import Path

from models import Model
from visualization import plot_pcd_one_view


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploaded/'


model = Model(num_pc=256, num_down=256, ratios=[4, 8]).cuda()
model.load_state_dict(torch.load('checkpoints/pretrained.pth'))
model.eval()


def read_point_cloud(filename):
    suffix = Path(filename).suffix
    if suffix in ['.ply', '.pcd', '.obj']:
        pc = o3d.io.read_point_cloud(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        points = np.array(pc.points)
    elif suffix == '.npy':
        points = np.load(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    else:
        raise Exception('不支持的文件格式')
    return np.asarray(points, dtype=np.float32)


def write_point_cloud(filename, points):
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud('outputs/{}'.format(filename), pc)


@app.route('/pcc')
def pcc():
    return render_template('pcc.html')


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    """
    获取客户端提交的文件并补全点云
    """
    if request.method == 'POST':
        # 获取文件
        file = request.files['file']
        print(file)
        filename = 'pc{}_{:04d}{}'.format(int(datetime.now().timestamp()),
                                              random.randint(0, 1024), Path(file.filename).suffix)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        file.close()

        # 点云补全
        points = read_point_cloud(filename)
        p = torch.from_numpy(points).cuda().unsqueeze(0)
        with torch.no_grad():
            coarse, _, _, denoiser, _, refine1, pred = model(p)
            pred = pred.squeeze(0).detach().cpu().numpy()

        output_filename = '{}.ply'.format(os.path.basename(filename))
        image_path = '{}.png'.format(os.path.basename(filename))

        write_point_cloud(output_filename, pred)
        plot_pcd_one_view('static/{}'.format(image_path), [points, pred], ['Input', 'Output'])
        
        return render_template('result.html', image_path=image_path, output_filename=output_filename)
    else:
        return render_template('pcc.html')


@app.route('/download/<filename>')
def download(filename):
    """下载.ply文件"""
    return send_from_directory('./outputs/', filename, as_attachment=True)


if __name__ == '__main__':
    app.run()
