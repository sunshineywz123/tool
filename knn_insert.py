import argparse
import imageio
import numpy as np
import torch
from torchvision.utils import save_image
from sklearn.neighbors import KNeighborsRegressor
from PIL import Image
import open3d as o3d
import json
import tqdm
import os
import re

# 使用 argparse 读取 root_path
parser = argparse.ArgumentParser(description="Process point cloud and depth data.")
parser.add_argument('--root_path', type=str, required=True, help='Root path to the data directory.')
args = parser.parse_args()



root_path = args.root_path
sensor_aligment_path = root_path + "/sensor_temporal_alignment_json/000.json"
sensor_aligment = json.load(open(sensor_aligment_path))
intrinsic_path = root_path + '/calib/center_camera_fov30/center_camera_fov30-intrinsic.json'
lidar_dir = os.path.join(root_path, "lidar", "top_center_lidar")
ply_files = [os.path.join(lidar_dir, f) for f in os.listdir(lidar_dir) if f.endswith('.ply')]
output_dir = os.path.join(root_path, "lidar", "top_center_lidar_depth")

for ply in ply_files:

    file_name = os.path.basename(ply)
    mmumeric_part=file_name[:19]
    
    pcd= o3d.io.read_point_cloud(ply)
    points = np.asarray(pcd.points)

    # 加载相机内参 json 文件
    intrinsic = json.load(open(intrinsic_path))
    # 获取相机内参矩阵
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])

    # 获取图像高度和宽度
    h, w = int(intrinsic['value0']['param']['img_dist_h']), int(intrinsic['value0']['param']['img_dist_w'])
    # 将内参矩阵赋值给 K
    K = intrinsic_np
    cx = K[0, 2]
    cy = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # 打印点云深度范围
    print("Depth range: ", points[:, 2].min(), points[:, 2].max())

    # 创建深度图像数组
    pts_depth = np.zeros([1, h, w])
    # 将点云数据赋值给 point_camera
    point_camera = points
    # 筛选出深度值大于 0 的点
    uvz = point_camera[point_camera[:, 2] > 0]
    # 将点云坐标转换到图像坐标系
    uvz = uvz @ K.T
    # 进行透视除法
    uvz[:, :2] /= uvz[:, 2:]
    # 筛选出在图像范围内的点
    uvz = uvz[uvz[:, 1] >= 0]
    uvz = uvz[uvz[:, 1] < h]
    uvz = uvz[uvz[:, 0] >= 0]
    uvz = uvz[uvz[:, 0] < w]
    # 获取图像坐标
    uv = uvz[:, :2]
    # 将坐标转换为整数
    uv = uv.astype(int)
    # 将深度值填充到深度图中
    pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]

    pts_depth_uint16 = (pts_depth[0] * 1000)
    pts_depth_uint16[pts_depth_uint16 > 65535] = 65535
    pts_depth_uint16 = pts_depth_uint16.astype(np.uint16)
    Image.fromarray(pts_depth_uint16).save('pts_depth_origin.png')
    # 将深度图转换为 torch 张量
    pts_depth = torch.from_numpy(pts_depth).float()

    # 使用 KNN 进行深度补全, K=4
    # 找到所有有深度值的点
    valid_points = []
    valid_depths = []
    # 遍历图像所有像素
    for i in range(h):
        for j in range(w):
            # 如果该点有深度值
            if pts_depth[0, i, j] > 0:
                # 记录该点的坐标和深度值
                valid_points.append([i, j])
                valid_depths.append(pts_depth[0, i, j])

    # 将列表转换为 numpy 数组
    valid_points = np.array(valid_points)
    valid_depths = np.array(valid_depths)

    # 如果存在有效点
    if len(valid_points) > 0:
        # 创建并训练 KNN 模型
        knn = KNeighborsRegressor(n_neighbors=4, weights='distance')
        knn.fit(valid_points, valid_depths)

        # 找到所有没有深度值的点
        invalid_points = []
        for i in range(h):
            for j in range(w):
                if pts_depth[0, i, j] == 0:
                    invalid_points.append([i, j])

        # 将列表转换为 numpy 数组
        invalid_points = np.array(invalid_points)

        # 如果存在无效点
        if len(invalid_points) > 0:
            # 使用 KNN 预测深度值
            pred_depths = knn.predict(invalid_points)

            # 将预测的深度值填充到深度图中
            for idx, (i, j) in enumerate(invalid_points):
                pts_depth[0, i, j] = pred_depths[idx]

    pts_depth = pts_depth.numpy()
    depth_int = (pts_depth[0] * 1000)
    depth_int[depth_int > 65535] = 65535
    depth_int = depth_int.astype(np.uint16)
    print(depth_int.shape)
    print(depth_int.min(), depth_int.max())
    Image.fromarray(depth_int).save(os.path.join(output_dir, f'{mmumeric_part}.png'))
    print(pts_depth.shape)
    print(pts_depth.min(), pts_depth.max())
    np.savez(os.path.join(output_dir, f'{mmumeric_part}.npz'), depth=pts_depth[0])