import argparse
import json

import imageio
import numpy as np
import open3d as o3d
import torch
from PIL import Image
from sklearn.neighbors import KNeighborsRegressor

parser = argparse.ArgumentParser(description="KNN Depth Completion")
parser.add_argument(
    "--root_path", type=str, required=True, help="Root path to the data"
)
args = parser.parse_args()

root_path = args.root_path
sensor_aligment_path = root_path + "/sensor_temporal_alignment.json/000.json"
sensor_aligment = json.load(open(sensor_aligment_path))

for alignment in sensor_aligment:
    for timestamp, data in alignment.items():
        ply_path = root_path + data["top_center_lidar"].replace(
            ".pcd", "_camera_fov30.ply"
        )
        intrinsic_path = (
            root_path + "/calib/center_camera_fov30/center_camera_fov30-intrinsic.json"
        )

        pcd = o3d.io.read_point_cloud(ply_path)
        points = np.asarray(pcd.points)
        intrinsic = json.load(open(intrinsic_path))
        intrinsic_np = np.array(intrinsic["value0"]["param"]["cam_K_new"]["data"])

        h, w = (
            int(intrinsic["value0"]["param"]["img_dist_h"]),
            int(intrinsic["value0"]["param"]["img_dist_w"]),
        )
        K = intrinsic_np
        cx = K[0, 2]
        cy = K[1, 2]
        fx = K[0, 0]
        fy = K[1, 1]

        print(
            f"Processing timestamp: {timestamp}, Depth range: {points[:, 2].min()} - {points[:, 2].max()}"
        )

        # 创建深度图像数组
        pts_depth = np.zeros([1, h, w])
        point_camera = points
        uvz = point_camera[point_camera[:, 2] > 0]
        uvz = uvz @ K.T
        uvz[:, :2] /= uvz[:, 2:]
        uvz = uvz[uvz[:, 1] >= 0]
        uvz = uvz[uvz[:, 1] < h]
        uvz = uvz[uvz[:, 0] >= 0]
        uvz = uvz[uvz[:, 0] < w]
        uv = uvz[:, :2].astype(int)
        pts_depth[0, uv[:, 1], uv[:, 0]] = uvz[:, 2]

        pts_depth_uint16 = pts_depth[0] * 1000
        pts_depth_uint16[pts_depth_uint16 > 65535] = 65535
        pts_depth_uint16 = pts_depth_uint16.astype(np.uint16)
        Image.fromarray(pts_depth_uint16).save(f"pts_depth_origin_{timestamp}.png")
        pts_depth = torch.from_numpy(pts_depth).float()

        valid_points = []
        valid_depths = []
        for i in range(h):
            for j in range(w):
                if pts_depth[0, i, j] > 0:
                    valid_points.append([i, j])
                    valid_depths.append(pts_depth[0, i, j])

        valid_points = np.array(valid_points)
        valid_depths = np.array(valid_depths)

        if len(valid_points) > 0:
            knn = KNeighborsRegressor(n_neighbors=4, weights="distance")
            knn.fit(valid_points, valid_depths)

            invalid_points = []
            for i in range(h):
                for j in range(w):
                    if pts_depth[0, i, j] == 0:
                        invalid_points.append([i, j])

            invalid_points = np.array(invalid_points)

            if len(invalid_points) > 0:
                pred_depths = knn.predict(invalid_points)
                for idx, (i, j) in enumerate(invalid_points):
                    pts_depth[0, i, j] = pred_depths[idx]

        pts_depth = pts_depth.numpy()
        depth_int = pts_depth[0] * 1000
        depth_int[depth_int > 65535] = 65535
        depth_int = depth_int.astype(np.uint16)
        Image.fromarray(depth_int).save(f"pts_depth_knn_{timestamp}.png")
        np.savez(f"pts_depth_{timestamp}.npz", depth=pts_depth[0])
