import open3d as o3d
import numpy as np
import json
from matplotlib import cm
import cv2
import os
from PIL import Image
import struct
def load_pts(file_path):
    pts = []
    pcd = o3d.geometry.PointCloud()
    # 2. 计算单条点的字节长度
    FORMAT = '<4fHd'
    POINT_SIZE = struct.calcsize(FORMAT)  # 应该是 26
    with open(file_path, 'rb') as fin:
        idx = 0
        while True:
            data = fin.read(POINT_SIZE)
            if not data:
                break

            # 1. 定义 format string
            #    '<'   → little-endian，无额外对齐
            #    '4f'  → 4 个 float32，分别对应 x, y, z, intensity
            #    'H'   → 1 个 uint16 对应 ring
            #    'd'   → 1 个 float64 对应 timestamp
            
            x, y, z, intensity, ring, timestamp = struct.unpack(FORMAT, data)
            pcd.points.append([x,y,z])
            pcd.colors.append([intensity/255,0,0])
            idx += 1
        real_num = idx
        
        print(f"real_num = {real_num}")
    return pcd

root_path = '/data/senseauto/高速远距离数据/2025_01_10_10_03_03_AutoCollect_pilotGtRawParser'
sensor_aligment_path = root_path + "/sensor_temporal_alignment.json/000.json"
sensor_aligment = json.load(open(sensor_aligment_path))
# sensor_aligment[1]
# {'1736503211099999780': {'top_center_lidar': 'lidar/top_center_lidar/1736503211099999780.pcd',
#   'center_camera_fov120': 'camera/center_camera_fov120/1736503211099999810.jpg',
#   'center_camera_fov30': 'camera/center_camera_fov30/1736503211099999810.jpg',
#   'left_front_camera': 'camera/left_front_camera/1736503211099999810.jpg',
#   'left_rear_camera': 'camera/left_rear_camera/1736503211099999810.jpg',
#   'rear_camera': 'camera/rear_camera/1736503211099999810.jpg',
#   'right_front_camera': 'camera/right_front_camera/1736503211099999810.jpg',
#   'right_rear_camera': 'camera/right_rear_camera/1736503211099999810.jpg',
#   'front_camera_fov195': 'camera/front_camera_fov195/1736503211099999645.jpg',
#   'left_camera_fov195': 'camera/left_camera_fov195/1736503211099999645.jpg',
#   'rear_camera_fov195': 'camera/rear_camera_fov195/1736503211099999645.jpg',
#   'right_camera_fov195': 'camera/right_camera_fov195/1736503211099999645.jpg',
#   'front_left_radar': 'radar/front_left_radar/1736503211088439552.json',
#   'front_radar': 'radar/front_radar/1736503211126470144.json',
#   'front_right_radar': 'radar/front_right_radar/1736503211109603328.json',
#   'rear_left_radar': 'radar/rear_left_radar/1736503211110062080.json',
#   'rear_right_radar': 'radar/rear_right_radar/1736503211088619264.json'}}
pcd_path = root_path + sensor_aligment['1736503211099999780']['top_center_lidar']
# pcd_path = root_path + "/lidar/top_center_lidar/1725782003299999850.pcd"
camera_path = root_path + sensor_aligment['1736503211099999780']['center_camera_fov30']
intrinsic_path = root_path + "/calib/center_camera_fov30/center_camera_fov30-intrinsic.json"
extrinsic_path = root_path + "/calib/center_camera_fov30/center_camera_fov30-to-car_center-extrinsic.json"
# image_filled_path = root_path + "/camera/center_camera_fov30/1725782003299999850_filled.jpg"
image_filled_path = root_path + "/camera/center_camera_fov30_proj/1736503211099999780_filled_intensity.jpg"

def proj_lidar(pcd_path, camera_path, intrinsic_path, extrinsic_path, image_filled_path):
    pcd = o3d.io.read_point_cloud(pcd_path)

    points = np.asarray(pcd.points)

# pcd_intensity = load_pts(pcd_path)
# 二进制读取 pcd_path
# 先定义一个对应该点格式的 dtype 
    dtype = np.dtype([
    ('x', 'float32'),
    ('y', 'float32'),
    ('z', 'float32'),
    ('intensity', 'float32'),
    ('ring', 'uint16'),
    # 这里用 '<f8' 明确指定 little-endian 的 float64
    ('timestamp', '<f8')
])
    with open(pcd_path, 'rb') as fin:
        data = fin.read()
        pts_points = np.frombuffer(data, dtype=dtype)
        points_intensity = pts_points['intensity'].copy()
        points_intensity[points_intensity>1e3] = 1e3
        points_intensity[points_intensity<-1e3] = -1e3
    # Assuming you have points_intensity from your existing code
        min_intensity = float('inf')
        max_intensity = float('-inf')
        for intensity in points_intensity:
            if intensity < min_intensity:
                min_intensity = intensity
            if intensity > max_intensity:
                max_intensity = intensity
        print(f"min_intensity = {min_intensity}, max_intensity = {max_intensity}")
    # Normalize intensity values
        invalid_mask = np.isnan(points_intensity)
        points_intensity[invalid_mask] = 1
        gap = max_intensity - min_intensity+1e-6
        if np.isinf(gap):
            gap = 1e-6
        normalized = (points_intensity - min_intensity) / gap
        normalized = np.clip(normalized, 0.0, 1.0)
        import ipdb;ipdb.set_trace()
        pcd_intensity_colors = np.zeros((points.shape[0], 3))
        pcd_intensity_colors[:, 0] = normalized
        pcd_intensity_colors[:, 1] = 255
        pcd_intensity_colors[:, 2] = 255
    # colors = pcd_intensity_colors
        colors = cm.jet(normalized)
# colors = np.asarray(pcd_intensity.colors)
# import ipdb;ipdb.set_trace()

    ply_path=pcd_path.replace('.pcd', '.ply')
# o3d.io.write_point_cloud(ply_path, pcd)



# o3d.visualization.draw_geometries([pcd])



    image0 = cv2.imread(camera_path)
    image0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)

#load json file
    intrinsic = json.load(open(intrinsic_path))
    extrinsic = json.load(open(extrinsic_path))
    intrinsic_np = np.array(intrinsic['value0']['param']['cam_K_new']['data'])
    extrinsic_np = np.array(extrinsic['center_camera_fov30-to-car_center-extrinsic']['param']['sensor_calib']['data'])
    camera2car=extrinsic_np



# import ipdb;ipdb.set_trace()
# lidar2camera=lidar2car@np.linalg.inv(camera2car)
    lidar2camera=np.linalg.inv(camera2car)

    K=intrinsic_np
    points_h = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)
    points_camera_fov30 = (lidar2camera @ points_h.T).T
# points_camera_fov30 = (points_h.T).T
    points_camera_fov30 = points_camera_fov30[:, :3]/points_camera_fov30[:, 3:4]



    pcd_camera_fov30 = o3d.geometry.PointCloud()
    pcd_camera_fov30.points = o3d.utility.Vector3dVector(points_camera_fov30)
    pcd_camera_fov30.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(ply_path.replace('.ply', '_camera_fov30.ply'), pcd_camera_fov30)
    u = K[0, 0] * points_camera_fov30[:, 0] / points_camera_fov30[:, 2] + K[0, 2]
    v = K[1, 1] * points_camera_fov30[:, 1] / points_camera_fov30[:, 2] + K[1, 2]
    depth = points_camera_fov30[:, 2]
# depth_Image = Image.fromarray(depth.reshape(1080, 1920).astype(np.uint8))
# import ipdb;ipdb.set_trace()
# depth_Image.save(root_path + "/camera/center_camera_fov30/1725782003299999850_depth.jpg")
    min_depth=0
    max_depth=1000
    mask = np.zeros_like(depth.shape[0])
    mask = np.logical_and(depth > min_depth, depth < max_depth)
# min_depth = depth[depth>0.5].min()
# max_depth = depth[depth>0.5].max()
# 归一化深度值
    min_normalized_depth = depth[mask].min()
    max_normalized_depth = depth[mask].max()
    normalized_depth = (depth - min_normalized_depth) / (max_normalized_depth - min_normalized_depth)
# normalized_depth = (depth - min_depth) / (max_depth - min_depth)

# 使用 jet 颜色映射
# color = cm.jet(normalized_depth)
    color = cm.jet(normalized)

    u = u.astype(int)
    v = v.astype(int)

    rgb_colors=[]
#将点云圆点画粗点

    for x,y,d in zip(u,v,depth):
        if 0 <= x < image0.shape[1] and 0 <= y < image0.shape[0] and min_depth<d<max_depth:
        # rgb_colors.append(image0[y, x])
            i = y*image0.shape[1]+x
            rgb_colors.append(colors[i])

        else:
            rgb_colors.append([0, 0, 0])  # 如果投影点超出图像边界，则赋予黑色
    image_filled = image0.copy()
    circle_radius=8
# Check and fill the pixels at the (u, v) coordinates with red color
    for i in range(len(u)):
        if 0 <= v[i] < image0.shape[0] and 0 <= u[i] < image0.shape[1] and min_depth<depth[i]<max_depth:
        # image_filled[v[i], u[i]] = color[i,:3]*255
            cv2.circle(image_filled, (u[i],v[i]), circle_radius, color[i,:3]*255, -1) 




# 获取父文件夹路径
    parent_dir = os.path.dirname(image_filled_path)

# 检查父文件夹是否存在
    if not os.path.exists(parent_dir):
    # 如果不存在，则创建父文件夹
        os.makedirs(parent_dir)
# 打开图片
    image = Image.fromarray(image_filled)
# 保存图片
    image.save(image_filled_path)

proj_lidar(pcd_path, camera_path, intrinsic_path, extrinsic_path, image_filled_path)








