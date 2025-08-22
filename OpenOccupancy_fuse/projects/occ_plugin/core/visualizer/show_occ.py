
# import torch.nn.functional as F
# import torch
# import numpy as np
# from os import path as osp
# import os

# def save_occ(pred_c, pred_f, img_metas, path, visible_mask=None, gt_occ=None, free_id=0, thres_low=0.4, thres_high=0.99):

#     """
#     visualization saving for paper:
#     1. gt
#     2. pred_f pred_c
#     3. gt visible
#     4. pred_f visible
#     """
#     pred_f = F.softmax(pred_f, dim=1)
#     pred_f = pred_f[0].cpu().numpy()  # C W H D
#     pred_c = F.softmax(pred_c, dim=1)
#     pred_c = pred_c[0].cpu().numpy()  # C W H D
#     # visible_mask = visible_mask[0].cpu().numpy().reshape(-1) > 0  # WHD
#     gt_occ = gt_occ.data[0][0].cpu().numpy()  # W H D
#     gt_occ[gt_occ==255] = 0
#     _, W, H, D = pred_f.shape
#     coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
#     _, W, H, D = pred_c.shape
#     coordinates_3D_c = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3) # (W*H*D, 3)
#     pred_f = np.argmax(pred_f, axis=0) # (W, H, D)
#     pred_c = np.argmax(pred_c, axis=0) # (W, H, D)
#     occ_pred_f_mask = (pred_f.reshape(-1))!=free_id
#     occ_pred_c_mask = (pred_c.reshape(-1))!=free_id
#     occ_gt_mask = (gt_occ.reshape(-1))!=free_id
#     pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
#     pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
#     # pred_f_visible_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask&visible_mask], pred_f.reshape(-1)[occ_pred_f_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
#     gt_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
#     # gt_visible_save = np.concatenate([coordinates_3D_f[occ_gt_mask&visible_mask], gt_occ.reshape(-1)[occ_gt_mask&visible_mask].reshape(-1, 1)], axis=1)[:, [2,1,0,3]]  # zyx cls
    
#     scene_token = img_metas.data[0][0]['scene_token']
#     lidar_token = img_metas.data[0][0]['lidar_token']
#     save_path = osp.join(path, scene_token, lidar_token)
#     if not osp.exists(save_path):
#         os.makedirs(save_path)
#     save_pred_f_path = osp.join(save_path, 'pred_f.npy')
#     save_pred_c_path = osp.join(save_path, 'pred_c.npy')
#     # save_pred_f_v_path = osp.join(save_path, 'pred_f_visible.npy')
#     save_gt_path = osp.join(save_path, 'gt.npy')
#     # save_gt_v_path = osp.join(save_path, 'gt_visible.npy')
#     np.save(save_pred_f_path, pred_f_save)
#     np.save(save_pred_c_path, pred_c_save)
#     # np.save(save_pred_f_v_path, pred_f_visible_save)
#     np.save(save_gt_path, gt_save)
#     # np.save(save_gt_v_path, gt_visible_save)



import os
import json
import numpy as np
import matplotlib.pyplot as plt
from plyfile import PlyData
import torch.nn.functional as F
import torch

# Function to save occupancy data (including LIDAR point cloud and predictions)
def save_occ(pred_c, pred_f, img_metas, path, visible_mask=None, gt_occ=None, free_id=0, thres_low=0.4, thres_high=0.99):
    """
    Save the occupancy prediction and ground truth for visualization and analysis.
    Includes:
    1. Ground truth (gt_occ)
    2. Predictions (pred_f, pred_c)
    3. Visualize and save LIDAR data
    """
    # Process predictions and ground truth
    pred_f = F.softmax(pred_f, dim=1).cpu().numpy()  # C W H D
    pred_c = F.softmax(pred_c, dim=1).cpu().numpy()  # C W H D

    gt_occ = gt_occ.data[0][0].cpu().numpy()  # W H D
    gt_occ[gt_occ == 255] = 0  # Remove invalid GT values
    
    # Generate 3D coordinates for prediction
    _, W, H, D = pred_f.shape
    coordinates_3D_f = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3)  # (W*H*D, 3)
    
    _, W, H, D = pred_c.shape
    coordinates_3D_c = np.stack(np.meshgrid(np.arange(W), np.arange(H), np.arange(D), indexing='ij'), axis=-1).reshape(-1, 3)  # (W*H*D, 3)
    
    # Convert predictions to classes
    pred_f = np.argmax(pred_f, axis=0)  # (W, H, D)
    pred_c = np.argmax(pred_c, axis=0)  # (W, H, D)

    # Create masks for filtering out 'free' areas (where free_id exists)
    occ_pred_f_mask = (pred_f.reshape(-1)) != free_id
    occ_pred_c_mask = (pred_c.reshape(-1)) != free_id
    occ_gt_mask = (gt_occ.reshape(-1)) != free_id

    # Save prediction and ground truth data
    pred_f_save = np.concatenate([coordinates_3D_f[occ_pred_f_mask], pred_f.reshape(-1)[occ_pred_f_mask].reshape(-1, 1)], axis=1)[:, [2, 1, 0, 3]]  # zyx cls
    pred_c_save = np.concatenate([coordinates_3D_c[occ_pred_c_mask], pred_c.reshape(-1)[occ_pred_c_mask].reshape(-1, 1)], axis=1)[:, [2, 1, 0, 3]]  # zyx cls
    gt_save = np.concatenate([coordinates_3D_f[occ_gt_mask], gt_occ.reshape(-1)[occ_gt_mask].reshape(-1, 1)], axis=1)[:, [2, 1, 0, 3]]  # zyx cls

    # Extract metadata for scene and lidar tokens
    scene_token = img_metas['data'][0][0]['scene_token']
    lidar_token = img_metas['data'][0][0]['lidar_token']
    
    save_path = os.path.join(path, scene_token, lidar_token)
    os.makedirs(save_path, exist_ok=True)
    
    # Save the predictions and GT
    np.save(os.path.join(save_path, 'pred_f.npy'), pred_f_save)
    np.save(os.path.join(save_path, 'pred_c.npy'), pred_c_save)
    np.save(os.path.join(save_path, 'gt.npy'), gt_save)

    # Now load and visualize the LIDAR data
    lidar_data = img_metas['data'][0][0]['lidar_data']
    process_and_visualize_lidar_data(lidar_data, save_path)

# Function to load and process LIDAR data
def load_data(data_list):
    """
    Load and process a list of data entries (in this case, sensor data from lidar and camera).
    """
    lidar_data = []
    for entry in data_list:
        lidar_entry = {
            'lidar_path': entry['lidar_path'],
            'token': entry['token'],
            'can_bus': entry['can_bus'],
            'frame_idx': entry['frame_idx'],
            'camera_paths': {camera_type: camera_info['data_path'] for camera_type, camera_info in entry['cams'].items()}
        }
        lidar_data.append(lidar_entry)
    return lidar_data

# Function to read LIDAR data from PCD file
def read_lidar_data(lidar_path):
    """
    Read and parse the LIDAR data from a PCD file.
    """
    if not os.path.exists(lidar_path):
        raise FileNotFoundError(f"LIDAR file {lidar_path} does not exist")
    
    with open(lidar_path, 'rb') as f:
        lidar_data = PlyData.read(f)
        points = np.array([list(point) for point in lidar_data['vertex']])
    
    return points

# Function to process and visualize LIDAR point cloud
def process_and_visualize_lidar_data(lidar_data, save_path=None):
    """
    Process and visualize the LIDAR point cloud data.
    """
    lidar_path = lidar_data['lidar_path']
    points = read_lidar_data(lidar_path)
    
    # Assume that the points are [x, y, z] coordinates
    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Visualize the point cloud
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c=z, cmap='jet', s=0.1)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Visualization of LIDAR Data: {lidar_data['token']}")
    plt.show()

    # Save the figure if a save path is provided
    if save_path:
        fig_path = os.path.join(save_path, 'lidar_visualization.png')
        fig.savefig(fig_path)
        print(f"LIDAR visualization saved at {fig_path}")

# Main function to run the process
def main(data_list):
    """
    Main function to load data, process LIDAR, and visualize results.
    """
    lidar_data_list = load_data(data_list)
    
    for lidar_data in lidar_data_list:
        # Simulating img_metas and calling save_occ for visualization
        img_metas = {
            'data': [{
                '0': [{
                    'scene_token': 'example_scene',
                    'lidar_token': 'example_lidar_token',
                    'lidar_data': lidar_data
                }]
            }]
        }
        save_occ(
            pred_c=np.random.randn(1, 3, 256, 256, 256),
            pred_f=np.random.randn(1, 3, 256, 256, 256),
            img_metas=img_metas,
            path='./output'
        )

if __name__ == "__main__":
    data = [
        {
            'lidar_path': './data/nuscenes/samples/LIDAR_TOP/n015-2018-08-02-17-16-37+0800__LIDAR_TOP__1533201470448696.pcd.bin',
            'token': 'fd8420396768425eabec9bdddf7e64b6',
            'can_bus': np.array([2.49895286e+02, 9.17552428e+02, 0.00000000e+00, 9.98467758e-01, 0.00000000e+00, 0.00000000e+00, -5.53365786e-02]),
            'frame_idx': 0,
            'cams': {
                'CAM_FRONT': {'data_path': './data/nuscenes/samples/CAM_FRONT/n015-2018-08-02-17-16-37+0800__CAM_FRONT__1533201470412460.jpg'},
                'CAM_FRONT_RIGHT': {'data_path': './data/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_RIGHT__1533201470420339.jpg'},
                'CAM_FRONT_LEFT': {'data_path': './data/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-08-02-17-16-37+0800__CAM_FRONT_LEFT__1533201470404874.jpg'},
                'CAM_BACK': {'data_path': './data/nuscenes/samples/CAM_BACK/n015-2018-08-02-17-16-37+0800__CAM_BACK__1533201470437525.jpg'}
            }
        }
    ]
    
    main(data)
