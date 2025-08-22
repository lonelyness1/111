import os
import sys
import mayavi.mlab as mlab
import numpy as np
from datetime import datetime
import open3d as o3d

# 设置无图形界面运行模式
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# 配置参数
voxel_size = 0.8
pc_range = [-50, -50, -5, 50, 50, 3]


def visualize_and_save(fov_voxels, save_path_img, save_path_ply):
    """渲染 3D 点云并保存为图像和 .ply 文件"""
    # 渲染 3D 点云
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    plt_plot_fov = mlab.points3d(
        fov_voxels[:, 0],
        fov_voxels[:, 1],
        fov_voxels[:, 2],
        fov_voxels[:, 3],
        colormap="viridis",
        scale_factor=voxel_size - 0.05 * voxel_size,
        mode="cube",
        opacity=1.0,
        vmin=0,
        vmax=19,
    )
    # 保存渲染结果为图片
    mlab.savefig(save_path_img)
    mlab.close()

    # 保存为 .ply 文件
    print(f"Saving PLY file: {save_path_ply}")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(fov_voxels[:, :3])

    # 颜色映射
    color_map = (
        np.array(
            [
                [0, 0, 0],  # 0 无颜色
                [255, 120, 50],  # 1 barrier
                [255, 192, 203],  # 2 bicycle
                [255, 255, 0],  # 3 bus
                [0, 150, 245],  # 4 car
                [0, 255, 255],  # 5 construction_vehicle
                [200, 180, 0],  # 6 motorcycle
                [255, 0, 0],  # 7 pedestrian
                [255, 240, 150],  # 8 traffic_cone
                [135, 60, 0],  # 9 trailer
                [160, 32, 240],  # 10 truck
                [255, 0, 255],  # 11 driveable_surface
                [139, 137, 137],  # 12 others
                [75, 0, 75],  # 13 sidewalk
                [150, 240, 80],  # 14 terrain
                [230, 230, 250],  # 15 manmade
                [0, 175, 0],  # 16 vegetation
                [0, 255, 127],  # 17 ego car
                [255, 99, 71],  # 18 other color
                [0, 191, 255],  # 19 another color
            ]
        )
        / 255.0
    )
    labels = fov_voxels[:, 3].astype(int)
    colors = color_map[labels]
    pcd.colors = o3d.utility.Vector3dVector(colors)

    o3d.io.write_point_cloud(save_path_ply, pcd)


def process_npy_files_recursive(input_dir, save_dir):
    """递归遍历文件夹下所有 .npy 文件进行可视化并保存"""
    for root, _, files in os.walk(input_dir):
        for file_name in files:
            if file_name.endswith(".npy"):
                input_file = os.path.join(root, file_name)
                print(f"Processing: {input_file}")

                # 加载点云数据
                fov_voxels = np.load(input_file)
                fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]  # 过滤无效点

                # 调整坐标系
                fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
                fov_voxels[:, 0] += pc_range[0]
                fov_voxels[:, 1] += pc_range[1]
                fov_voxels[:, 2] += pc_range[2]

                # 构建输出目录，保持目录结构
                relative_path = os.path.relpath(root, input_dir)
                target_dir = os.path.join(save_dir, relative_path)
                os.makedirs(target_dir, exist_ok=True)

                # 生成文件名
                base_name = os.path.splitext(file_name)[0]
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_img = os.path.join(target_dir, f"{base_name}_{timestamp}.png")
                output_ply = os.path.join(target_dir, f"{base_name}_{timestamp}.ply")

                # 可视化并保存
                visualize_and_save(fov_voxels, output_img, output_ply)
                print(f"Saved visualization to {output_img} and {output_ply}")


# 主程序入口
if __name__ == "__main__":
    input_path = sys.argv[1]  # 输入文件或目录路径
    save_directory = sys.argv[2]  # 保存目录

    if os.path.isfile(input_path):
        # 如果是单个 .npy 文件
        fov_voxels = np.load(input_path)
        fov_voxels = fov_voxels[fov_voxels[..., 3] > 0]
        fov_voxels[:, :3] = (fov_voxels[:, :3] + 0.5) * voxel_size
        fov_voxels[:, 0] += pc_range[0]
        fov_voxels[:, 1] += pc_range[1]
        fov_voxels[:, 2] += pc_range[2]

        os.makedirs(save_directory, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_img = os.path.join(save_directory, f"visualization_{timestamp}.png")
        output_ply = os.path.join(save_directory, f"visualization_{timestamp}.ply")
        visualize_and_save(fov_voxels, output_img, output_ply)
        print(f"Saved visualization to {output_img} and {output_ply}")
    elif os.path.isdir(input_path):
        # 如果是目录，递归处理所有 .npy 文件
        process_npy_files_recursive(input_path, save_directory)
    else:
        print("Error: Invalid input path. Please provide a valid file or directory.")
