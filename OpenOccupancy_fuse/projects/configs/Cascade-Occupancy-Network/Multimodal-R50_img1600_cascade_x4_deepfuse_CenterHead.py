from datetime import datetime

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = "projects/occ_plugin"


custom_imports = dict(
    imports=['projects.occ_plugin.occupancy.detectors.occnet_with_centerpoint'],
    allow_failed_imports=False
)

input_modality = dict(
    use_lidar=True,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False
)

# 数据路径
occ_path = "./data/nuScenes-Occupancy"
depth_gt_path = './data/depth_gt'
train_ann_file = "./data/nuscenes/nuscenes_occ_infos_train.pkl"
val_ann_file = "./data/nuscenes/nuscenes_occ_infos_val.pkl"
data_root = 'data/nuscenes/'

# 数据集类型
dataset_type = 'NuscOCCDataset'

# 类别与范围
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
occ_size = [512, 512, 40]
empty_idx = 0
visible_mask = False
num_cls = 17

# 网格与体素
voxel_size = [0.1, 0.1, 0.1]
out_size_factor = 8
pillar_size = [voxel_size[0]*out_size_factor, voxel_size[1]*out_size_factor, voxel_size[2]*out_size_factor]
lss_downsample = [4, 4, 4]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

# 时间戳与输出目录
TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
work_dir = f"autodl-tmp/OpenOccupancy_fuse/work_dirs/occ_with_center_{TIMESTAMP}"

# 摄像头配置
data_config = {
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size': (900, 1600),
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
}

# 生成 BEV 网格配置
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0, 58.0, 0.5],
}

# 特征维度
numC_Trans = 128
voxel_out_channel = 256
voxel_out_indices = (0, 1, 2, 3)
voxel_channels = [128, 256, 384, 512]

# -------------------------------------------------------------------------
# 模型定义：OCCNetFusion + CenterPoint
# -------------------------------------------------------------------------
model = dict(
    type='OCCNetFusionWithCenter',
    loss_norm=True,

    img_backbone=dict(
        pretrained='torchvision://resnet50', type='ResNet', depth=50,
        num_stages=4, out_indices=(0, 1, 2, 3), frozen_stages=0,
        with_cp=True, norm_cfg=dict(type='SyncBN', requires_grad=True),
        norm_eval=False, style='pytorch'
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[1, 2, 4, 8],
        out_channels=[numC_Trans // 4] * 4
    ),

    pts_voxel_layer=dict(
        max_num_points=10, point_cloud_range=point_cloud_range,
        voxel_size=voxel_size, max_voxels=(90000, 120000)
    ),
    pts_pillar_layer=dict(
        max_num_points=20, point_cloud_range=point_cloud_range,
        voxel_size=pillar_size, max_voxels=(90000, 180000)
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x', input_channel=4, base_channel=16,
        out_channel=numC_Trans, norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024, 1024, 80]
    ),

    occ_fuser=dict(
        type='DeepFuser', in_channels=numC_Trans,
        out_channels=numC_Trans, dropout=0.1, fuse_nonzero=True
    ),

    occ_encoder_backbone=dict(
        type='CustomResNet3D', depth=18, n_input_channels=numC_Trans,
        block_inplanes=voxel_channels, out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ),
    occ_encoder_neck=dict(
        type='FPN3D', with_cp=True,
        in_channels=voxel_channels, out_channels=voxel_out_channel,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ),

    # ———— Occupancy 头 ————
    occ_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=4,
        sample_from_voxel=True,
        sample_from_img=True,
        final_occ_size=occ_size,
        fine_topk=10000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel] * len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0
        )
    ),

    # ———— CenterPoint 检测头 ————
    det_bbox_head=dict(
        type='CenterHead',
        in_channels=384,
        tasks=[dict(
            num_class=len(class_names),
            class_names=class_names
        )],
        common_heads=dict(
            reg=(2, 2), height=(1, 2),
            dim=(3, 2), rot=(2, 2), vel=(2, 2)
        ),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range,
            post_center_range=point_cloud_range,
            max_num=500,
            score_threshold=0.1,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size,
            code_size=7
        ),
        separate_head=dict(type='SeparateHead', init_bias=0.1, final_kernel=1),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        train_cfg=dict(
            pts=dict(
                grid_size=[
                    int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0]),
                    int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1]),
                    1
                ],
                voxel_size=voxel_size,
                out_size_factor=out_size_factor,
                dense_reg=1,
                gaussian_overlap=0.1,
                max_objs=500,
                min_radius=2,
                point_cloud_range=point_cloud_range,
                code_weights=[1.0] * 10
            )
        ),
        test_cfg=dict(
            pts=dict(
                post_center_limit_range=point_cloud_range,
                nms=dict(type='circle', iou_threshold=0.7),
                score_threshold=0.1,
                max_num=500
            )
        )
    ),

    empty_idx=empty_idx
)

# -------------------------------------------------------------------------
# 数据流水线
# -------------------------------------------------------------------------
bda_aug_conf = dict(
    rot_lim=(-0, 0),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        is_train=True,
        data_config=data_config,
        sequential=False,
        aligned=True,
        trans_only=False,
        depth_gt_path=depth_gt_path,
        mmlabnorm=True,
        load_depth=True,
        img_norm_cfg=None
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality
    ),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path=occ_path,
        grid_size=occ_size,
        use_vel=False,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        cal_visible=visible_mask
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points']),
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=data_config,
        depth_gt_path=depth_gt_path,
        sequential=False,
        aligned=True,
        trans_only=False,
        mmlabnorm=True,
        img_norm_cfg=None
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality,
        is_train=False
    ),
    dict(
        type='LoadOccupancy',
        to_float32=True,
        use_semantic=True,
        occ_path=occ_path,
        grid_size=occ_size,
        use_vel=False,
        unoccupied=empty_idx,
        pc_range=point_cloud_range,
        cal_visible=visible_mask
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ', 'points'],
        meta_keys=['pc_range', 'occ_size', 'scene_token', 'lidar_token']
    ),
]

test_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_path,
    ann_file=val_ann_file,
    pipeline=test_pipeline,
    classes=class_names,
    modality=input_modality,
    occ_size=occ_size,
    pc_range=point_cloud_range
)

train_config = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_path,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    classes=class_names,
    modality=input_modality,
    test_mode=False,
    use_valid_flag=True,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    box_type_3d='LiDAR'
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=train_config,
    val=test_config,
    test=test_config,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler'),
)

# -------------------------------------------------------------------------
# 优化器 & 学习率调度 & Runner & 评估
# -------------------------------------------------------------------------
optimizer = dict(
    type='AdamW',
    lr=3e-4,
    paramwise_cfg=dict(
        custom_keys={'img_backbone': dict(lr_mult=0.1)}
    ),
    weight_decay=0.01
)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3
)

runner = dict(type='EpochBasedRunner', max_epochs=24)

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    save_best='SSC_fine_mean',
    rule='greater'
)
