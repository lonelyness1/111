from datetime import datetime

# ============== 全局配置加固 ==============
opencv_num_threads = 0
mp_cfg = dict(mp_start_method='fork')
cudnn_benchmark = True

# 分布式参数（去掉 timeout，避免 pretty_text 报错）
dist_params = dict(backend='nccl')

# mmcv 的 DDP 配置
ddp_cfg = dict(backend='nccl', find_unused_parameters=True)

# =========================================

_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]

plugin = True
plugin_dir = "projects/occ_plugin"

custom_imports = dict(
    imports=[
        'projects.occ_plugin.occupancy.detectors.occnet_with_centerpoint',
        'projects.occ_plugin.datasets.nuscenes_occ_dataset',
        'projects.occ_plugin.datasets.pipelines.formating'
    ],
    allow_failed_imports=False
)

input_modality = dict(
    use_lidar=True, use_camera=True,
    use_radar=False, use_map=False,
    use_external=False
)

occ_path        = "data/nuScenes-Occupancy"
depth_gt_path   = "data/depth_gt"
data_root       = "data/nuscenes/"
train_ann_file  = "data/nuscenes/nuscenes_infos_train_20.pkl"
val_ann_file    = "data/nuscenes/nuscenes_infos_val_500.pkl"
dataset_type    = 'NuscOCCDataset'

class_names = [
    'car','truck','construction_vehicle','bus','trailer','barrier',
    'motorcycle','bicycle','pedestrian','traffic_cone'
]

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]

occ_size     = [512, 512, 40]
empty_idx    = 0
visible_mask = False
num_cls      = 17

voxel_size      = [0.1, 0.1, 0.1]
out_size_factor = 8
pillar_size     = [v * out_size_factor for v in voxel_size]

voxel_x = (point_cloud_range[3] - point_cloud_range[0]) / occ_size[0]
voxel_y = (point_cloud_range[4] - point_cloud_range[1]) / occ_size[1]
voxel_z = (point_cloud_range[5] - point_cloud_range[2]) / occ_size[2]

TIMESTAMP = datetime.now().strftime('%Y-%m-%d_%H-%M')
work_dir  = f"work_dirs/Multimodal-TWO_{TIMESTAMP}"

data_config = {
    'cams': ['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (896, 1600),
    'src_size':   (900, 1600),
    'resize': (-0.06, 0.11),
    'rot':    (-5.4, 5.4),
    'flip':   True,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.0
}

lss_downsample = [4,4,4]
grid_config = {
    'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_x * lss_downsample[0]],
    'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_y * lss_downsample[1]],
    'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_z * lss_downsample[2]],
    'dbound': [2.0,58.0,0.5]
}

numC_Trans        = 128
voxel_out_channel = 256
voxel_out_indices = (0,1,2,3)
voxel_channels    = [128,256,384,512]

model = dict(
    type='OCCNetFusionWithCenter',
    loss_norm=False,
    allow_no_det_label=True,

    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0,1,2,3),
        frozen_stages=0,
        norm_eval=False,
        with_cp=False,
        style='pytorch',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
    ),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256,512,1024,2048],
        upsample_strides=[1,2,4,8],
        out_channels=[numC_Trans//4]*4
    ),

    pts_voxel_layer=dict(
        max_num_points=10, voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(50000, 80000)
    ),
    pts_pillar_layer=dict(
        max_num_points=20, voxel_size=pillar_size,
        point_cloud_range=point_cloud_range,
        max_voxels=(50000, 100000)
    ),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=5),
    pts_middle_encoder=dict(
        type='SparseLiDAREnc8x', input_channel=4, base_channel=16,
        out_channel=numC_Trans,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        sparse_shape_xyz=[1024,1024,80]
    ),

    occ_fuser=dict(
        type='DeepFuser', in_channels=numC_Trans,
        out_channels=numC_Trans, fuse_nonzero=True,
        dropout=0.1
    ),

    occ_encoder_backbone=dict(
        type='CustomResNet3D', depth=18,
        n_input_channels=numC_Trans,
        block_inplanes=voxel_channels,
        out_indices=voxel_out_indices,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ),
    occ_encoder_neck=dict(
        type='FPN3D', in_channels=voxel_channels,
        out_channels=voxel_out_channel,
        with_cp=False,
        norm_cfg=dict(type='SyncBN', requires_grad=True)
    ),

    pts_bbox_head=dict(
        type='OccHead',
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        soft_weights=True,
        cascade_ratio=4,
        sample_from_voxel=True,
        sample_from_img=True,
        final_occ_size=occ_size,
        fine_topk=5000,
        empty_idx=empty_idx,
        num_level=len(voxel_out_indices),
        in_channels=[voxel_out_channel]*len(voxel_out_indices),
        out_channel=num_cls,
        point_cloud_range=point_cloud_range,
        loss_weight_cfg=dict(
            loss_voxel_ce_weight=1.0,
            loss_voxel_sem_scal_weight=1.0,
            loss_voxel_geo_scal_weight=1.0,
            loss_voxel_lovasz_weight=1.0
        )
    ),

    det_bbox_head=dict(
        type='CenterHead',
        in_channels=numC_Trans,
        tasks=[dict(num_class=len(class_names), class_names=class_names)],
        common_heads=dict(
            reg=(2,2), height=(1,2),
            dim=(3,2), rot=(2,2), vel=(2,2)
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
        loss_cls=dict(
            type='GaussianFocalLoss',
            reduction='mean',
            loss_weight=0.01
        ),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        train_cfg=dict(
            grid_size=[
                int((point_cloud_range[3]-point_cloud_range[0]) / voxel_size[0]),
                int((point_cloud_range[4]-point_cloud_range[1]) / voxel_size[1]),
                1
            ],
            voxel_size=voxel_size,
            out_size_factor=out_size_factor,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            point_cloud_range=point_cloud_range,
            code_weights=[1.0]*10
        ),
        test_cfg=dict(
            post_center_limit_range=point_cloud_range,
            max_per_img=500,
            score_threshold=0.1,
            nms_type='circle',
            nms_thr=0.7,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size
        )
    ),

    empty_idx=empty_idx
)

bda_aug_conf = dict(
    rot_lim=(0,0), scale_lim=(0.95,1.05),
    flip_dx_ratio=0.5, flip_dy_ratio=0.5
)

train_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, pad_empty_sweeps=True),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=data_config,
        is_train=True,
        sequential=False,
        aligned=True,
        trans_only=False,
        depth_gt_path=depth_gt_path,
        load_depth=True,
        img_norm_cfg=None,
        mmlabnorm=True
    ),
    dict(
        type='LoadAnnotationsBEVDepth',
        bda_aug_conf=bda_aug_conf,
        classes=class_names,
        input_modality=input_modality
    ),
    dict(
        type='LoadOccupancy',
        occ_path=occ_path,
        use_semantic=True,
        to_float32=True,
        grid_size=occ_size,
        pc_range=point_cloud_range,
        use_vel=False,
        unoccupied=empty_idx,
        cal_visible=visible_mask
    ),
    dict(type='PrepareDetAnnotations', allow_missing=True),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_occ', 'points', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=5, use_dim=5),
    dict(type='LoadPointsFromMultiSweeps', sweeps_num=10, pad_empty_sweeps=True),
    dict(
        type='LoadMultiViewImageFromFiles_BEVDet',
        data_config=data_config,
        sequential=False,
        aligned=True,
        trans_only=False,
        depth_gt_path=depth_gt_path,
        img_norm_cfg=None,
        mmlabnorm=True
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
        occ_path=occ_path,
        use_semantic=True,
        to_float32=True,
        grid_size=occ_size,
        pc_range=point_cloud_range,
        use_vel=False,
        unoccupied=empty_idx,
        cal_visible=visible_mask
    ),
    dict(type='OccDefaultFormatBundle3D', class_names=class_names, with_label=False),
    dict(
        type='Collect3D',
        keys=['img_inputs', 'gt_occ', 'points'],
        meta_keys=['pc_range','occ_size','scene_token','lidar_token']
    )
]

train_dataset = dict(
    type=dataset_type,
    data_root=data_root,
    occ_root=occ_path,
    ann_file=train_ann_file,
    pipeline=train_pipeline,
    classes=class_names,
    modality=input_modality,
    use_valid_flag=True,
    test_mode=False,
    occ_size=occ_size,
    pc_range=point_cloud_range,
    box_type_3d='LiDAR'
)
test_dataset = dict(
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

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    persistent_workers=True,
    pin_memory=True,
    drop_last=True,
    train=train_dataset,
    val=test_dataset,
    test=test_dataset,
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=3e-4,
    weight_decay=0.01,
    paramwise_cfg=dict(custom_keys={'img_backbone': dict(lr_mult=0.1)})
)
optimizer_config = dict(
    grad_clip=dict(max_norm=5, norm_type=2)
)

lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0/3,
    min_lr_ratio=1e-3
)

runner = dict(type='EpochBasedRunner', max_epochs=1)

find_unused_parameters = True

evaluation = dict(
    interval=1,
    pipeline=test_pipeline,
    metric=['occupancy','bbox'],
    save_best='SSC_fine_mean',
    rule='greater'
)

log_config = dict(
    interval=50,
    hooks=[dict(type='TextLoggerHook')]
)
