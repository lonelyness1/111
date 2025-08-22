import torch
import torch.nn.functional as F
import numpy as np
import time
import copy

from mmdet.models import DETECTORS
from mmdet3d.models.detectors import MVXTwoStageDetector
from mmdet3d.ops import Voxelization
from mmcv.runner import force_fp32
from mmdet3d.models.builder import build_fusion_layer, build_backbone, build_neck


@DETECTORS.register_module(force=True)
class OCCNetFusion(MVXTwoStageDetector):
    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 #
                 pts_pillar_layer=None,
                 occ_encoder_backbone=None,
                 occ_encoder_neck=None,
                 occ_fuser=None,
                 loss_norm=False,
                 empty_idx=0,
                 #
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OCCNetFusion,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                              pts_middle_encoder, pts_fusion_layer,
                              img_backbone, pts_backbone, img_neck, pts_neck,
                              pts_bbox_head, img_roi_head, img_rpn_head,
                              train_cfg, test_cfg, pretrained, init_cfg)
        self.pts_pillar_layer = Voxelization(**pts_pillar_layer)
        self.occ_fuser = build_fusion_layer(occ_fuser)
        self.occ_encoder_backbone = build_backbone(occ_encoder_backbone)
        self.occ_encoder_neck = build_neck(occ_encoder_neck)
        self.loss_norm = loss_norm
        self.record_time = False
        self.empty_idx = empty_idx

    @force_fp32()
    def occ_encoder(self, x):
        x = self.occ_encoder_backbone(x)
        x = self.occ_encoder_neck(x)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, pts, mode='voxel'):
        assert mode in ['voxel', 'pillar']
        voxels, num_points, coors = [], [], []
        for pt in pts:
            if mode == 'voxel':
                res_voxels, res_coors, res_num_points = self.pts_voxel_layer(pt)
            else:
                res_voxels, res_coors, res_num_points = self.pts_pillar_layer(pt)
            voxels.append(res_voxels)
            num_points.append(res_num_points)
            coors.append(res_coors)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coor_w_batch = []
        for ii, coor in enumerate(coors):
            pad_coor = F.pad(coor, (1, 0), mode='constant', value=ii)
            coor_w_batch.append(pad_coor)
        coors = torch.cat(coor_w_batch, dim=0)
        return voxels, num_points, coors

    def extract_pts_feat(self, pts):
        # voxelize + encode
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)

        # pillar 方式
        pillars, pillar_num_points, pillar_coors = self.voxelize(pts, mode='pillar')
        pillar_features = self.pts_voxel_encoder(pillars, pillar_num_points, pillar_coors)

        # 如果 fuse_nonzero，需要额外计算 refs
        if self.occ_fuser.fuse_nonzero:
            pc_range = self.pts_pillar_layer.point_cloud_range
            grid_size = self.pts_pillar_layer.grid_size
            H, W, D = grid_size[0], grid_size[1], grid_size[2]
            dtype = pillar_features.dtype
            device = pillar_features.device
            xs = torch.linspace(0.5, H - 0.5, H, dtype=dtype, device=device).view(H, 1, 1).expand(H, W, D) / H
            ys = torch.linspace(0.5, W - 0.5, W, dtype=dtype, device=device).view(1, W, 1).expand(H, W, D) / W
            zs = torch.linspace(0.5, D - 0.5, D, dtype=dtype, device=device).view(1, 1, D).expand(H, W, D) / D
            zeros = torch.zeros((H, W, D), dtype=dtype, device=device)
            ref_3d = torch.stack([xs, ys, zs, zeros], dim=-1).reshape(-1, 4)
            # 将归一化坐标映射回真实坐标
            ref_3d[..., 0:1] = ref_3d[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
            ref_3d[..., 1:2] = ref_3d[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
            ref_3d[..., 2:3] = ref_3d[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]
            ref_pts = [torch.vstack([pt, ref_3d]) for pt in pts]
            refs, ref_num_points, ref_coors = self.voxelize(ref_pts, mode='pillar')
            mask = ref_num_points == 1

            pts_metas = dict(
                pillar_center=pillar_features,
                pillars=pillars,
                pillar_num_points=pillar_num_points,
                pillar_coors=pillar_coors,
                refs=refs[mask, 0].unsqueeze(1),
                ref_coors=ref_coors[mask],
                pts=pts,
            )
        else:
            pts_metas = dict(
                pillar_center=pillar_features,
                pillars=pillars,
                pillar_num_points=pillar_num_points,
                pillar_coors=pillar_coors,
                pts=pts,
            )

        return x['x'], pts_metas

    def extract_img_feat(self, img, img_metas):
        B, N, C, H, W = img.shape
        img = img.view(B * N, C, H, W)

        img_feats = self.img_backbone(img)
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            if isinstance(img_feats, (list, tuple)):
                img_feats = img_feats[0]
        _, output_dim, output_H, output_W = img_feats.shape
        img_feats = img_feats.view(B, N, output_dim, output_H, output_W)
        return [img_feats]

    def extract_feat(self, pts, img, img_metas):
        """
        提取并融合多模态 BEV 特征，同时保存到 self.bev_feat 供子类使用。
        """
        img_feats = self.extract_img_feat(img[0], img_metas)
        pts_feats, pts_metas = self.extract_pts_feat(pts)
        transforms = img[1:8]

        if self.record_time:
            torch.cuda.synchronize()
            t0 = time.time()
        bev_feats = self.occ_fuser(img_feats[0], pts_feats, pts_metas, transforms)
        # *** 关键：保存融合后的 BEV 特征 ***
        self.bev_feat = bev_feats
        if self.record_time:
            torch.cuda.synchronize()
            t1 = time.time()
            self.time_stats['occ_fuser'].append(t1 - t0)

        voxel_feats_enc = self.occ_encoder(bev_feats)
        if not isinstance(voxel_feats_enc, list):
            voxel_feats_enc = [voxel_feats_enc]
        if self.record_time:
            torch.cuda.synchronize()
            t2 = time.time()
            self.time_stats['occ_encoder'].append(t2 - t1)

        return (voxel_feats_enc, img_feats, pts_feats)

    @force_fp32(apply_to=('voxel_feats'))
    def forward_pts_train(self,
                          voxel_feats,
                          gt_occ=None,
                          points_occ=None,
                          img_metas=None,
                          transform=None,
                          img_feats=None,
                          pts_feats=None,
                          visible_mask=None):
        outs = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        losses = self.pts_bbox_head.loss(
            output_voxels=outs['output_voxels'],
            output_voxels_fine=outs['output_voxels_fine'],
            output_coords_fine=outs['output_coords_fine'],
            target_voxels=gt_occ,
            target_points=points_occ,
            img_metas=img_metas,
            visible_mask=visible_mask,
        )
        return losses

    def forward_train(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      gt_occ=None,
                      points_occ=None,
                      visible_mask=None,
                      **kwargs):
        voxel_feats, img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        losses = dict()
        transform = img_inputs[1:8] if img_inputs is not None else None
        losses_occupancy = self.forward_pts_train(
            voxel_feats,
            gt_occ,
            points_occ,
            img_metas,
            transform=transform,
            img_feats=img_feats,
            pts_feats=pts_feats,
            visible_mask=visible_mask
        )
        losses.update(losses_occupancy)

        if self.loss_norm:
            for k, v in losses.items():
                if k.startswith('loss'):
                    losses[k] = v / (v.detach() + 1e-9)
        return losses

    def forward_test(self,
                     points=None,
                     img_metas=None,
                     img_inputs=None,
                     gt_occ=None,
                     visible_mask=None,
                     **kwargs):
        return self.simple_test(
            img_metas=img_metas,
            img=img_inputs,
            points=points,
            gt_occ=gt_occ,
            visible_mask=visible_mask,
            **kwargs)

    def simple_test(self,
                    img_metas=None,
                    img=None,
                    points=None,
                    rescale=False,
                    points_occ=None,
                    gt_occ=None,
                    visible_mask=None):
        voxel_feats, img_feats, pts_feats = self.extract_feat(
            points, img=img, img_metas=img_metas)
        transform = img[1:8] if img is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )

        pred_c = output['output_voxels'][0]
        SC_metric, _ = self.evaluation_semantic(
            pred_c, gt_occ, eval_type='SC', visible_mask=visible_mask)
        SSC_metric, SSC_occ_metric = self.evaluation_semantic(
            pred_c, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        pred_f = None
        SSC_metric_fine = None
        if output['output_voxels_fine'] is not None:
            if output['output_coords_fine'] is not None:
                fine_pred = output['output_voxels_fine'][0]
                fine_coord = output['output_coords_fine'][0]
                pred_f = self.empty_idx * torch.ones_like(
                    gt_occ)[:, None].repeat(1,
                                            fine_pred.shape[1], 1, 1,
                                            1).float()
                pred_f[:, :, fine_coord[0], fine_coord[1], fine_coord[2]] = fine_pred.permute(1, 0)[None]
            else:
                pred_f = output['output_voxels_fine'][0]
            SC_metric, _ = self.evaluation_semantic(
                pred_f, gt_occ, eval_type='SC', visible_mask=visible_mask)
            SSC_metric_fine, _ = self.evaluation_semantic(
                pred_f, gt_occ, eval_type='SSC', visible_mask=visible_mask)

        test_output = {
            'SC_metric': SC_metric,
            'SSC_metric': SSC_metric,
            'pred_c': pred_c,
            'pred_f': pred_f,
        }
        if SSC_metric_fine is not None:
            test_output['SSC_metric_fine'] = SSC_metric_fine
        return test_output

    def evaluation_semantic(self, pred, gt, eval_type, visible_mask=None):
        _, H, W, D = gt.shape
        pred = F.interpolate(pred,
                             size=[H, W, D],
                             mode='trilinear',
                             align_corners=False).contiguous()
        pred = torch.argmax(pred[0], dim=0).cpu().numpy()
        gt = gt[0].cpu().numpy().astype(np.int)
        noise_mask = gt != 255

        if eval_type == 'SC':
            gt[gt != self.empty_idx] = 1
            pred[pred != self.empty_idx] = 1
            return fast_hist(pred[noise_mask], gt[noise_mask], max_label=2), None

        if eval_type == 'SSC':
            hist_occ = None
            if visible_mask is not None:
                visible_mask = visible_mask[0].cpu().numpy()
                mask = noise_mask & (visible_mask != 0)
                hist_occ = fast_hist(pred[mask], gt[mask], max_label=17)
            hist = fast_hist(pred[noise_mask], gt[noise_mask], max_label=17)
            return hist, hist_occ

    def forward_dummy(self,
                      points=None,
                      img_metas=None,
                      img_inputs=None,
                      points_occ=None,
                      **kwargs):
        voxel_feats, img_feats, pts_feats = self.extract_feat(
            points, img=img_inputs, img_metas=img_metas)
        transform = img_inputs[1:8] if img_inputs is not None else None
        output = self.pts_bbox_head(
            voxel_feats=voxel_feats,
            points=points_occ,
            img_metas=img_metas,
            img_feats=img_feats,
            pts_feats=pts_feats,
            transform=transform,
        )
        return output


def fast_hist(pred, label, max_label=18):
    pred = copy.deepcopy(pred.flatten())
    label = copy.deepcopy(label.flatten())
    bin_count = np.bincount(max_label * label.astype(int) + pred,
                            minlength=max_label ** 2)
    return bin_count[:max_label ** 2].reshape(max_label, max_label)
