from typing import List

import torch
from torch import nn
import torch.nn.functional as F

from mmdet3d.models.builder import FUSION_LAYERS


@FUSION_LAYERS.register_module()
class DeepFuser(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        fuse_nonzero=False,
    ):
        super(DeepFuser, self).__init__()
        self.in_channels = in_channels
        self.out_channel = out_channels
        self.learnedAlign = nn.MultiheadAttention(in_channels, 1, dropout=dropout, 
                                             kdim=in_channels, vdim=in_channels, batch_first=True)
        self.fuse_nonzero = fuse_nonzero

    def points_to_pixel(self, points, transforms):
        """project points in voxel to image,
        return pixel coors

        Args:
            points (torch.Tensor): []
            transforms (tuple):
                - tuple of [num_cams, *]
        """
        with torch.no_grad():
            # project 3D point cloud (after bev-aug) onto multi-view images for corresponding 2D coordinates
            rots, trans, intrins, post_rots, post_trans, bda_mat, H_img, W_img = transforms
            inv_bda = bda_mat.inverse()
            points = (inv_bda @ points.unsqueeze(-1)).squeeze(-1)
            
            # from lidar to camera
            points = points.view(-1, 1, 3)
            points = points - trans.view(1, -1, 3)
            inv_rots = rots.inverse().unsqueeze(0)
            points = (inv_rots @ points.unsqueeze(-1))
            
            # from camera to raw pixel
            points = (intrins.unsqueeze(0) @ points).squeeze(-1)
            points_d = points[..., 2:3]
            points_uv = points[..., :2] / (points_d + 1e-5)
            
            # from raw pixel to transformed pixel
            points_uv = post_rots[..., :2, :2].unsqueeze(0) @ points_uv.unsqueeze(-1)
            points_uv = points_uv.squeeze(-1) + post_trans[..., :2].unsqueeze(0)

            points_uv[..., 0] = (points_uv[..., 0] / (W_img-1) - 0.5) * 2
            points_uv[..., 1] = (points_uv[..., 1] / (H_img-1) - 0.5) * 2

            mask = (points_d[..., 0] > 1e-5) \
                & (points_uv[..., 0] > -1) & (points_uv[..., 0] < 1) \
                & (points_uv[..., 1] > -1) & (points_uv[..., 1] < 1)
            mask = torch.nan_to_num(mask)

        return points_uv.permute(2,1,0,3), mask
    
    def forward(self, img_feats, pts_feats, pts_metas, transforms):
        """_summary_

        Args:
            img_feats (torch.Tensor): [num_cam, C, CH, CW] ? B
            pts_feats (torch.Tensor): [B, C, PW, PH, D]
            pts_metas (dict)
            transforms (tuple):
                - rots [B, num_cam, 3, 3]
                ...
        """
        stop = 1
        rots, trans, intrins, post_rots, post_trans, bda_mat, img_shape = transforms
        img_H, img_W = img_shape
        batch_size = pts_feats.shape[0]
        batch_cnt = pts_feats.new_zeros(batch_size).int()  
        decorated_pts_feat = torch.zeros_like(pts_feats)
        for b in range(batch_size):
            batch_cnt[b] = (pts_metas['pillar_coors'][:,0] == b).sum()
        batch_bound = batch_cnt.cumsum(dim=0)
        if self.fuse_nonzero:
            batch_cnt_ref = pts_feats.new_zeros(batch_size).int()
            for b in range(batch_size):
                batch_cnt_ref[b] = (pts_metas['ref_coors'][:,0] == b).sum()
            batch_bound_ref = batch_cnt_ref.cumsum(dim=0)
            cur_start_ref = 0
        cur_start = 0
        for b in range(batch_size):
            cur_end = batch_bound[b]
            voxel = pts_metas['pillars'][cur_start:cur_end]
            voxel_coor = pts_metas['pillar_coors'][cur_start:cur_end]
            pillars_num_points = pts_metas['pillar_num_points'][cur_start:cur_end]
            num_voxels, max_points, p_dim = voxel.shape
            num_pts = num_voxels * max_points
            pts = voxel.view(num_pts, p_dim)[...,:3]

            batch_rots, batch_trans, batch_intrins, batch_post_rots, batch_post_trans, batch_bda_mat, batch_img_H, batch_img_W = \
                rots[b:b+1], trans[b:b+1], intrins[b:b+1], post_rots[b:b+1], post_trans[b:b+1], bda_mat[b:b+1], img_H[b:b+1], img_W[b:b+1]
            batch_transform = (batch_rots, batch_trans, batch_intrins, batch_post_rots, batch_post_trans, batch_bda_mat, batch_img_H, batch_img_W)
            num_cam = batch_rots.shape[1]
            img_uv, mask = self.points_to_pixel(pts, batch_transform)
            sampled_feat = F.grid_sample(img_feats[b].contiguous(), img_uv.contiguous(), align_corners=True, mode='bilinear', padding_mode='zeros')
            sampled_feat = sampled_feat.squeeze(-1).permute(2, 0, 1)
            sampled_feat = sampled_feat.view(num_voxels, max_points, num_cam, self.in_channels)
            mask = mask.permute(1, 0, 2).view(num_voxels, max_points, num_cam, 1)
            
            mask_points = mask.new_zeros((mask.shape[0], mask.shape[1] + 1))
            mask_points[torch.arange(mask.shape[0], device=mask_points.device).long(), pillars_num_points.long()] = 1
            mask_points = mask_points.cumsum(dim=1).bool()
            mask_points = ~mask_points
            mask = mask_points[:,:-1].unsqueeze(-1).unsqueeze(-1) & mask

            mask = mask.reshape(num_voxels, max_points*num_cam, 1)
            sampled_feat = sampled_feat.reshape(num_voxels, max_points*num_cam, self.in_channels)
            K = sampled_feat
            V = sampled_feat
            Q = pts_feats[b, :, voxel_coor[:, 3].long(), voxel_coor[:, 2].long(), voxel_coor[:, 1].long()].t().unsqueeze(1)
            valid = mask[...,0].sum(dim=1) > 0 
            attn_output = pts_feats.new_zeros(num_voxels, 1, self.in_channels)
            attn_output[valid] = self.learnedAlign(Q[valid], K[valid], V[valid], attn_mask=(~mask[valid]).permute(0,2,1))[0]
            decorated_pts_feat[b, :, voxel_coor[:, 3].long(), voxel_coor[:, 2].long(), voxel_coor[:, 1].long()] = attn_output.squeeze(1).t()
            cur_start = cur_end
            if self.fuse_nonzero:
                cur_end_ref = batch_bound_ref[b]
                voxel = pts_metas['refs'][cur_start_ref:cur_end_ref]
                voxel_coor = pts_metas['ref_coors'][cur_start_ref:cur_end_ref]
                num_voxels, max_points, p_dim = voxel.shape
                num_pts = num_voxels * max_points
                pts = voxel.view(num_pts, p_dim)[...,:3]
                img_uv, mask = self.points_to_pixel(pts, batch_transform)
                mask = mask.permute(1, 2, 0)
                sampled_feat = F.grid_sample(img_feats[b].contiguous(), img_uv.contiguous(), align_corners=True, mode='bilinear', padding_mode='zeros')
                sampled_feat = sampled_feat.squeeze(-1).permute(2, 0, 1)
                sampled_feat = sampled_feat.view(num_voxels, max_points, num_cam, self.in_channels)
                sampled_feat = sampled_feat.reshape(num_voxels, max_points*num_cam, self.in_channels)
                K = sampled_feat
                V = sampled_feat
                Q = pts_feats[b, :, voxel_coor[:, 3].long(), voxel_coor[:, 2].long(), voxel_coor[:, 1].long()].t().unsqueeze(1)
                valid = mask[...,0].sum(dim=1) > 0 
                attn_output = pts_feats.new_zeros(num_voxels, 1, self.in_channels)
                attn_output[valid] = self.learnedAlign(Q[valid], K[valid], V[valid], attn_mask=(~mask[valid]).permute(0,2,1))[0]
                decorated_pts_feat[b, :, voxel_coor[:, 3].long(), voxel_coor[:, 2].long(), voxel_coor[:, 1].long()] = attn_output.squeeze(1).t()
                cur_start_ref = cur_end_ref
        return decorated_pts_feat
