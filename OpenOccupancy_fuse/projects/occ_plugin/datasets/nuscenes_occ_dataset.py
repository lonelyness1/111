import os
import numpy as np
import torch

from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS
from mmdet3d.datasets import NuScenesDataset
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def _unwrap_dc(x):
    return x.data if isinstance(x, DC) or hasattr(x, 'data') else x


def _as_lidar_boxes(boxes):
    boxes = _unwrap_dc(boxes)
    if boxes is None:
        return None
    if isinstance(boxes, LiDARInstance3DBoxes):
        return boxes
    if isinstance(boxes, (torch.Tensor, np.ndarray)):
        if isinstance(boxes, np.ndarray):
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        boxes = boxes.reshape(-1, boxes.shape[-1])
        assert boxes.shape[-1] >= 7, f'Expect last dim >=7, got {boxes.shape}'
        return LiDARInstance3DBoxes(boxes[:, :7].contiguous(), box_dim=7)
    return boxes


def _pick_gt(results):
    gt_b = results.get('gt_bboxes_3d', None)
    gt_l = results.get('gt_labels_3d', None)
    if gt_b is None or gt_l is None:
        ann = results.get('ann_info', {}) or {}
        if gt_b is None:
            gt_b = ann.get('gt_bboxes_3d', None)
        if gt_l is None:
            gt_l = ann.get('gt_labels_3d', None)
    gt_b = _as_lidar_boxes(gt_b)
    gt_l = _unwrap_dc(gt_l)
    return gt_b, gt_l


def _wrap_gt_into_results(results, gt_b, gt_l):
    """统一写回 GT，包成 DataContainer"""
    results['gt_bboxes_3d'] = DC(gt_b, cpu_only=True)
    results['gt_labels_3d'] = DC(
        gt_l if isinstance(gt_l, torch.Tensor)
        else torch.as_tensor(gt_l, dtype=torch.long)
    )

    # 同步 ann_info（有些算子还会用到），但后续在 pipeline 的 OccDefaultFormatBundle3D 会 pop 掉
    ann = results.get('ann_info', {}) or {}
    ann['gt_bboxes_3d'] = gt_b
    ann['gt_labels_3d'] = gt_l if isinstance(gt_l, torch.Tensor) else torch.as_tensor(gt_l, dtype=torch.long)
    results['ann_info'] = ann
    return results


def _sanitize_boxes_anywhere(obj):
    """
    递归“消毒”：把任何位置出现的 LiDARInstance3DBoxes 统一包成 DataContainer(cpu_only=True)。
    避免 mmcv.collate 在嵌套结构里遇到裸 boxes 报错。
    """
    if isinstance(obj, DC):
        return obj  # 已经安全

    if isinstance(obj, LiDARInstance3DBoxes):
        return DC(obj, cpu_only=True)

    if isinstance(obj, dict):
        return {k: _sanitize_boxes_anywhere(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple)):
        new_list = [_sanitize_boxes_anywhere(v) for v in obj]
        return type(obj)(new_list)

    return obj


@DATASETS.register_module()
class NuscOCCDataset(NuScenesDataset):
    def __init__(self, occ_size, pc_range, occ_root, **kwargs):
        super().__init__(**kwargs)
        self.data_infos = list(sorted(self.data_infos, key=lambda e: e['timestamp']))
        self.data_infos = self.data_infos[:: self.load_interval]
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.occ_root = occ_root
        self._set_group_flag()

    def __getitem__(self, idx):
        if self.test_mode:
            data = self.prepare_test_data(idx)
        else:
            original_idx = idx
            for _ in range(10):
                data = self.prepare_train_data(idx)
                if data is not None:
                    break
                idx = self._rand_another(idx)
            else:
                data = self.prepare_train_data(original_idx)

            if data is None:
                # fallback
                input_dict = self.get_data_info(original_idx)
                self.pre_pipeline(input_dict)
                data = self.pipeline(input_dict)
                empty_boxes = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7)
                empty_labels = torch.zeros((0,), dtype=torch.long)
                data = _wrap_gt_into_results(data, empty_boxes, empty_labels)

        # ⭐ 最后一步强制包装顶层 GT
        gt_b, gt_l = _pick_gt(data)
        if gt_b is None:
            gt_b = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7)
            gt_l = torch.zeros((0,), dtype=torch.long)
        data = _wrap_gt_into_results(data, gt_b, gt_l)

        # ⭐ 终极保险：递归把任意位置的裸 LiDARInstance3DBoxes 都包成 DataContainer(cpu_only=True)
        data = _sanitize_boxes_anywhere(data)
        b = data['gt_bboxes_3d']; b = b.data if isinstance(b, DC) else b
        #print('[DBG] num_boxes =', [len(bb) for bb in (b if isinstance(b, (list,tuple)) else [b])])

        return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        results = self.pipeline(input_dict)

        gt_b, gt_l = _pick_gt(results)
        if gt_b is None or gt_l is None:
            gt_b = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7)
            gt_l = torch.zeros((0,), dtype=torch.long)
        results = _wrap_gt_into_results(results, gt_b, gt_l)

        return results

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        self.pre_pipeline(input_dict)
        results = self.pipeline(input_dict)

        gt_b, gt_l = _pick_gt(results)
        if gt_b is None:
            gt_b = LiDARInstance3DBoxes(torch.zeros((0, 7), dtype=torch.float32), box_dim=7)
            gt_l = torch.zeros((0,), dtype=torch.long)
        results = _wrap_gt_into_results(results, gt_b, gt_l)

        return results

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            sample_idx=info['token'],
            pts_filename=info['lidar_path'],
            sweeps=info['sweeps'],
            lidar2ego_translation=info['lidar2ego_translation'],
            lidar2ego_rotation=info['lidar2ego_rotation'],
            ego2global_translation=info['ego2global_translation'],
            ego2global_rotation=info['ego2global_rotation'],
            prev_idx=info['prev'],
            next_idx=info['next'],
            scene_token=info['scene_token'],
            can_bus=info['can_bus'],
            timestamp=info['timestamp'] / 1e6,
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            lidar_token=info['lidar_token'],
            lidarseg=info['lidarseg'],
            curr=info,
        )

        if self.modality.get('use_camera', False):
            image_paths, lidar2img_rts, lidar2cam_rts, cam_intrinsics = [], [], [], []
            lidar2cam_dic = {}
            for cam_type, cam_info in info['cams'].items():
                image_paths.append(cam_info['data_path'])

                lidar2cam_r = np.linalg.inv(cam_info['sensor2lidar_rotation'])
                lidar2cam_t = cam_info['sensor2lidar_translation'] @ lidar2cam_r.T
                lidar2cam_rt = np.eye(4)
                lidar2cam_rt[:3, :3] = lidar2cam_r.T
                lidar2cam_rt[3, :3] = -lidar2cam_t

                intrinsic = cam_info['cam_intrinsic']
                viewpad = np.eye(4)
                viewpad[:intrinsic.shape[0], :intrinsic.shape[1]] = intrinsic
                lidar2img_rt = (viewpad @ lidar2cam_rt.T)

                cam_intrinsics.append(viewpad)
                lidar2img_rts.append(lidar2img_rt)
                lidar2cam_rts.append(lidar2cam_rt.T)
                lidar2cam_dic[cam_type] = lidar2cam_rt.T

            input_dict.update(
                dict(
                    img_filename=image_paths,
                    lidar2img=lidar2img_rts,
                    cam_intrinsic=cam_intrinsics,
                    lidar2cam=lidar2cam_rts,
                    lidar2cam_dic=lidar2cam_dic,
                )
            )

        annos = self.get_ann_info(index)
        input_dict['ann_info'] = annos

        if self.modality.get('use_lidar', False):
            input_dict['pts_filename'] = input_dict['pts_filename'].replace(
                './data/nuscenes/', self.data_root
            )
            for sw in input_dict['sweeps']:
                sw['data_path'] = sw['data_path'].replace(
                    './data/nuscenes/', self.data_root
                )
        return input_dict

    # 评估不变（如果你原来用了 cm_to_ious / format_*，别忘了在相应 utils 里导入）
    def evaluate(self, results, logger=None, **kwargs):
        from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results

        eval_results = {}
        evaluation_semantic = sum(results['SC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SC_results(ious[1:], return_dic=True)
        for key, val in res_dic.items():
            eval_results[f'SC_{key}'] = val
        if logger is not None:
            logger.info('SC Evaluation')
            logger.info(res_table)

        evaluation_semantic = sum(results['SSC_metric'])
        ious = cm_to_ious(evaluation_semantic)
        res_table, res_dic = format_SSC_results(ious, return_dic=True)
        for key, val in res_dic.items():
            eval_results[f'SSC_{key}'] = val
        if logger is not None:
            logger.info('SSC Evaluation')
            logger.info(res_table)

        if 'SSC_metric_fine' in results:
            evaluation_semantic = sum(results['SSC_metric_fine'])
            ious = cm_to_ious(evaluation_semantic)
            res_table, res_dic = format_SSC_results(ious, return_dic=True)
            for key, val in res_dic.items():
                eval_results[f'SSC_fine_{key}'] = val
            if logger is not None:
                logger.info('SSC fine Evaluation')
                logger.info(res_table)
        return eval_results
