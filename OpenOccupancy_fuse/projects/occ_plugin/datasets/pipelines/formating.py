# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from mmdet3d.datasets.pipelines import DefaultFormatBundle3D
from mmdet3d.core.bbox import LiDARInstance3DBoxes


def _unwrap_dc(x):
    """如果是 DataContainer 就取其 data；否则原样返回。"""
    return x.data if isinstance(x, DC) or hasattr(x, 'data') else x


def _to_tensor_safe(x):
    """
    更鲁棒的 tensor 化：
    - 先解包 DataContainer
    - 兼容 memoryview / np.memmap（先 np.asarray 再 torch.as_tensor）
    - 其它情况优先用 mmdet 的 to_tensor，失败再兜底 as_tensor(np.asarray)
    """
    x = _unwrap_dc(x)

    if isinstance(x, torch.Tensor):
        return x

    if isinstance(x, (memoryview, np.memmap)):
        return torch.as_tensor(np.asarray(x))

    try:
        return to_tensor(x)
    except Exception:
        return torch.as_tensor(np.asarray(x))


@PIPELINES.register_module()
class OccDefaultFormatBundle3D(DefaultFormatBundle3D):
    """默认3D数据格式化，同时支持 occupancy 与 3D检测 GT。

    关键点：
    - 统一先解包 DataContainer，再按类型重打包为 DataContainer
    - LiDARInstance3DBoxes 强制 DC(cpu_only=True)，避免 collate 报错
    - gt_occ/gt_vel 使用 _to_tensor_safe 兼容 memoryview/np.memmap
    - 最后删除 ann_info，防止 collate 递归进去遇到裸的 boxes
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, results):
        # 先执行父类默认处理（会处理 img/pts/gt_labels 等常规字段）
        results = super(OccDefaultFormatBundle3D, self).__call__(results)

        # -------- Occupancy 相关 --------
        if 'gt_occ' in results:
            occ = _unwrap_dc(results['gt_occ'])
            results['gt_occ'] = DC(_to_tensor_safe(occ), stack=True)

        if 'gt_vel' in results:
            vel = _unwrap_dc(results['gt_vel'])
            results['gt_vel'] = DC(_to_tensor_safe(vel), stack=False)

        # -------- 检测框相关（核心修复点）--------
        if 'gt_bboxes_3d' in results:
            bboxes_3d = _unwrap_dc(results['gt_bboxes_3d'])
            if isinstance(bboxes_3d, LiDARInstance3DBoxes):
                results['gt_bboxes_3d'] = DC(bboxes_3d, cpu_only=True)
            else:
                results['gt_bboxes_3d'] = DC(_to_tensor_safe(bboxes_3d), stack=False)

        if 'gt_labels_3d' in results:
            labels_3d = _unwrap_dc(results['gt_labels_3d'])
            results['gt_labels_3d'] = DC(_to_tensor_safe(labels_3d), stack=True)

        if 'gt_bboxes_ignore' in results:
            bboxes_ignore = _unwrap_dc(results['gt_bboxes_ignore'])
            results['gt_bboxes_ignore'] = DC(_to_tensor_safe(bboxes_ignore), stack=False)

        # -------- 彻底终结嵌套 collate 报错 --------
        # 训练时已经把需要的 GT 放到顶层并打包好了，ann_info 会让 collate 递归进去再次遇到裸对象，干脆删掉
        if 'ann_info' in results:
            results.pop('ann_info', None)

        return results
