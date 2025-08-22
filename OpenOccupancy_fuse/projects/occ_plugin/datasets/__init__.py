# 确保 custom pipeline 注册被执行：最保底地 import
import projects.occ_plugin.occupancy.pipelines.prepare_det_annotations  # noqa: F401
from mmdet.datasets.builder import PIPELINES as MMDET_PIPELINES

from projects.occ_plugin.occupancy.pipelines.prepare_det_annotations import PrepareDetAnnotations

# 如果还没在 registry 里，手动注册一次（保险）
if 'PrepareDetAnnotations' not in MMDET_PIPELINES.module_dict:
    MMDET_PIPELINES.register_module(module=PrepareDetAnnotations)
#from .nuscenes_dataset import CustomNuScenesDataset
from .nuscenes_occ_dataset import NuscOCCDataset
from .builder import custom_build_dataset

__all__ = [
    'CustomNuScenesDataset', 'NuscOCCDataset'
]
