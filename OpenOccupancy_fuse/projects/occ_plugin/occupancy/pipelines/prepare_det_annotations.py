import mmdet.datasets.pipelines as _mmdet_pipelines
from mmcv.utils import Registry

if not hasattr(_mmdet_pipelines, 'PIPELINES'):
    _mmdet_pipelines.PIPELINES = Registry('pipeline')

PIPELINES = _mmdet_pipelines.PIPELINES

# 避免重复注册
if 'PrepareDetAnnotations' not in PIPELINES.module_dict:
    @PIPELINES.register_module()
    class PrepareDetAnnotations:
        def __init__(self, allow_missing=False):
            self.allow_missing = allow_missing

        def __call__(self, results):
            ann = results.get('ann_info', {}) or {}
            gt_bboxes_3d = ann.get('gt_bboxes_3d', None)
            gt_labels_3d = ann.get('gt_labels_3d', None)

            if not self.allow_missing:
                if gt_bboxes_3d is None or gt_labels_3d is None:
                    raise RuntimeError(
                        f"PrepareDetAnnotations: missing detection annotations: "
                        f"gt_bboxes_3d={gt_bboxes_3d}, gt_labels_3d={gt_labels_3d}"
                    )
            results['gt_bboxes_3d'] = gt_bboxes_3d
            results['gt_labels_3d'] = gt_labels_3d
            return results

        def __repr__(self):
            return f"{self.__class__.__name__}(allow_missing={self.allow_missing})"