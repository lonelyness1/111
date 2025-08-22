from .core.evaluation.eval_hooks import OccDistEvalHook, OccEvalHook
from .core.evaluation.efficiency_hooks import OccEfficiencyHook
from .core.visualizer import save_occ

# 只导出你需要的 pipeline helpers，避免从 mmdet.datasets.pipelines import PIPELINES 的错误
from .datasets.pipelines import (
    PhotoMetricDistortionMultiViewImage, PadMultiViewImage,
    NormalizeMultiviewImage, CustomCollect3D
)

from .occupancy import *
# 不要在这里 import prepare_det_annotations，也不要访问 mmdet.datasets.pipelines.PIPELINES
import importlib
importlib.import_module('projects.occ_plugin.occupancy.pipelines.prepare_det_annotations')
