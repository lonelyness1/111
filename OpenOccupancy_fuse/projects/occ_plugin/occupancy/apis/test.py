# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights.
# Modified by Zhiqi Li & patched for OCC eval
# ---------------------------------------------
import os.path as osp
import shutil
import tempfile
import time

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info
from mmdet.utils import get_root_logger
from fvcore.nn import parameter_count_table

from projects.occ_plugin.utils.formating import cm_to_ious, format_SC_results, format_SSC_results
from projects.occ_plugin.core import save_occ


ALLOWED_KEYS = {'img_inputs', 'points', 'img_metas', 'gt_occ', 'visible_mask'}

def _keep_only_allowed_keys(batch):
    """仅保留模型推理需要的键，其他（如 ann_info、gt_bboxes_3d 等）全部删除。"""
    if isinstance(batch, dict):
        for k in list(batch.keys()):
            if k not in ALLOWED_KEYS:
                batch.pop(k, None)
    return batch


@torch.no_grad()
def custom_single_gpu_test(model, data_loader, show=False, out_dir=None, show_score_thr=0.3):
    model.eval()

    SC_metric = 0
    SSC_metric = 0
    SSC_metric_fine = 0
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    for i, data in enumerate(data_loader):
        data = _keep_only_allowed_keys(data)  # ★ 关键过滤

        result = model(return_loss=False, rescale=True, **data)

        if show:
            save_occ(result.get('pred_c'), result.get('pred_f'),
                     data.get('img_metas'), out_dir,
                     data.get('visible_mask'), data.get('gt_occ'))

        if 'SC_metric' in result:
            SC_metric += result['SC_metric']
            print(format_SC_results(cm_to_ious(SC_metric)[1:]))

        if 'SSC_metric' in result:
            SSC_metric += result['SSC_metric']
            print(format_SSC_results(cm_to_ious(SSC_metric)))

        if 'SSC_metric_fine' in result:
            SSC_metric_fine += result['SSC_metric_fine']
            print(format_SSC_results(cm_to_ious(SSC_metric_fine)))

        prog_bar.update()

    return {
        'SC_metric': SC_metric,
        'SSC_metric': SSC_metric,
        'SSC_metric_fine': SSC_metric_fine,
    }


@torch.no_grad()
def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False, out_dir=None):
    """Multi-GPU 测试（DDP）"""
    model.eval()

    SC_metric = []
    SSC_metric = []
    SSC_metric_fine = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    time.sleep(2)  # 防止个别环境死锁
    logger = get_root_logger()
    logger.info(parameter_count_table(model))

    for i, data in enumerate(data_loader):
        data = _keep_only_allowed_keys(data)  # ★ 关键过滤

        result = model(return_loss=False, rescale=True, **data)

        if show:
            save_occ(result.get('pred_c'), result.get('pred_f'),
                     data.get('img_metas'), out_dir, gt_occ=data.get('gt_occ'))

        if 'SC_metric' in result:
            SC_metric.append(result['SC_metric'])
        if 'SSC_metric' in result:
            SSC_metric.append(result['SSC_metric'])
        if 'SSC_metric_fine' in result:
            SSC_metric_fine.append(result['SSC_metric_fine'])

        if rank == 0:
            prog_bar.update(world_size)

    res = {}
    if len(SC_metric) > 0:
        SC_metric = [sum(SC_metric)]
        SC_metric = collect_results_cpu(SC_metric, len(dataset), tmpdir)
        res['SC_metric'] = SC_metric
    if len(SSC_metric) > 0:
        SSC_metric = [sum(SSC_metric)]
        SSC_metric = collect_results_cpu(SSC_metric, len(dataset), tmpdir)
        res['SSC_metric'] = SSC_metric
    if len(SSC_metric_fine) > 0:
        SSC_metric_fine = [sum(SSC_metric_fine)]
        SSC_metric_fine = collect_results_cpu(SSC_metric_fine, len(dataset), tmpdir)
        res['SSC_metric_fine'] = SSC_metric_fine

    return res


def collect_results_cpu(result_part, size, tmpdir=None, type='list'):
    rank, world_size = get_dist_info()

    if tmpdir is None:
        MAX_LEN = 512
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)

    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()

    if rank == 0:
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))

        if type == 'list':
            ordered_results = []
            for res in part_list:
                ordered_results.extend(list(res))
            ordered_results = ordered_results[:size]
        else:
            raise NotImplementedError

        shutil.rmtree(tmpdir)

    dist.barrier()

    if rank != 0:
        return None

    return ordered_results
