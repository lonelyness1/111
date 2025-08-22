# tools/train.py
from __future__ import division

import argparse
import os
import time
import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed

from projects.occ_plugin.occupancy.apis.train import custom_train_model  # 你原来的封装
from mmdet.datasets.builder import PIPELINES as MMPIPE
print("[CHECK] PrepareDetAnnotations in PIPELINES:", 'PrepareDetAnnotations' in MMPIPE.module_dict)

def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path', required=True)
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true', help='whether not to evaluate')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--gpus', type=int, help='number of gpus to use (non-distributed)')
    group.add_argument('--gpu-ids', type=int, nargs='+', help='ids of gpus to use (non-distributed)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--deterministic', action='store_true', help='set deterministic cudnn')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction, help='override some settings')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='pytorch', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=int(os.environ.get('LOCAL_RANK', 0)))
    parser.add_argument('--autoscale-lr', action='store_true', help='automatically scale lr with num gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args
def _log_cfg_safely(cfg, logger):
    """
    优先使用 cfg.pretty_text；若 yapf/ast 解析失败，自动回退到 cfg.text。
    解决 datetime.timedelta 在 MMCV 1.4.0 下 pretty_text 打印时的 SyntaxError。
    """
    try:
        logger.info(f'Config:\n{cfg.pretty_text}')
    except Exception as e:
        logger.warning(f'cfg.pretty_text failed ({e}), fallback to raw text.')
        logger.info(f'Config(raw):\n{cfg.text}')
def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    try:
        if hasattr(cfg, '_cfg_dict') and 'datetime' in cfg._cfg_dict:
            cfg._cfg_dict.pop('datetime')
    except Exception:
        pass
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # 通过 plugin 机制加载自定义模块（包括 pipeline 注册）
    if getattr(cfg, 'plugin', False):
        assert cfg.plugin_dir is not None
        import importlib
        module_name = cfg.plugin_dir.replace('/', '.').rstrip('.')
        importlib.import_module(module_name)

    # distributed setup
    if args.launcher != 'none':
        distributed = True
        init_dist(args.launcher, **cfg.get('dist_params', {}))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        torch.cuda.set_device(local_rank)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    else:
        distributed = False
        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids
        else:
            cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    if args.autoscale_lr and 'optimizer' in cfg:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8

    # work_dir
    if args.work_dir:
        cfg.work_dir = args.work_dir
    elif not cfg.get('work_dir', None):
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config to work_dir
    cfg_path = osp.join(cfg.work_dir, osp.basename(args.config))
    try:
        cfg.dump(cfg_path)
    except Exception:
        with open(cfg_path, 'w', encoding='utf-8') as f:
            if hasattr(cfg, 'text'):
                f.write(cfg.text)
            else:
                f.write(repr(cfg))

    # logger
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.get('log_level', 'INFO'), name='mmdet')

    # env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    logger.info('Environment info:\n' + '-' * 60 + '\n' + env_info + '\n' + '-' * 60)
    logger.info(f'Distributed training: {distributed}')
    _log_cfg_safely(cfg, logger)

    # seed
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, deterministic={args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
        cfg.seed = args.seed

    meta = {
        'seed': getattr(cfg, 'seed', None),
        'exp_name': osp.basename(args.config),
        'env_info': env_info,
        'config': cfg.pretty_text
    }

    # build model and dataset
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Number of params: {n_parameters}')

    datasets = [build_dataset(cfg.data.train)]

    custom_train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta
    )

if __name__ == '__main__':
    main()
