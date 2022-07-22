# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import division

import argparse
import copy
from black import out
import mmcv
import os
import time
import torch
import warnings
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from os import path as osp

from mmdet import __version__ as mmdet_version
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmdet.apis import set_random_seed
from mmseg import __version__ as mmseg_version

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from tqdm import tqdm, trange

import imageio
import pdb

from projects.mmdet3d_plugin.visualize import Visualizer


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--resume-from', help='the checkpoint file to resume from')
    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='whether not to evaluate the checkpoint during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument(
        '--gpus',
        type=int,
        help='number of gpus to use '
        '(only applicable to non-distributed training)')
    group_gpus.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file (deprecate), '
        'change to --cfg-options instead.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument(
        '--autoscale-lr',
        action='store_true',
        help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both specified, '
            '--options is deprecated in favor of --cfg-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    use_train = False

    # build training dataset
    if use_train:
        datasets = build_dataset(cfg.data.train)
        datasets = datasets.dataset
    else:
        datasets = build_dataset(cfg.data.test)

    shuffle = True
    out_dir = 'visualize/debug_dataset'
    visualizer = Visualizer(out_dir=out_dir)

    # shuffle the dataset
    traverse_indices = list(range(len(datasets)))
    if shuffle:
        import random
        random.shuffle(traverse_indices)

    cam_matrics_set = []

    for index in tqdm(traverse_indices):
        # data_info = datasets.get_data_info(index)
        # if not data_info['has_invalid_frame']:
        #     det_ann_infos = data_info['ann_info']
        #     plt.figure(figsize=(12, 3))
        #     for frame_index in range(len(det_ann_infos)):
        #         plt.subplot(1, len(det_ann_infos), frame_index + 1)
        #         lidar_box_segmentation = visualizer.lidar_boxes_to_binary_segmentation(
        #             det_ann_infos[frame_index]['gt_bboxes_3d'])

        #         plt.imshow(lidar_box_segmentation)
        #         plt.axis('off')
        #         plt.title('Frame = {}'.format(frame_index))

        #     plt.savefig('temporal_det_debug.png')

        #     '''
        #     ego motion: 在 x 方向上基本为 0, 即车的左右方向
        #     在 y 方向上为 0 或 负数，代表车辆从 (t + 1) ~ (t) 发生了后退，比较合理
        #     '''

        sample = datasets[index]
        # x, rots, trans, intrins, post_rots, post_trans = sample['img_inputs'][0]
        # b, num_cam = trans.shape[:2]
        # extrics = torch.eye(4).unsqueeze(0).repeat(num_cam, 1, 1)
        # extrics[:, :3, :3] = rots[0]
        # extrics[:, :3, 3] = trans[0]

        # flatten_params = torch.cat((intrins[0].view(-1), extrics.view(-1)))

        # # non-empty
        # if cam_matrics_set:
        #     find = False
        #     min_max_error = np.inf
        #     for k, param in enumerate(cam_matrics_set):
        #         if torch.allclose(flatten_params, param, rtol=0.001, atol=0.01):
        #             find = True
        #         else:
        #             max_error = (flatten_params - param).abs().max()
        #             min_max_error = min(min_max_error, max_error)

        #     if not find:

        #         pdb.set_trace()

        #         cam_matrics_set.append(flatten_params)
        #         print('Adding new camera paramters, max_error = {:.2f}, current length = {}'.format(
        #             min_max_error, len(cam_matrics_set)))
        # else:
        #     cam_matrics_set.append(flatten_params)

    pdb.set_trace()


if __name__ == '__main__':
    main()
