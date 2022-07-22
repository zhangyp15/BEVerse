# Copyright (c) OpenMMLab. All rights reserved.
from collections import defaultdict
import mmcv
import torch
from mmcv.image import tensor2imgs
from os import path as osp
import pdb
import time

import numpy as np
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

# define semantic metrics
from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

from fvcore.nn import FlopCountAnalysis, parameter_count_table, flop_count_table


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3):
    """Test model with single gpu.

    This method tests model with single gpu and gives the 'show' option.
    By setting ``show=True``, it saves the visualization results under
    ``out_dir``.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        show (bool): Whether to save viualization results.
            Default: True.
        out_dir (str): The path to save visualization results.
            Default: None.

    Returns:
        list[dict]: The prediction results.
    """

    model.eval()
    dataset = data_loader.dataset
    # whether for test submission
    test_mode = dataset.test_submission

    # bev coordinate system, LiDAR or ego
    coordinate_system = dataset.coordinate_system

    prog_bar = mmcv.ProgressBar(len(dataset))
    if show:
        if test_mode:
            default_out_dir = 'test_visualize'
        else:
            default_out_dir = 'eval_visualize'

        out_dir = out_dir or default_out_dir
        visualizer = Visualizer(
            out_dir=out_dir, coordinate_system=coordinate_system)

    # logging interval
    logging_interval = 50

    # whether each task is enabled
    task_enable = model.module.pts_bbox_head.task_enbale
    det_enable = task_enable.get('3dod', False)
    map_enable = task_enable.get('map', False)
    motion_enable = task_enable.get('motion', False)
    det_results = []

    # define metrics
    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class).cuda()

    if motion_enable:
        # evaluate motion in (short, long) ranges
        EVALUATION_RANGES = {'30x30': (70, 130), '100x100': (0, 200)}
        num_motion_class = 2

        motion_panoptic_metrics = {}
        motion_iou_metrics = {}
        for key in EVALUATION_RANGES.keys():
            motion_panoptic_metrics[key] = PanopticMetric(
                n_classes=num_motion_class, temporally_consistent=True).cuda()
            motion_iou_metrics[key] = IntersectionOverUnion(
                num_motion_class).cuda()

        motion_eval_count = 0

    latencies = []
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            if test_mode:
                motion_distribution_targets = None
            else:
                motion_distribution_targets = {
                    # for motion prediction
                    'motion_segmentation': data['motion_segmentation'][0],
                    'motion_instance': data['motion_instance'][0],
                    'instance_centerness': data['instance_centerness'][0],
                    'instance_offset': data['instance_offset'][0],
                    'instance_flow': data['instance_flow'][0],
                    'future_egomotion': data['future_egomotions'][0],
                }

            result = model(
                return_loss=False,
                rescale=True,
                img_metas=data['img_metas'],
                img_inputs=data['img_inputs'],
                future_egomotions=data['future_egomotions'],
                motion_targets=motion_distribution_targets,
                img_is_valid=data['img_is_valid'][0],
            )

            time_stats = result['time_stats']
            num_input_frame = data['img_inputs'][0][0].shape[1]
            latency = (time_stats['t_BEV'] - time_stats['t0']) / \
                num_input_frame + time_stats['t_end'] - time_stats['t_BEV']

            latencies.append(latency)

        # detection results
        if det_enable:
            det_results.extend(result['bbox_results'])

        # map segmentation results
        if map_enable:
            pred_semantic_indices = result['pred_semantic_indices']

            if not test_mode:
                target_semantic_indices = data['semantic_indices'][0].cuda()
                semantic_map_iou_val(pred_semantic_indices,
                                     target_semantic_indices)
            else:
                target_semantic_indices = None

        # motion prediction results
        if motion_enable:
            motion_segmentation, motion_instance = result['motion_segmentation'], result['motion_instance']
            has_invalid_frame = data['has_invalid_frame'][0]
            # valid future frames < n_future_frame, skip the evaluation
            if not has_invalid_frame.item():
                motion_eval_count += 1
                if not test_mode:
                    # generate targets
                    motion_targets = {
                        'motion_segmentation': data['motion_segmentation'][0],
                        'motion_instance': data['motion_instance'][0],
                        'instance_centerness': data['instance_centerness'][0],
                        'instance_offset': data['instance_offset'][0],
                        'instance_flow': data['instance_flow'][0],
                        'future_egomotion': data['future_egomotions'][0],
                    }
                    motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                        motion_targets)

                    for key, grid in EVALUATION_RANGES.items():
                        limits = slice(grid[0], grid[1])
                        motion_panoptic_metrics[key](motion_instance[..., limits, limits].contiguous(
                        ), motion_labels['instance'][..., limits, limits].contiguous().cuda())

                        motion_iou_metrics[key](motion_segmentation[..., limits, limits].contiguous(
                        ), motion_labels['segmentation'][..., limits, limits].contiguous().cuda())
                else:
                    motion_labels = None

        # update prog_bar
        for _ in range(data_loader.batch_size):
            prog_bar.update()

        # for paper show, combining all results
        if show:
            target_semantic_indices = data['semantic_indices'][0].cuda()
            motion_targets = {
                'motion_segmentation': data['motion_segmentation'][0],
                'motion_instance': data['motion_instance'][0],
                'instance_centerness': data['instance_centerness'][0],
                'instance_offset': data['instance_offset'][0],
                'instance_flow': data['instance_flow'][0],
                'future_egomotion': data['future_egomotions'][0],
            }
            motion_labels, _ = model.module.pts_bbox_head.task_decoders['motion'].prepare_future_labels(
                motion_targets)

            visualizer.visualize_beverse(
                img_metas=data['img_metas'][0].data[0][0],
                bbox_results=result['bbox_results'][0],
                gt_bboxes_3d=data['gt_bboxes_3d'][0],
                gt_labels_3d=data['gt_labels_3d'][0],
                map_labels=target_semantic_indices,
                map_preds=result['pred_semantic_indices'],
                motion_labels=motion_labels,
                motion_preds=result['motion_predictions'],
                save_path='beverse_val_visualize/{:04d}'.format(i)
            )

        if (i + 1) % logging_interval == 0:
            if map_enable:
                scores = semantic_map_iou_val.compute()
                mIoU = sum(scores[1:]) / (len(scores) - 1)
                print('[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
                    i + 1, len(dataset), scores, mIoU,
                ))

            if motion_enable:
                print(
                    '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))

                for key, grid in EVALUATION_RANGES.items():
                    results_str = 'grid = {}: '.format(key)

                    panoptic_scores = motion_panoptic_metrics[key].compute()
                    iou_scores = motion_iou_metrics[key].compute()

                    results_str += 'iou = {:.3f}, '.format(
                        iou_scores[1].item() * 100)

                    for panoptic_key, value in panoptic_scores.items():
                        results_str += '{} = {:.3f}, '.format(
                            panoptic_key, value[1].item() * 100)

                    print(results_str)

            robust_latencies = latencies[20:]
            avg_latency = sum(robust_latencies) / len(robust_latencies)
            print(
                ", average forward time = {:.2f}, fps = {:.2f}".format(
                    avg_latency,
                    1 / avg_latency,
                )
            )

    if map_enable:
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        print('\n[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
            len(dataset), len(dataset), scores, mIoU,
        ))

    if motion_enable:
        print(
            '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(motion_eval_count, len(dataset)))

        for key, grid in EVALUATION_RANGES.items():
            results_str = 'grid = {}: '.format(key)

            panoptic_scores = motion_panoptic_metrics[key].compute()
            iou_scores = motion_iou_metrics[key].compute()

            results_str += 'iou = {:.3f}, '.format(
                iou_scores[1].item() * 100)

            for panoptic_key, value in panoptic_scores.items():
                results_str += '{} = {:.3f}, '.format(
                    panoptic_key, value[1].item() * 100)

            print(results_str)

    return det_results
