import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.runner import get_dist_info

from ..metrics import IntersectionOverUnion, PanopticMetric
from ..visualize import Visualizer

import pdb


def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False, show=False):
    """Test model with multiple gpus.

    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.

    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.

    Returns:
        list: The prediction results.
    """

    # multi-task settings
    test_mode = data_loader.dataset.test_submission

    task_enable = model.module.pts_bbox_head.task_enbale
    det_enable = task_enable.get('3dod', False)
    map_enable = task_enable.get('map', False)
    motion_enable = task_enable.get('motion', False)
    det_results = []

    if test_mode:
        map_enable = False
        motion_enable = False

    # define metrics
    if map_enable:
        num_map_class = 4
        semantic_map_iou_val = IntersectionOverUnion(num_map_class)
        semantic_map_iou_val = semantic_map_iou_val.cuda()

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

    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))

    out_dir = 'eval_visualize'
    coordinate_system = dataset.coordinate_system
    visualizer = Visualizer(
        out_dir=out_dir, coordinate_system=coordinate_system)

    time.sleep(2)  # This line can prevent deadlock problem in some cases.

    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(
                return_loss=False,
                rescale=True,
                img_metas=data['img_metas'],
                img_inputs=data['img_inputs'],
                future_egomotions=data['future_egomotions'],
                img_is_valid=data['img_is_valid'][0],
            )

        if det_enable:
            det_results.extend(result['bbox_results'])

        if map_enable:
            pred_semantic_indices = result['pred_semantic_indices']
            target_semantic_indices = data['semantic_indices'][0].cuda()

            semantic_map_iou_val(pred_semantic_indices,
                                 target_semantic_indices)

        if motion_enable:
            motion_segmentation, motion_instance = result['motion_segmentation'], result['motion_instance']
            has_invalid_frame = data['has_invalid_frame'][0]
            # valid future frames < n_future_frame, skip the evaluation
            if not has_invalid_frame.item():

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
                save_path='beverse_demo_visualize_v2/{}'.format(
                    data['img_metas'][0].data[0][0]['sample_idx'])
            )

        if rank == 0:
            for _ in range(data_loader.batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if det_enable:
        if gpu_collect:
            det_results = collect_results_gpu(det_results, len(dataset))
        else:
            det_results = collect_results_cpu(
                det_results, len(dataset), tmpdir)

    if map_enable:
        scores = semantic_map_iou_val.compute()
        mIoU = sum(scores[1:]) / (len(scores) - 1)
        if rank == 0:
            print('\n[Validation {:04d} / {:04d}]: semantic map iou = {}, mIoU = {:.3f}'.format(
                len(dataset), len(dataset), scores, mIoU,
            ))

    if motion_enable:
        if rank == 0:
            print(
                '\n[Validation {:04d} / {:04d}]: motion metrics: '.format(len(dataset), len(dataset)))

        for key, grid in EVALUATION_RANGES.items():
            results_str = 'grid = {}: '.format(key)

            panoptic_scores = motion_panoptic_metrics[key].compute()
            iou_scores = motion_iou_metrics[key].compute()

            # logging
            if rank == 0:
                results_str += 'iou = {:.3f}, '.format(
                    iou_scores[1].item() * 100)
                for panoptic_key, value in panoptic_scores.items():
                    results_str += '{} = {:.3f}, '.format(
                        panoptic_key, value[1].item() * 100)
                print(results_str)

    return det_results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results
