from enum import unique
from unittest import result
import torch
import numpy as np
import cv2
import pdb

from ...models.dense_heads.map_head import calculate_birds_eye_view_parameters
from mmdet.datasets.builder import PIPELINES
from ..utils.instance import convert_instance_mask_to_center_and_offset_label, convert_instance_mask_to_center_and_offset_label_with_warper
from ..utils.warper import FeatureWarper

import pdb


@PIPELINES.register_module()
class ConvertMotionLabels(object):
    def __init__(self, grid_conf, ignore_index=255, only_vehicle=True, filter_invisible=True):
        self.grid_conf = grid_conf
        # torch.tensor
        self.bev_resolution, self.bev_start_position, self.bev_dimension = calculate_birds_eye_view_parameters(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
        )
        # convert numpy
        self.bev_resolution = self.bev_resolution.numpy()
        self.bev_start_position = self.bev_start_position.numpy()
        self.bev_dimension = self.bev_dimension.numpy()
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.ignore_index = ignore_index
        self.only_vehicle = only_vehicle
        self.filter_invisible = filter_invisible

        nusc_classes = ['car', 'truck', 'construction_vehicle', 'bus', 'trailer',
                        'barrier', 'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone']
        vehicle_classes = ['car', 'bus', 'construction_vehicle',
                           'bicycle', 'motorcycle', 'truck', 'trailer']

        self.vehicle_cls_ids = np.array([nusc_classes.index(
            cls_name) for cls_name in vehicle_classes])

        self.warper = FeatureWarper(grid_conf=grid_conf)

    def __call__(self, results):
        # annotation_token ==> instance_id
        instance_map = {}

        # convert LiDAR bounding boxes to motion labels
        num_frame = len(results['gt_bboxes_3d'])
        all_gt_bboxes_3d = results['gt_bboxes_3d']
        all_gt_labels_3d = results['gt_labels_3d']
        all_instance_tokens = results['instance_tokens']
        all_vis_tokens = results['gt_vis_tokens']
        # 4x4 transformation matrix (if exist)
        bev_transform = results.get('aug_transform', None)

        segmentations = []
        instances = []

        # 对于 invalid frame: 所有 label 均为 255
        # 对于 valid frame: seg & instance 背景是 0，其它背景为255

        for frame_index in range(num_frame):
            gt_bboxes_3d, gt_labels_3d = all_gt_bboxes_3d[frame_index], all_gt_labels_3d[frame_index]
            instance_tokens = all_instance_tokens[frame_index]
            vis_tokens = all_vis_tokens[frame_index]

            if gt_bboxes_3d is None:
                # for invalid samples
                segmentation = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
                instance = np.ones(
                    (self.bev_dimension[1], self.bev_dimension[0])) * self.ignore_index
            else:
                # for valid samples
                segmentation = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))
                instance = np.zeros(
                    (self.bev_dimension[1], self.bev_dimension[0]))

                if self.only_vehicle:
                    vehicle_mask = np.isin(gt_labels_3d, self.vehicle_cls_ids)
                    gt_bboxes_3d = gt_bboxes_3d[vehicle_mask]
                    gt_labels_3d = gt_labels_3d[vehicle_mask]
                    instance_tokens = instance_tokens[vehicle_mask]
                    vis_tokens = vis_tokens[vehicle_mask]

                if self.filter_invisible:
                    visible_mask = (vis_tokens != 1)
                    gt_bboxes_3d = gt_bboxes_3d[visible_mask]
                    gt_labels_3d = gt_labels_3d[visible_mask]
                    instance_tokens = instance_tokens[visible_mask]

                # valid sample and has objects
                if len(gt_bboxes_3d.tensor) > 0:
                    bbox_corners = gt_bboxes_3d.corners[:, [
                        0, 3, 7, 4], :2].numpy()
                    bbox_corners = np.round(
                        (bbox_corners - self.bev_start_position[:2] + self.bev_resolution[:2] / 2.0) / self.bev_resolution[:2]).astype(np.int32)

                    for index, instance_token in enumerate(instance_tokens):
                        if instance_token not in instance_map:
                            instance_map[instance_token] = len(
                                instance_map) + 1

                        # instance_id start from 1
                        instance_id = instance_map[instance_token]
                        poly_region = bbox_corners[index]
                        cv2.fillPoly(segmentation, [poly_region], 1.0)
                        cv2.fillPoly(instance, [poly_region], instance_id)

            segmentations.append(segmentation)
            instances.append(instance)

        # segmentation = 1 where objects are located
        segmentations = torch.from_numpy(
            np.stack(segmentations, axis=0)).long()
        instances = torch.from_numpy(np.stack(instances, axis=0)).long()

        # generate heatmap & offset from segmentation & instance
        future_egomotions = results['future_egomotions'][- num_frame:]
        instance_centerness, instance_offset, instance_flow = convert_instance_mask_to_center_and_offset_label_with_warper(
            instance_img=instances,
            future_egomotion=future_egomotions,
            num_instances=len(instance_map),
            ignore_index=self.ignore_index,
            subtract_egomotion=True,
            warper=self.warper,
            bev_transform=bev_transform,
        )

        invalid_mask = (segmentations[:, 0, 0] == self.ignore_index)
        instance_centerness[invalid_mask] = self.ignore_index

        # only keep detection labels for the current frame
        results['gt_bboxes_3d'] = all_gt_bboxes_3d[0]
        results['gt_labels_3d'] = all_gt_labels_3d[0]
        results['instance_tokens'] = all_instance_tokens[0]
        results['gt_valid_flag'] = results['gt_valid_flag'][0]

        results.update({
            'motion_segmentation': segmentations,
            'motion_instance': instances,
            'instance_centerness': instance_centerness,
            'instance_offset': instance_offset,
            'instance_flow': instance_flow,
        })

        return results
