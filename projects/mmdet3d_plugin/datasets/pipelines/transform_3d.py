# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
import warnings
from mmcv import is_tuple_of
from mmcv.utils import build_from_cfg

from mmdet3d.core import VoxelGenerator
from mmdet3d.core.bbox import (CameraInstance3DBoxes, DepthInstance3DBoxes,
                               LiDARInstance3DBoxes, box_np_ops)
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import RandomFlip

import pdb


@PIPELINES.register_module()
class MTLRandomFlip3D(RandomFlip):
    """Flip the points & bbox.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        sync_2d (bool, optional): Whether to apply flip according to the 2D
            images. If True, it will apply the same flip as that to 2D images.
            If False, it will decide whether to flip randomly and independently
            to that of 2D images. Defaults to True.
        flip_ratio_bev_horizontal (float, optional): The flipping probability
            in horizontal direction. Defaults to 0.0.
        flip_ratio_bev_vertical (float, optional): The flipping probability
            in vertical direction. Defaults to 0.0.
    """

    def __init__(self,
                 sync_2d=True,
                 flip_ratio_bev_horizontal=0.0,
                 flip_ratio_bev_vertical=0.0,
                 update_img2lidar=False,
                 **kwargs):
        super(MTLRandomFlip3D, self).__init__(
            flip_ratio=flip_ratio_bev_horizontal, **kwargs)

        self.sync_2d = sync_2d
        self.flip_ratio_bev_vertical = flip_ratio_bev_vertical
        if flip_ratio_bev_horizontal is not None:
            assert isinstance(
                flip_ratio_bev_horizontal,
                (int, float)) and 0 <= flip_ratio_bev_horizontal <= 1
        if flip_ratio_bev_vertical is not None:
            assert isinstance(
                flip_ratio_bev_vertical,
                (int, float)) and 0 <= flip_ratio_bev_vertical <= 1
        self.update_img2lidar = update_img2lidar

    def random_flip_data_3d(self, input_dict, direction='horizontal'):
        """Flip 3D data randomly.

        Args:
            input_dict (dict): Result dict from loading pipeline.
            direction (str): Flip direction. Default: horizontal.

        Returns:
            dict: Flipped results, 'points', 'bbox3d_fields' keys are \
                updated in the result dict.
        """
        assert direction in ['horizontal', 'vertical']

        if len(input_dict['bbox3d_fields']) == 0:  # test mode
            input_dict['bbox3d_fields'].append('empty_box3d')
            input_dict['empty_box3d'] = input_dict['box_type_3d'](
                np.array([], dtype=np.float32))

        assert len(input_dict['bbox3d_fields']) == 1

        for key in input_dict['bbox3d_fields']:
            # flip bounding boxes for each frame
            for frame in range(len(input_dict[key])):
                if input_dict[key][frame] is not None:
                    input_dict[key][frame].flip(direction)

        # flip map vectors
        if direction == 'horizontal':
            for vector in input_dict['vectors']:
                vector['pts'] = vector['pts'] * \
                    np.array([1, -1, 1]).reshape(1, 3)

        elif direction == 'vertical':
            for vector in input_dict['vectors']:
                vector['pts'] = vector['pts'] * \
                    np.array([-1, 1, 1]).reshape(1, 3)

        else:
            raise ValueError

        if 'centers2d' in input_dict:
            assert self.sync_2d is True and direction == 'horizontal', \
                'Only support sync_2d=True and horizontal flip with images'
            w = input_dict['ori_shape'][1]
            input_dict['centers2d'][..., 0] = \
                w - input_dict['centers2d'][..., 0]
            # need to modify the horizontal position of camera center
            # along u-axis in the image (flip like centers2d)
            # ['cam2img'][0][2] = c_u
            # see more details and examples at
            # https://github.com/open-mmlab/mmdetection3d/pull/744
            input_dict['cam2img'][0][2] = w - input_dict['cam2img'][0][2]

    def update_transform(self, input_dict):
        transform = torch.zeros(
            (*input_dict['img_inputs'][1].shape[:-2], 4, 4)).float()

        transform[..., :3, :3] = input_dict['img_inputs'][1]
        transform[..., :3, -1] = input_dict['img_inputs'][2]
        transform[..., -1, -1] = 1.0

        aug_transform = torch.eye(4).float()
        if input_dict['pcd_horizontal_flip']:
            aug_transform[1, 1] = -1
        if input_dict['pcd_vertical_flip']:
            aug_transform[0, 0] = -1

        aug_transform = aug_transform.view(1, 1, 4, 4)
        new_transform = aug_transform.matmul(transform)

        input_dict['img_inputs'][1][...] = new_transform[..., :3, :3]
        input_dict['img_inputs'][2][...] = new_transform[..., :3, -1]
        input_dict['aug_transform'] = aug_transform[0, 0].matmul(
            input_dict['aug_transform'])

    def __call__(self, input_dict):
        """Call function to flip points, values in the ``bbox3d_fields`` and \
        also flip 2D image and its annotations.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction', \
                'pcd_horizontal_flip' and 'pcd_vertical_flip' keys are added \
                into result dict.
        """
        # filp 2D image and its annotations
        super(MTLRandomFlip3D, self).__call__(input_dict)

        if self.sync_2d:
            input_dict['pcd_horizontal_flip'] = input_dict['flip']
            input_dict['pcd_vertical_flip'] = False
        else:
            if 'pcd_horizontal_flip' not in input_dict:
                flip_horizontal = True if np.random.rand(
                ) < self.flip_ratio else False
                input_dict['pcd_horizontal_flip'] = flip_horizontal
            if 'pcd_vertical_flip' not in input_dict:
                flip_vertical = True if np.random.rand(
                ) < self.flip_ratio_bev_vertical else False
                input_dict['pcd_vertical_flip'] = flip_vertical

        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        if input_dict['pcd_horizontal_flip']:
            self.random_flip_data_3d(input_dict, 'horizontal')
            input_dict['transformation_3d_flow'].extend(['HF'])

        if input_dict['pcd_vertical_flip']:
            self.random_flip_data_3d(input_dict, 'vertical')
            input_dict['transformation_3d_flow'].extend(['VF'])

        if 'img_inputs' in input_dict:
            assert self.update_img2lidar
            self.update_transform(input_dict)

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(sync_2d={self.sync_2d},'
        repr_str += f' flip_ratio_bev_vertical={self.flip_ratio_bev_vertical})'
        return repr_str


@PIPELINES.register_module()
class MTLGlobalRotScaleTrans(object):
    """Apply global rotation, scaling and translation to a 3D scene.

    Args:
        rot_range (list[float]): Range of rotation angle.
            Defaults to [-0.78539816, 0.78539816] (close to [-pi/4, pi/4]).
        scale_ratio_range (list[float]): Range of scale ratio.
            Defaults to [0.95, 1.05].
        translation_std (list[float]): The standard deviation of translation
            noise. This applies random translation to a scene by a noise, which
            is sampled from a gaussian distribution whose standard deviation
            is set by ``translation_std``. Defaults to [0, 0, 0]
        shift_height (bool): Whether to shift height.
            (the fourth dimension of indoor points) when scaling.
            Defaults to False.
    """

    def __init__(self,
                 rot_range=[-0.78539816, 0.78539816],
                 scale_ratio_range=[0.95, 1.05],
                 translation_std=[0, 0, 0],
                 shift_height=False,
                 update_img2lidar=False):
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(rot_range, seq_types):
            assert isinstance(rot_range, (int, float)), \
                f'unsupported rot_range type {type(rot_range)}'
            rot_range = [-rot_range, rot_range]
        self.rot_range = rot_range

        assert isinstance(scale_ratio_range, seq_types), \
            f'unsupported scale_ratio_range type {type(scale_ratio_range)}'
        self.scale_ratio_range = scale_ratio_range

        if not isinstance(translation_std, seq_types):
            assert isinstance(translation_std, (int, float)), \
                f'unsupported translation_std type {type(translation_std)}'
            translation_std = [
                translation_std, translation_std, translation_std
            ]
        assert all([std >= 0 for std in translation_std]), \
            'translation_std should be positive'
        self.translation_std = translation_std
        self.shift_height = shift_height
        self.update_img2lidar = update_img2lidar

    def _trans_bbox_points(self, input_dict):
        """Private function to translate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after translation, 'points', 'pcd_trans' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        translation_std = np.array(self.translation_std, dtype=np.float32)
        trans_factor = np.random.normal(scale=translation_std, size=3).T
        input_dict['pcd_trans'] = trans_factor

        for key in input_dict['bbox3d_fields']:
            for frame in range(len(input_dict[key])):
                if input_dict[key][frame] is not None:
                    input_dict[key][frame].translate(trans_factor)

        for vector in input_dict['vectors']:
            vector['pts'] = vector['pts'] + trans_factor.reshape(1, 3)

    def _rot_bbox_points(self, input_dict):
        """Private function to rotate bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after rotation, 'points', 'pcd_rotation' \
                and keys in input_dict['bbox3d_fields'] are updated \
                in the result dict.
        """
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        # if no bbox in input_dict, only rotate points
        if len(input_dict['bbox3d_fields']) == 0:
            raise ValueError

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            # rotate each frame
            for frame in range(len(input_dict[key])):
                if input_dict[key][frame] is not None and len(input_dict[key][frame].tensor) > 0:
                    points, rot_mat_T = input_dict[key][frame].rotate(
                        noise_rotation, input_dict['points'])

                    input_dict['points'] = points
                    input_dict['pcd_rotation'] = rot_mat_T

        # rotate map vectors
        for vector in input_dict['vectors']:
            vector['pts'] = np.dot(
                vector['pts'], input_dict['pcd_rotation'].numpy())

    def robust_rot_bbox_points(self, input_dict):
        rotation = self.rot_range
        noise_rotation = np.random.uniform(rotation[0], rotation[1])

        angle = torch.tensor(noise_rotation)
        rot_sin = torch.sin(angle)
        rot_cos = torch.cos(angle)
        rot_mat_T = torch.tensor([[rot_cos, -rot_sin, 0],
                                  [rot_sin, rot_cos, 0],
                                  [0, 0, 1]])
        input_dict['pcd_rotation'] = rot_mat_T

        # rotate points with bboxes
        for key in input_dict['bbox3d_fields']:
            # rotate each frame
            for frame in range(len(input_dict[key])):
                if input_dict[key][frame] is not None and len(input_dict[key][frame].tensor) > 0:
                    input_dict[key][frame].rotate(rot_mat_T)

        # rotate map vectors
        for vector in input_dict['vectors']:
            vector['pts'] = np.dot(
                vector['pts'], input_dict['pcd_rotation'].numpy())

    def _scale_bbox_points(self, input_dict):
        """Private function to scale bounding boxes and points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points'and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        scale = input_dict['pcd_scale_factor']

        for key in input_dict['bbox3d_fields']:
            # scale each frame
            for frame in range(len(input_dict[key])):
                if input_dict[key][frame] is not None:
                    input_dict[key][frame].scale(scale)

        for vector in input_dict['vectors']:
            vector['pts'] = vector['pts'] * scale

    def _random_scale(self, input_dict):
        """Private function to randomly set the scale factor.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'pcd_scale_factor' are updated \
                in the result dict.
        """
        scale_factor = np.random.uniform(self.scale_ratio_range[0],
                                         self.scale_ratio_range[1])
        input_dict['pcd_scale_factor'] = scale_factor

    def update_transform(self, input_dict):
        # img_inputs: imgs, rots, trans, intrins, post_rots, post_trans

        # generate transform matrices, [batch, num_cam, 4, 4]
        transfroms = torch.zeros(
            (*input_dict['img_inputs'][1].shape[:-2], 4, 4)).float()
        transfroms[..., :3, :3] = input_dict['img_inputs'][1]
        transfroms[..., :3, -1] = input_dict['img_inputs'][2]
        transfroms[..., -1, -1] = 1.0
        input_dict['extrinsics'] = transfroms

        aug_transforms = torch.zeros_like(transfroms).float()
        if 'pcd_rotation' in input_dict:
            aug_transforms[..., :3, :3] = input_dict['pcd_rotation'].T * \
                input_dict['pcd_scale_factor']
        else:
            aug_transforms[..., :3, :3] = torch.eye(3).view(
                1, 1, 3, 3) * input_dict['pcd_scale_factor']

        aug_transforms[..., :3, -
                       1] = torch.from_numpy(input_dict['pcd_trans']).reshape(1, 3)
        aug_transforms[..., -1, -1] = 1.0
        new_transform = aug_transforms.matmul(transfroms)

        input_dict['img_inputs'][1][...] = new_transform[..., :3, :3]
        input_dict['img_inputs'][2][...] = new_transform[..., :3, -1]
        input_dict['aug_transform'] = aug_transforms[0, 0]

    def __call__(self, input_dict):
        """Private function to rotate, scale and translate bounding boxes and \
        points.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after scaling, 'points', 'pcd_rotation',
                'pcd_scale_factor', 'pcd_trans' and keys in \
                input_dict['bbox3d_fields'] are updated in the result dict.
        """
        if 'transformation_3d_flow' not in input_dict:
            input_dict['transformation_3d_flow'] = []

        # rotate
        # self._rot_bbox_points(input_dict)
        self.robust_rot_bbox_points(input_dict)

        # scale
        if 'pcd_scale_factor' not in input_dict:
            self._random_scale(input_dict)
        self._scale_bbox_points(input_dict)

        # translate
        self._trans_bbox_points(input_dict)

        # updating transform
        input_dict['transformation_3d_flow'].extend(['R', 'S', 'T'])
        if 'img_inputs' in input_dict:
            assert self.update_img2lidar
            self.update_transform(input_dict)

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(rot_range={self.rot_range},'
        repr_str += f' scale_ratio_range={self.scale_ratio_range},'
        repr_str += f' translation_std={self.translation_std},'
        repr_str += f' shift_height={self.shift_height})'

        return repr_str


@PIPELINES.register_module()
class TemporalObjectRangeFilter(object):
    """Filter objects by the range.

    Args:
        point_cloud_range (list[float]): Point cloud range.
    """

    def __init__(self, point_cloud_range):
        self.pcd_range = np.array(point_cloud_range, dtype=np.float32)

    def __call__(self, input_dict):
        """Call function to filter objects by the range.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """
        # Check points instance type and initialise bev_range

        if isinstance(input_dict['gt_bboxes_3d'][0],
                      (LiDARInstance3DBoxes, DepthInstance3DBoxes)):
            bev_range = self.pcd_range[[0, 1, 3, 4]]

        elif isinstance(input_dict['gt_bboxes_3d'][0], CameraInstance3DBoxes):
            bev_range = self.pcd_range[[0, 2, 3, 5]]

        gt_bboxes_3d_list = input_dict['gt_bboxes_3d']
        gt_labels_3d_list = input_dict['gt_labels_3d']
        instance_tokens_list = input_dict['instance_tokens']

        for index, (gt_bboxes_3d, gt_labels_3d, instance_tokens) in enumerate(zip(gt_bboxes_3d_list, gt_labels_3d_list, instance_tokens_list)):
            # filter each frame
            if gt_bboxes_3d is not None:
                mask = gt_bboxes_3d.in_range_bev(bev_range)
                gt_bboxes_3d = gt_bboxes_3d[mask]
                gt_bboxes_3d.limit_yaw(offset=0.5, period=2 * np.pi)

                # masking labels & tokens
                np_mask = mask.numpy().astype(np.bool)
                gt_labels_3d = gt_labels_3d[np_mask]
                instance_tokens = instance_tokens[np_mask]

                # updating
                gt_bboxes_3d_list[index] = gt_bboxes_3d
                gt_labels_3d_list[index] = gt_labels_3d
                instance_tokens_list[index] = instance_tokens

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d_list
        input_dict['gt_labels_3d'] = gt_labels_3d_list
        input_dict['instance_tokens'] = instance_tokens_list

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(point_cloud_range={self.pcd_range.tolist()})'
        return repr_str


@PIPELINES.register_module()
class TemporalObjectNameFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self, classes):
        self.classes = classes
        self.labels = list(range(len(self.classes)))

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """

        gt_bboxes_3d_list = input_dict['gt_bboxes_3d']
        gt_labels_3d_list = input_dict['gt_labels_3d']
        instance_tokens_list = input_dict['instance_tokens']

        for index, (gt_bboxes_3d, gt_labels_3d, instance_tokens) in enumerate(zip(gt_bboxes_3d_list, gt_labels_3d_list, instance_tokens_list)):
            if gt_labels_3d is None:
                continue

            # filter each frame
            gt_bboxes_mask = np.array([n in self.labels for n in gt_labels_3d],
                                      dtype=np.bool_)
            # updating
            gt_bboxes_3d_list[index] = gt_bboxes_3d[gt_bboxes_mask]
            gt_labels_3d_list[index] = gt_labels_3d[gt_bboxes_mask]
            instance_tokens_list[index] = instance_tokens[gt_bboxes_mask]

        input_dict['gt_bboxes_3d'] = gt_bboxes_3d_list
        input_dict['gt_labels_3d'] = gt_labels_3d_list
        input_dict['instance_tokens'] = instance_tokens_list

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(classes={self.classes})'
        return repr_str


@PIPELINES.register_module()
class ObjectValidFilter(object):
    """Filter GT objects by their names.

    Args:
        classes (list[str]): List of class names to be kept for training.
    """

    def __init__(self):
        pass

    def __call__(self, input_dict):
        """Call function to filter objects by their names.

        Args:
            input_dict (dict): Result dict from loading pipeline.

        Returns:
            dict: Results after filtering, 'gt_bboxes_3d', 'gt_labels_3d' \
                keys are updated in the result dict.
        """

        valid_flag = input_dict['gt_valid_flag']
        input_dict['gt_bboxes_3d'] = input_dict['gt_bboxes_3d'][valid_flag]
        input_dict['gt_labels_3d'] = input_dict['gt_labels_3d'][valid_flag]

        return input_dict

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(filter objects according to num_lidar_pts & num_radar_pts)'
        return repr_str
