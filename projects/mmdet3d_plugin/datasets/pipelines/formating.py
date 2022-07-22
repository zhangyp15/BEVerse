# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
from mmcv.parallel import DataContainer as DC

from mmdet3d.core.bbox import BaseInstance3DBoxes
from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor

from mmdet3d.datasets.pipelines import DefaultFormatBundle

import pdb


@PIPELINES.register_module()
class MTLFormatBundle3D(DefaultFormatBundle):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields for voxels,
    including "proposals", "gt_bboxes", "gt_labels", "gt_masks" and
    "gt_semantic_seg".
    These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - proposals: (1)to tensor, (2)to DataContainer
    - gt_bboxes: (1)to tensor, (2)to DataContainer
    - gt_bboxes_ignore: (1)to tensor, (2)to DataContainer
    - gt_labels: (1)to tensor, (2)to DataContainer
    """

    def __init__(self, class_names, with_gt=True, with_label=True):
        super(MTLFormatBundle3D, self).__init__()
        self.class_names = class_names
        self.with_gt = with_gt
        self.with_label = with_label

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """
        # Format 3D data
        if 'points' in results:
            assert isinstance(results['points'], BasePoints)
            results['points'] = DC(results['points'].tensor)

        for key in ['voxels', 'coors', 'voxel_centers', 'num_points']:
            if key not in results:
                continue
            results[key] = DC(to_tensor(results[key]), stack=False)

        for key in [
                'proposals', 'gt_bboxes', 'gt_bboxes_ignore', 'gt_labels',
                'gt_labels_3d', 'attr_labels', 'pts_instance_mask',
                'pts_semantic_mask', 'centers2d', 'depths'
        ]:
            if key not in results:
                continue

            if isinstance(results[key], list):
                results[key] = DC([to_tensor(res) if res is not None else res for res in results[key]])
            else:
                results[key] = DC(to_tensor(results[key]))

        # list of LiDARInstanceBoxes, size = num_future
        results['gt_bboxes_3d'] = DC(results['gt_bboxes_3d'], cpu_only=True)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(class_names={self.class_names}, '
        repr_str += f'with_gt={self.with_gt}, with_label={self.with_label})'
        return repr_str
