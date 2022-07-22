import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from .base_taskhead import BaseTaskHead
import pdb

from .loss_utils import SegmentationLoss, BinarySegmentationLoss
from mmcv.runner import auto_fp16, force_fp32
from mmdet3d.models.utils import clip_sigmoid


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height

    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor(
        [row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor(
        [row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2]
                                 for row in [x_bounds, y_bounds, z_bounds]], dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class BevFeatureSlicer(nn.Module):
    # crop the interested area in BEV feature for semantic map segmentation
    def __init__(self, grid_conf, map_grid_conf):
        super().__init__()

        if grid_conf == map_grid_conf:
            self.identity_mapping = True
        else:
            self.identity_mapping = False

            bev_resolution, bev_start_position, bev_dimension = calculate_birds_eye_view_parameters(
                grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'],
            )

            map_bev_resolution, map_bev_start_position, map_bev_dimension = calculate_birds_eye_view_parameters(
                map_grid_conf['xbound'], map_grid_conf['ybound'], map_grid_conf['zbound'],
            )

            self.map_x = torch.arange(
                map_bev_start_position[0], map_grid_conf['xbound'][1], map_bev_resolution[0])

            self.map_y = torch.arange(
                map_bev_start_position[1], map_grid_conf['ybound'][1], map_bev_resolution[1])

            # convert to normalized coords
            self.norm_map_x = self.map_x / (- bev_start_position[0])
            self.norm_map_y = self.map_y / (- bev_start_position[1])

            self.map_grid = torch.stack(torch.meshgrid(
                self.norm_map_x, self.norm_map_y, indexing='xy'), dim=2)

    def forward(self, x):
        # x: bev feature map tensor of shape (b, c, h, w)
        if self.identity_mapping:
            return x
        else:
            grid = self.map_grid.unsqueeze(0).type_as(
                x).repeat(x.shape[0], 1, 1, 1)

            return F.grid_sample(x, grid=grid, mode='bilinear', align_corners=True)


@HEADS.register_module()
class MapHead(BaseTaskHead):
    def __init__(self, task_dict, in_channels, inter_channels=None,
                 train_cfg=None,
                 test_cfg=None,
                 class_weights=[1.0, 2.0, 2.0, 2.0],
                 binary_cls=False,
                 pos_weight=2.0,
                 semantic_thresh=0.25,
                 init_cfg=None,
                 norm_cfg=dict(type='BN'),
                 **kwargs,
                 ):
        super(MapHead, self).__init__(
            task_dict, in_channels, inter_channels, init_cfg, norm_cfg)

        self.semantic_thresh = semantic_thresh
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # loss function for segmentation of semantic map
        self.binary_cls = binary_cls
        if self.binary_cls:
            self.semantic_seg_criterion = BinarySegmentationLoss(
                pos_weight=pos_weight)
        else:
            self.semantic_seg_criterion = SegmentationLoss(
                class_weights=torch.tensor(class_weights).float(),
            )

    def get_semantic_indices(self, predictions, targets=None):
        if self.binary_cls:
            pred_semantic_scores = predictions['semantic_seg'].float(
            ).sigmoid()
            pred_semantic_scores, pred_semantic_indices = torch.max(
                pred_semantic_scores, dim=1)
            background_mask = pred_semantic_scores < self.semantic_thresh
            pred_semantic_indices[background_mask] = 0
            pred_semantic_indices = pred_semantic_indices.long()
        else:
            pred_semantic_logits = predictions['semantic_seg'].clone()
            pred_semantic_indices = torch.argmax(pred_semantic_logits, dim=1)

        return pred_semantic_indices

    @force_fp32(apply_to=('x'))
    def forward(self, x, targets=None):
        x = x[0]
        return {task_key: task_head(x) for task_key, task_head in self.task_heads.items()}

    @force_fp32(apply_to=('predictions'))
    def loss(self, predictions, targets):
        loss_dict = {}

        # computing semanatic_map segmentation
        if self.binary_cls:
            assert predictions['semantic_seg'].shape == targets['semantic_map'].shape
            loss_dict['loss_semantic_seg'] = self.semantic_seg_criterion(
                clip_sigmoid(predictions['semantic_seg'].float()),
                targets['semantic_map'].float(),
            )
        else:
            assert predictions['semantic_seg'].shape[-2:
                                                     ] == targets['semantic_seg'].shape[-2:]
            loss_dict['loss_semantic_seg'] = self.semantic_seg_criterion(
                predictions['semantic_seg'].unsqueeze(dim=1).float(),
                targets['semantic_seg'].unsqueeze(dim=1).long(),
            )

        return loss_dict
