import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models import builder
from mmcv.cnn import build_norm_layer
from mmdet3d.models.builder import HEADS, build_loss

from .bev_encoder import BevEncode
from .map_head import BevFeatureSlicer
from mmcv.runner import auto_fp16, force_fp32

import pdb


@HEADS.register_module()
class MultiTaskHead(BaseModule):
    def __init__(self, init_cfg=None, task_enbale=None, task_weights=None,
                 in_channels=64,
                 out_channels=256,
                 bev_encode_block='BottleNeck',
                 bev_encoder_type='resnet18',
                 bev_encode_depth=[2, 2, 2],
                 num_channels=None,
                 backbone_output_ids=None,
                 norm_cfg=dict(type='BN'),
                 bev_encoder_fpn_type='lssfpn',
                 grid_conf=None,
                 det_grid_conf=None,
                 map_grid_conf=None,
                 motion_grid_conf=None,
                 out_with_activision=False,
                 using_ego=False,
                 shared_feature=False,
                 cfg_3dod=None,
                 cfg_map=None,
                 cfg_motion=None,
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(MultiTaskHead, self).__init__(init_cfg)

        self.fp16_enabled = False
        self.task_enbale = task_enbale
        self.task_weights = task_weights
        self.using_ego = using_ego

        if det_grid_conf is None:
            det_grid_conf = grid_conf

        # build task-features
        self.task_names_ordered = ['map', '3dod', 'motion']
        self.taskfeat_encoders = nn.ModuleDict()
        assert bev_encoder_type == 'resnet18'

        # whether to use shared features
        self.shared_feature = shared_feature
        if self.shared_feature:
            self.taskfeat_encoders['shared'] = BevEncode(
                numC_input=in_channels,
                numC_output=out_channels,
                num_channels=num_channels,
                backbone_output_ids=backbone_output_ids,
                num_layer=bev_encode_depth,
                bev_encode_block=bev_encode_block,
                norm_cfg=norm_cfg,
                bev_encoder_fpn_type=bev_encoder_fpn_type,
                out_with_activision=out_with_activision,
            )
        else:
            for task_name in self.task_names_ordered:
                is_enable = task_enbale.get(task_name, False)
                if not is_enable:
                    continue

                self.taskfeat_encoders[task_name] = BevEncode(
                    numC_input=in_channels,
                    numC_output=out_channels,
                    num_channels=num_channels,
                    backbone_output_ids=backbone_output_ids,
                    num_layer=bev_encode_depth,
                    bev_encode_block=bev_encode_block,
                    norm_cfg=norm_cfg,
                    bev_encoder_fpn_type=bev_encoder_fpn_type,
                    out_with_activision=out_with_activision,
                )

        # build task-decoders
        self.task_decoders = nn.ModuleDict()
        self.task_feat_cropper = nn.ModuleDict()

        # 3D object detection
        if task_enbale.get('3dod', False):
            cfg_3dod.update(train_cfg=train_cfg)
            cfg_3dod.update(test_cfg=test_cfg)

            self.task_feat_cropper['3dod'] = BevFeatureSlicer(
                grid_conf, det_grid_conf)
            self.task_decoders['3dod'] = builder.build_head(cfg_3dod)

        # static map
        if task_enbale.get('map', False):
            cfg_map.update(train_cfg=train_cfg)
            cfg_map.update(test_cfg=test_cfg)

            self.task_feat_cropper['map'] = BevFeatureSlicer(
                grid_conf, map_grid_conf)
            self.task_decoders['map'] = builder.build_head(cfg_map)

        # motion_head
        if task_enbale.get('motion', False):
            cfg_motion.update(train_cfg=train_cfg)
            cfg_motion.update(test_cfg=test_cfg)

            self.task_feat_cropper['motion'] = BevFeatureSlicer(
                grid_conf, motion_grid_conf)
            self.task_decoders['motion'] = builder.build_head(cfg_motion)

    def scale_task_losses(self, task_name, task_loss_dict):
        task_sum = 0
        for key, val in task_loss_dict.items():
            task_sum += val.item()
            task_loss_dict[key] = val * self.task_weights.get(task_name, 1.0)

        task_loss_summation = sum(list(task_loss_dict.values()))
        task_loss_dict['{}_sum'.format(task_name)] = task_loss_summation

        return task_loss_dict

    def loss(self, predictions, targets):
        loss_dict = {}

        if self.task_enbale.get('3dod', False):
            det_loss_dict = self.task_decoders['3dod'].loss(
                gt_bboxes_3d=targets['gt_bboxes_3d'],
                gt_labels_3d=targets['gt_labels_3d'],
                preds_dicts=predictions['3dod'],
            )
            loss_dict.update(self.scale_task_losses(
                task_name='3dod', task_loss_dict=det_loss_dict))

        if self.task_enbale.get('map', False):
            map_loss_dict = self.task_decoders['map'].loss(
                predictions['map'], targets,
            )
            loss_dict.update(self.scale_task_losses(
                task_name='map', task_loss_dict=map_loss_dict))

        if self.task_enbale.get('motion', False):
            motion_loss_dict = self.task_decoders['motion'].loss(
                predictions['motion'])
            loss_dict.update(self.scale_task_losses(
                task_name='motion', task_loss_dict=motion_loss_dict))

        return loss_dict

    def inference(self, predictions, img_metas, rescale):
        res = {}
        # derive bounding boxes for detection head
        if self.task_enbale.get('3dod', False):
            res['bbox_list'] = self.task_decoders['3dod'].get_bboxes(
                predictions['3dod'],
                img_metas=img_metas,
                rescale=rescale
            )

            # convert predicted boxes in ego to LiDAR coordinates
            if self.using_ego:
                for index, (bboxes, scores, labels) in enumerate(res['bbox_list']):
                    img_meta = img_metas[index]
                    lidar2ego_rot, lidar2ego_tran = img_meta['lidar2ego_rots'], img_meta['lidar2ego_trans']

                    bboxes = bboxes.to('cpu')
                    bboxes.translate(-lidar2ego_tran)
                    bboxes.rotate(lidar2ego_rot.t().inverse().float())

                    res['bbox_list'][index] = (bboxes, scores, labels)

        # derive semantic maps for map head
        if self.task_enbale.get('map', False):
            res['pred_semantic_indices'] = self.task_decoders['map'].get_semantic_indices(
                predictions['map'],
            )

        if self.task_enbale.get('motion', False):
            seg_prediction, pred_consistent_instance_seg = self.task_decoders['motion'].inference(
                predictions['motion'],
            )

            res['motion_predictions'] = predictions['motion']
            res['motion_segmentation'] = seg_prediction
            res['motion_instance'] = pred_consistent_instance_seg

        return res

    def forward_with_shared_features(self, bev_feats, targets=None):
        predictions = {}
        auxiliary_features = {}

        bev_feats = self.taskfeat_encoders['shared']([bev_feats])

        for task_name in self.task_feat_cropper:
            # crop feature before the encoder
            task_feat = self.task_feat_cropper[task_name](bev_feats)
            
            # task-specific decoder
            if task_name == 'motion':
                task_pred = self.task_decoders[task_name](
                    [task_feat], targets=targets)
            else:
                task_pred = self.task_decoders[task_name]([task_feat])
            
            predictions[task_name] = task_pred

        return predictions

    def forward(self, bev_feats, targets=None):
        if self.shared_feature:
            return self.forward_with_shared_features(bev_feats, targets)

        predictions = {}
        for task_name, task_feat_encoder in self.taskfeat_encoders.items():

            # crop feature before the encoder
            task_feat = self.task_feat_cropper[task_name](bev_feats)

            # task-specific feature encoder
            task_feat = task_feat_encoder([task_feat])

            # task-specific decoder
            if task_name == 'motion':
                task_pred = self.task_decoders[task_name](
                    [task_feat], targets=targets)
            else:
                task_pred = self.task_decoders[task_name]([task_feat])
            
            predictions[task_name] = task_pred

        return predictions
