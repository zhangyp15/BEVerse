import torch

from mmdet3d.core import bbox3d2result, merge_aug_bboxes_3d
from mmdet.models import DETECTORS
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models import builder
from ...datasets.utils.geometry import cumulative_warp_features

import pdb
import time

from mmcv.runner import auto_fp16, force_fp32


@DETECTORS.register_module()
class BEVerse(MVXTwoStageDetector):
    """Base class of Multi-modality VoxelNet."""

    def __init__(self,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 transformer=None,
                 temporal_model=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 data_aug_conf=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(BEVerse,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained, init_cfg)

        # building view transformer
        self.transformer = builder.build_neck(transformer)
        # building temporal model
        self.temporal_model = builder.build_neck(temporal_model)

        self.fp16_enabled = False
        self.data_aug_conf = data_aug_conf

    # shared step
    def extract_img_feat(self, img, img_metas, future_egomotion=None,
                         aug_transform=None, img_is_valid=None, count_time=False):
        # image-view feature extraction
        imgs = img[0]

        B, S, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * S * N, C, imH, imW)
        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)

        if isinstance(x, tuple):
            x_list = []
            for x_tmp in x:
                _, output_dim, ouput_H, output_W = x_tmp.shape
                x_list.append(x_tmp.view(B, N, output_dim, ouput_H, output_W))
            x = x_list
        else:
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, S, N, output_dim, ouput_H, output_W)

        # lifting with LSS
        x = self.transformer([x] + img[1:])

        torch.cuda.synchronize()
        t_BEV = time.time()

        # temporal processing
        x = self.temporal_model(x, future_egomotion=future_egomotion,
                                aug_transform=aug_transform, img_is_valid=img_is_valid)

        torch.cuda.synchronize()
        t_temporal = time.time()

        if count_time:
            return x, {'t_BEV': t_BEV, 't_temporal': t_temporal}
        else:
            return x

    def forward_pts_train(self,
                          pts_feats,
                          img_metas,
                          mtl_targets):
        """Forward function for point cloud branch.

        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.

        Returns:
            dict: Losses of each branch.
        """

        # decoders for multi-task
        outs = self.pts_bbox_head(pts_feats, targets=mtl_targets)

        # loss functions for multi-task
        losses = self.pts_bbox_head.loss(predictions=outs, targets=mtl_targets)

        return losses

    @auto_fp16(apply_to=('img_inputs'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img_inputs=None,
                      proposals=None,
                      semantic_map=None,
                      aug_transform=None,
                      future_egomotions=None,
                      motion_segmentation=None,
                      motion_instance=None,
                      instance_centerness=None,
                      instance_offset=None,
                      instance_flow=None,
                      semantic_indices=None,
                      gt_bboxes_ignore=None,
                      img_is_valid=None,
                      has_invalid_frame=None):
        """Forward training function.

        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.

        Returns:
            dict: Losses of different branches.
        """

        img_feats = self.extract_img_feat(
            img=img_inputs,
            img_metas=img_metas,
            future_egomotion=future_egomotions,
            aug_transform=aug_transform,
            img_is_valid=img_is_valid,
        )

        mtl_targets = {
            # for detection
            'gt_bboxes_3d': gt_bboxes_3d,
            'gt_labels_3d': gt_labels_3d,
            'gt_bboxes_ignore': gt_bboxes_ignore,
            # for map segmentation
            'semantic_seg': semantic_indices,
            'semantic_map': semantic_map,
            # for motion prediction
            'motion_segmentation': motion_segmentation,
            'motion_instance': motion_instance,
            'instance_centerness': instance_centerness,
            'instance_offset': instance_offset,
            'instance_flow': instance_flow,
            'future_egomotion': future_egomotions,
            # for bev_augmentation
            'aug_transform': aug_transform,
            'img_is_valid': img_is_valid,
        }

        loss_dict = self.forward_pts_train(img_feats, img_metas, mtl_targets)

        return loss_dict

    def forward_dummy(self, img_inputs=None, **kwargs):
        # generate default img_metas, future_egomotions
        batch_size = 1
        single_frame = False
        input_height = 256

        if input_height == 256:
            dummy_inputs = torch.load('dummy_input.pt')
        else:
            dummy_inputs = torch.load('dummy_input_512.pt')

        future_egomotions = torch.zeros((batch_size, 7, 6)).type_as(img_inputs)
        img_is_valid = torch.ones((batch_size, 7)).type_as(img_inputs) > 0

        if single_frame:
            # [B, T, N, C, H, W]
            future_egomotions = future_egomotions[:, :1]
            img_is_valid = img_is_valid[:, :1]
            for index in range(len(dummy_inputs)):
                dummy_inputs[index] = dummy_inputs[index][:, -1:]

        # lidar2ego transformation
        dummy_lidar2ego_rots = torch.tensor([
            [-5.4280e-04,  9.9893e-01,  4.6229e-02],
            [-1.0000e+00, -4.0569e-04, -2.9750e-03],
            [-2.9531e-03, -4.6231e-02,  9.9893e-01]]).type_as(img_inputs).cpu()
        dummy_lidar2ego_trans = torch.tensor(
            [0.9858, 0.0000, 1.8402]).type_as(img_inputs).cpu()
        from mmdet3d.core.bbox.structures.box_3d_mode import LiDARInstance3DBoxes
        img_metas = [dict(box_type_3d=LiDARInstance3DBoxes,
                          lidar2ego_rots=dummy_lidar2ego_rots, lidar2ego_trans=dummy_lidar2ego_trans)]

        img_feats = self.extract_img_feat(
            img=dummy_inputs, img_metas=img_metas,
            future_egomotion=future_egomotions,
            img_is_valid=img_is_valid,
        )
        predictions = self.simple_test_pts(
            img_feats, img_metas, rescale=True, motion_targets=None)

        return predictions

    def forward_test(self, points=None, img_metas=None, img_inputs=None, future_egomotions=None, **kwargs):
        """
        Args:
            points (list[torch.Tensor]): the outer list indicates test-time
                augmentations and inner torch.Tensor should have a shape NxC,
                which contains all points in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch
            img (list[torch.Tensor], optional): the outer
                list indicates test-time augmentations and inner
                torch.Tensor should have a shape NxCxHxW, which contains
                all images in the batch. Defaults to None.
        """

        for var, name in [(img_inputs, 'img_inputs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))

        num_augs = len(img_inputs)
        if num_augs != len(img_metas):
            raise ValueError(
                'num of augmentations ({}) != num of image meta ({})'.format(
                    len(img_inputs), len(img_metas)))

        if not isinstance(img_inputs[0][0], list):
            img_inputs = [img_inputs] if img_inputs is None else img_inputs
            return self.simple_test(img_metas[0], img_inputs[0], future_egomotions[0], **kwargs)
        else:
            return self.aug_test(img_metas[0], img_inputs[0], future_egomotions[0], **kwargs)

    def simple_test(self, img_metas, img=None, future_egomotions=None,
                    rescale=False, motion_targets=None, img_is_valid=None):
        """Test function without augmentaiton."""
        torch.cuda.synchronize()
        t0 = time.time()

        img_feats, time_stats = self.extract_img_feat(
            img=img, img_metas=img_metas,
            future_egomotion=future_egomotions,
            img_is_valid=img_is_valid,
            count_time=True,
        )
        time_stats['t0'] = t0

        predictions = self.simple_test_pts(
            img_feats, img_metas, rescale=rescale, motion_targets=motion_targets)

        torch.cuda.synchronize()
        time_stats['t_end'] = time.time()

        if 'bbox_results' in predictions:
            bbox_list = [dict() for i in range(len(img_metas))]
            for result_dict, pts_bbox in zip(bbox_list, predictions['bbox_results']):
                result_dict['pts_bbox'] = pts_bbox
            predictions['bbox_results'] = bbox_list

        predictions['time_stats'] = time_stats

        return predictions

    def simple_test_pts(self, x, img_metas, rescale=False, motion_targets=None):
        """Test function of point cloud branch."""
        outs = self.pts_bbox_head(x, targets=motion_targets)

        predictions = self.pts_bbox_head.inference(
            outs, img_metas, rescale=rescale,
        )

        # convert bbox predictions
        if 'bbox_list' in predictions:
            bbox_list = predictions.pop('bbox_list')
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]
            predictions['bbox_results'] = bbox_results

        return predictions

    def aug_test(self, img_metas, img=None, future_egomotions=None,
                 rescale=False, motion_targets=None, img_is_valid=None):
        '''
        Test function with augmentaiton
        1. forward with different image-view & bev-view augmentations
        2. combine dense outputs with averaging
        '''

        flip_aug_mask = img_metas[0]['flip_aug']
        scale_aug_mask = img_metas[0]['scale_aug']
        assert len(flip_aug_mask) == len(scale_aug_mask) == len(img)

        img_aug_outs = []
        for i in range(len(img)):
            tta_flip_bev = self.data_aug_conf.get('tta_flip_bev', False)
            if tta_flip_bev:
                # 前后、左右两种翻转方式构成四种组合
                choices = [False, True]
                flip_bev_list = [[d, f] for d in choices for f in choices]
            else:
                flip_bev_list = [[False, False], ]

            bev_outputs = []
            for flip_bev in flip_bev_list:
                # backbone + view-transformer + temporal model
                bev_feat = self.extract_img_feat_tta(
                    img=img[i], img_metas=img_metas,
                    future_egomotion=future_egomotions,
                    img_is_valid=img_is_valid,
                    flip_x=flip_bev[0],
                    flip_y=flip_bev[1],
                )
                # decoder head
                bev_output = self.pts_bbox_head(
                    bev_feat, targets=motion_targets)
                # flip-back outputs
                bev_output = self.flip_bev_output(
                    bev_output, flip_x=flip_bev[0], flip_y=flip_bev[1])
                bev_outputs.append(bev_output)

            averaged_bev_output = self.combine_bev_output(bev_outputs)
            img_aug_outs.append(averaged_bev_output)

        bev_output = self.combine_bev_output(img_aug_outs)
        predictions = self.pts_bbox_head.inference(
            bev_output, img_metas, rescale=rescale,
        )

        if 'bbox_list' in predictions:
            bbox_list = predictions.pop('bbox_list')
            bbox_results = [
                bbox3d2result(bboxes, scores, labels)
                for bboxes, scores, labels in bbox_list
            ]

            bbox_list = [dict() for i in range(len(img_metas))]
            for result_dict, pts_bbox in zip(bbox_list, bbox_results):
                result_dict['pts_bbox'] = pts_bbox
            predictions['bbox_results'] = bbox_list

        return predictions

    def flip_feature(self, feature, flip_x=False, flip_y=False):
        if flip_x:
            feature = torch.flip(feature, dims=[-1])
        if flip_y:
            feature = torch.flip(feature, dims=[-2])

        return feature

    def flip_bev_output(self, output, flip_x=False, flip_y=False):
        # flip map
        map_key = 'map'
        for pred_key, pred_val in output[map_key].items():
            output[map_key][pred_key][...] = self.flip_feature(pred_val,
                                                               flip_x=flip_x, flip_y=flip_y)

        # flip 3d detection
        det_key = '3dod'
        for head_id in range(len(output[det_key])):
            for key in output[det_key][0][0]:
                output[det_key][head_id][0][key][...] = self.flip_feature(
                    output[det_key][head_id][0][key], flip_x=flip_x, flip_y=flip_y)

                if key in ['rot', 'vel']:
                    if flip_x:
                        output[det_key][head_id][0][key][:, 0] = - \
                            output[det_key][head_id][0][key][:, 0]

                    if flip_y:
                        output[det_key][head_id][0][key][:, 1] = - \
                            output[det_key][head_id][0][key][:, 1]

                elif key == 'reg':
                    if flip_x:
                        output[det_key][head_id][0][key][:, 0] = 1.0 - \
                            output[det_key][head_id][0][key][:, 0]

                    if flip_y:
                        output[det_key][head_id][0][key][:, 1] = 1.0 - \
                            output[det_key][head_id][0][key][:, 1]

        # flip motion
        motion_key = 'motion'
        for pred_key, pred_val in output[motion_key].items():
            # offset 和 flow 都需要 flip
            output[motion_key][pred_key][...] = self.flip_feature(pred_val,
                                                                  flip_x=flip_x, flip_y=flip_y)
            if pred_key in ['instance_offset', 'instance_flow']:
                if flip_x:
                    output[motion_key][pred_key][:, :, 0] = - \
                        output[motion_key][pred_key][:, :, 0]
                if flip_y:
                    output[motion_key][pred_key][:, :, 1] = - \
                        output[motion_key][pred_key][:, :, 1]

        return output

    def combine_bev_output(self, bev_outputs):
        for task_key in bev_outputs[0]:
            if task_key != '3dod':
                for pred_key in bev_outputs[0][task_key]:
                    bev_outputs[0][task_key][pred_key] = torch.mean(torch.cat(
                        [x[task_key][pred_key] for x in bev_outputs], dim=0), dim=0, keepdim=True)
                continue

            for head_id in range(len(bev_outputs[0][task_key])):
                for pred_key in bev_outputs[0][task_key][0][0]:
                    bev_outputs[0][task_key][head_id][0][pred_key] = torch.mean(torch.cat(
                        [x[task_key][head_id][0][pred_key] for x in bev_outputs], dim=0), dim=0, keepdim=True)

        return bev_outputs[0]

    def extract_img_feat_tta(self, img, img_metas, future_egomotion=None,
                             aug_transform=None, img_is_valid=None, flip_x=False, flip_y=False):
        # image-view feature extraction
        imgs = img[0]

        B, S, N, C, imH, imW = imgs.shape
        imgs = imgs.view(B * S * N, C, imH, imW)
        x = self.img_backbone(imgs)

        if self.with_img_neck:
            x = self.img_neck(x)

        if isinstance(x, tuple):
            x_list = []
            for x_tmp in x:
                _, output_dim, ouput_H, output_W = x_tmp.shape
                x_list.append(x_tmp.view(B, N, output_dim, ouput_H, output_W))
            x = x_list
        else:
            _, output_dim, ouput_H, output_W = x.shape
            x = x.view(B, S, N, output_dim, ouput_H, output_W)

        # lifting with LSS
        x = self.transformer([x] + img[1:], flip_x=flip_x, flip_y=flip_y)

        # temporal processing
        x = self.temporal_model(x, future_egomotion=future_egomotion,
                                aug_transform=aug_transform, img_is_valid=img_is_valid)

        return x
