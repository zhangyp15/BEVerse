import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS
from .base_taskhead import BaseTaskHead
from .loss_utils import MotionSegmentationLoss, SpatialRegressionLoss, ProbabilisticLoss, GaussianFocalLoss
from ...datasets.utils.geometry import cumulative_warp_features_reverse
from ...datasets.utils.instance import predict_instance_segmentation_and_trajectories
from ...datasets.utils.warper import FeatureWarper

from ...visualize import Visualizer
from ..motion_modules import DistributionModule, FuturePrediction
from mmcv.runner import auto_fp16, force_fp32

import pdb


@HEADS.register_module()
class MotionHead(BaseTaskHead):
    def __init__(
        self,
        task_dict,
        in_channels,
        class_weights,
        use_topk=True,
        topk_ratio=0.25,
        grid_conf=None,
        receptive_field=3,
        n_future=0,
        future_discount=0.95,
        ignore_index=255,
        probabilistic_enable=True,
        future_dim=6,
        prob_latent_dim=32,
        distribution_log_sigmas=None,
        n_gru_blocks=3,
        n_res_layers=3,
        using_focal_loss=False,
        sample_ignore_mode='all_valid',
        focal_cfg=dict(type='GaussianFocalLoss', reduction='none'),
        loss_weights=None,
        train_cfg=None,
        test_cfg=None,
        init_cfg=None,
        norm_cfg=dict(type='BN'),
        **kwargs,
    ):
        super(MotionHead, self).__init__(
            task_dict, in_channels, init_cfg, norm_cfg)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.in_channels = in_channels
        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])
        self.receptive_field = receptive_field
        self.n_future = n_future
        self.probabilistic_enable = probabilistic_enable
        self.prob_latent_dim = prob_latent_dim
        self.future_dim = future_dim
        self.loss_weights = loss_weights
        self.ignore_index = ignore_index
        self.using_focal_loss = using_focal_loss

        self.sample_ignore_mode = sample_ignore_mode
        assert self.sample_ignore_mode in ['all_valid', 'past_valid', 'none']

        self.visualizer = Visualizer(out_dir='train_visualize')
        self.warper = FeatureWarper(grid_conf=grid_conf)

        if self.n_future > 0:
            distri_min_log_sigma, distri_max_log_sigma = distribution_log_sigmas

            self.present_distribution = DistributionModule(
                self.in_channels,
                self.prob_latent_dim,
                min_log_sigma=distri_min_log_sigma,
                max_log_sigma=distri_max_log_sigma,
            )

            future_distribution_in_channels = self.in_channels + self.n_future * future_dim

            self.future_distribution = DistributionModule(
                future_distribution_in_channels,
                self.prob_latent_dim,
                min_log_sigma=distri_min_log_sigma,
                max_log_sigma=distri_max_log_sigma,
            )

            self.future_prediction = FuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_gru_blocks=n_gru_blocks,
                n_res_layers=n_res_layers,
            )

        # loss functions
        # 1. loss for foreground segmentation
        self.seg_criterion = MotionSegmentationLoss(
            class_weights=torch.tensor(class_weights),
            use_top_k=use_topk,
            top_k_ratio=topk_ratio,
            future_discount=future_discount,
        )
        # 2. loss for instance center heatmap
        self.reg_instance_center_criterion = SpatialRegressionLoss(
            norm=2,
            future_discount=future_discount,
        )

        self.cls_instance_center_criterion = GaussianFocalLoss(
            focal_cfg=focal_cfg,
            ignore_index=ignore_index,
            future_discount=future_discount,
        )

        # 3. loss for instance offset
        self.reg_instance_offset_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        # 4. loss for instance flow
        self.reg_instance_flow_criterion = SpatialRegressionLoss(
            norm=1,
            future_discount=future_discount,
            ignore_index=ignore_index,
        )

        # 5. loss for probabilistic distribution
        self.probabilistic_loss = ProbabilisticLoss()

    def forward(self, bevfeats, targets=None, noise=None):
        '''
        the forward process of motion head:
        1. get present & future distributions
        2. iteratively get future states with ConvGRU
        3. decode present & future states with the decoder heads
        '''

        bevfeats = bevfeats[0]

        if self.training:
            self.training_labels, future_distribution_inputs = self.prepare_future_labels(
                targets)
        else:
            future_distribution_inputs = None

        res = {}
        if self.n_future > 0:
            present_state = bevfeats.unsqueeze(dim=1).contiguous()

            # sampling probabilistic distribution
            sample, output_distribution = self.distribution_forward(
                present_state, future_distribution_inputs, noise
            )

            b, _, _, h, w = present_state.shape
            hidden_state = present_state[:, 0]

            # 采样的 sample，直接应用于每个 future frame
            future_prediction_input = sample.expand(
                -1, self.n_future, -1, -1, -1)
            
            # expand sample: [b, 1, c, h, w] ==> [b, num_future, c, h, w]
            future_states = self.future_prediction(
                future_prediction_input, hidden_state)
            future_states = torch.cat([present_state, future_states], dim=1)
            # flatten dimensions of (batch, sequence)
            batch, seq = future_states.shape[:2]
            flatten_states = future_states.flatten(0, 1)

            res.update(output_distribution)
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(
                    flatten_states).view(batch, seq, -1, h, w)
        else:
            b, _, h, w = bevfeats.shape
            for task_key, task_head in self.task_heads.items():
                res[task_key] = task_head(bevfeats).view(b, 1, -1, h, w)

        return res

    def distribution_forward(
        self,
        present_features,
        future_distribution_inputs=None,
        noise=None,
    ):
        """
        Parameters
        ----------
            present_features: 5-D output from dynamics module with shape (b, 1, c, h, w)
            future_distribution_inputs: 5-D tensor containing labels shape (b, s, cfg.PROB_FUTURE_DIM, h, w)
            noise: a sample from a (0, 1) gaussian with shape (b, s, latent_dim). If None, will sample in function

        Returns
        -------
            sample: sample taken from present/future distribution, broadcast to shape (b, s, latent_dim, h, w)
            present_distribution_mu: shape (b, s, latent_dim)
            present_distribution_log_sigma: shape (b, s, latent_dim)
            future_distribution_mu: shape (b, s, latent_dim)
            future_distribution_log_sigma: shape (b, s, latent_dim)
        """
        b, s, _, h, w = present_features.size()
        assert s == 1

        present_mu, present_log_sigma = self.present_distribution(
            present_features)

        future_mu, future_log_sigma = None, None
        if future_distribution_inputs is not None:
            # Concatenate future labels to z_t
            future_features = future_distribution_inputs[:, 1:].contiguous().view(
                b, 1, -1, h, w)
            future_features = torch.cat(
                [present_features, future_features], dim=2)
            future_mu, future_log_sigma = self.future_distribution(
                future_features)

        if noise is None:
            if self.training:
                noise = torch.randn_like(present_mu)
            else:
                noise = torch.zeros_like(present_mu)

        if self.training:
            mu = future_mu
            sigma = torch.exp(future_log_sigma)
        else:
            mu = present_mu
            sigma = torch.exp(present_log_sigma)
        sample = mu + sigma * noise

        # Spatially broadcast sample to the dimensions of present_features
        sample = sample.view(b, s, self.prob_latent_dim, 1, 1).expand(
            b, s, self.prob_latent_dim, h, w)

        output_distribution = {
            'present_mu': present_mu,
            'present_log_sigma': present_log_sigma,
            'future_mu': future_mu,
            'future_log_sigma': future_log_sigma,
        }

        return sample, output_distribution

    def prepare_future_labels(self, batch):
        labels = {}
        future_distribution_inputs = []

        segmentation_labels = batch['motion_segmentation']
        instance_center_labels = batch['instance_centerness']
        instance_offset_labels = batch['instance_offset']
        instance_flow_labels = batch['instance_flow']
        gt_instance = batch['motion_instance']
        future_egomotion = batch['future_egomotion']
        bev_transform = batch.get('aug_transform', None)
        labels['img_is_valid'] = batch.get('img_is_valid', None)

        if bev_transform is not None:
            bev_transform = bev_transform.float()

        segmentation_labels = self.warper.cumulative_warp_features_reverse(
            segmentation_labels.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()
        labels['segmentation'] = segmentation_labels
        future_distribution_inputs.append(segmentation_labels)

        # Warp instance labels to present's reference frame
        gt_instance = self.warper.cumulative_warp_features_reverse(
            gt_instance.float().unsqueeze(2),
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).long().contiguous()[:, :, 0]
        labels['instance'] = gt_instance

        instance_center_labels = self.warper.cumulative_warp_features_reverse(
            instance_center_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['centerness'] = instance_center_labels

        instance_offset_labels = self.warper.cumulative_warp_features_reverse(
            instance_offset_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['offset'] = instance_offset_labels

        future_distribution_inputs.append(instance_center_labels)
        future_distribution_inputs.append(instance_offset_labels)

        instance_flow_labels = self.warper.cumulative_warp_features_reverse(
            instance_flow_labels,
            future_egomotion[:, (self.receptive_field - 1):],
            mode='nearest', bev_transform=bev_transform,
        ).contiguous()
        labels['flow'] = instance_flow_labels
        future_distribution_inputs.append(instance_flow_labels)

        if len(future_distribution_inputs) > 0:
            future_distribution_inputs = torch.cat(
                future_distribution_inputs, dim=2)

        # self.visualizer.visualize_motion(labels=labels)
        # pdb.set_trace()

        return labels, future_distribution_inputs

    @force_fp32(apply_to=('predictions'))
    def loss(self, predictions, targets=None):
        loss_dict = {}

        '''
        prediction dict:
            'segmentation': 2,
            'instance_center': 1,
            'instance_offset': 2,
            'instance_flow': 2,
        '''

        for key, val in self.training_labels.items():
            self.training_labels[key] = val.float()

        frame_valid_mask = self.training_labels['img_is_valid'].bool()
        past_valid_mask = frame_valid_mask[:, :self.receptive_field]
        future_frame_mask = frame_valid_mask[:, (self.receptive_field - 1):]

        if self.sample_ignore_mode is 'all_valid':
            # only supervise when all 7 frames are valid
            batch_valid_mask = frame_valid_mask.all(dim=1)
            future_frame_mask[~batch_valid_mask] = False
            prob_valid_mask = batch_valid_mask

        elif self.sample_ignore_mode is 'past_valid':
            # only supervise when past 3 frames are valid
            past_valid = torch.all(past_valid_mask, dim=1)
            future_frame_mask[~past_valid] = False
            prob_valid_mask = past_valid

        elif self.sample_ignore_mode is 'none':
            prob_valid_mask = frame_valid_mask.any(dim=1)

        # segmentation
        loss_dict['loss_motion_seg'] = self.seg_criterion(
            predictions['segmentation'], self.training_labels['segmentation'].long(),
            frame_mask=future_frame_mask,
        )

        # instance centerness, but why not focal loss
        if self.using_focal_loss:
            loss_dict['loss_motion_centerness'] = self.cls_instance_center_criterion(
                predictions['instance_center'], self.training_labels['centerness'],
                frame_mask=future_frame_mask,
            )
        else:
            loss_dict['loss_motion_centerness'] = self.reg_instance_center_criterion(
                predictions['instance_center'], self.training_labels['centerness'],
                frame_mask=future_frame_mask,
            )

        # instance offset
        loss_dict['loss_motion_offset'] = self.reg_instance_offset_criterion(
            predictions['instance_offset'], self.training_labels['offset'],
            frame_mask=future_frame_mask,
        )

        if self.n_future > 0:
            # instance flow
            loss_dict['loss_motion_flow'] = self.reg_instance_flow_criterion(
                predictions['instance_flow'], self.training_labels['flow'],
                frame_mask=future_frame_mask,
            )

            # temporal probabilistic distribution
            loss_dict['loss_motion_prob'] = self.probabilistic_loss(
                predictions,
                batch_valid_mask=prob_valid_mask,
            )

        for key in loss_dict:
            loss_dict[key] *= self.loss_weights.get(key, 1.0)

        return loss_dict

    def inference(self, predictions):
        # [b, s, num_cls, h, w]
        seg_prediction = torch.argmax(
            predictions['segmentation'], dim=2, keepdims=True)

        if self.using_focal_loss:
            predictions['instance_center'] = torch.sigmoid(
                predictions['instance_center'])

        pred_consistent_instance_seg = predict_instance_segmentation_and_trajectories(
            predictions, compute_matched_centers=False,
        )

        return seg_prediction, pred_consistent_instance_seg
