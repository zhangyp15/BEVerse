import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import HEADS

from ..motion_modules import FuturePrediction
from ._base_motion_head import BaseMotionHead

import pdb


@HEADS.register_module()
class FieryMotionHead(BaseMotionHead):
    def __init__(
        self,
        detach_state=False,
        n_gru_blocks=3,
        n_res_layers=3,
        **kwargs,
    ):
        super(FieryMotionHead, self).__init__(**kwargs)

        if self.n_future > 0:
            self.future_prediction = FuturePrediction(
                in_channels=self.in_channels,
                latent_dim=self.prob_latent_dim,
                n_gru_blocks=n_gru_blocks,
                n_res_layers=n_res_layers,
            )

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
