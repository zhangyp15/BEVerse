from matplotlib.pyplot import autoscale
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import build_loss
from mmdet3d.models.utils import clip_sigmoid

import pdb
from mmcv.runner import auto_fp16, force_fp32


class BinarySegmentationLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(BinarySegmentationLoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt):
        loss = self.loss_fn(ypred, ytgt)

        return loss


class SegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0):

        super().__init__()

        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

        # self.ce_criterion = nn.CrossEntropyLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

        # self.nll_criterion = nn.NLLLoss(
        #     weight=self.class_weights, ignore_index=self.ignore_index, reduction='mean')

    def forward(self, prediction, target):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        loss = F.cross_entropy(
            prediction,
            target,
            reduction='none',
            weight=self.class_weights.to(target.device).float(),
        )

        # ce_loss = self.ce_criterion(prediction, target)
        # pred_logsoftmax = F.log_softmax(prediction)
        # loss = self.nll_criterion(pred_logsoftmax, target)

        loss = loss.view(b, s, h, w)
        future_discounts = self.future_discount ** torch.arange(
            s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s, 1, 1)
        loss = loss * future_discounts.float()

        loss = loss.view(b, s, -1)
        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[2])
            loss, _ = torch.sort(loss, dim=2, descending=True)
            loss = loss[:, :, :k]

        return torch.mean(loss)

class MotionSegmentationLoss(nn.Module):
    def __init__(self, class_weights, ignore_index=255, use_top_k=False,
                 top_k_ratio=1.0, future_discount=1.0):

        super().__init__()

        self.class_weights = class_weights
        self.ignore_index = ignore_index
        self.use_top_k = use_top_k
        self.top_k_ratio = top_k_ratio
        self.future_discount = future_discount

    def forward(self, prediction, target, frame_mask=None):
        b, s, c, h, w = prediction.shape

        prediction = prediction.view(b * s, c, h, w)
        target = target.view(b * s, h, w)

        future_discounts = self.future_discount ** torch.arange(
            s).type_as(prediction)
        future_discounts = future_discounts.view(1, s).repeat(b, 1)
        future_discounts = future_discounts.view(-1, 1)

        frame_mask = frame_mask.contiguous().view(-1)
        valid_prediction = prediction[frame_mask]
        valid_target = target[frame_mask]
        future_discounts = future_discounts[frame_mask]

        if frame_mask.sum().item() == 0:
            return prediction.abs().sum().float() * 0.0

        loss = F.cross_entropy(
            valid_prediction,
            valid_target,
            ignore_index=self.ignore_index,
            reduction='none',
            weight=self.class_weights.to(target.device),
        )
        loss = loss.flatten(start_dim=1)
        loss *= future_discounts

        if self.use_top_k:
            # Penalises the top-k hardest pixels
            k = int(self.top_k_ratio * loss.shape[1])
            loss, _ = torch.sort(loss, dim=1, descending=True)
            loss = loss[:, :k]

        return torch.mean(loss)


class GaussianFocalLoss(nn.Module):
    def __init__(
        self,
        focal_cfg=dict(type='GaussianFocalLoss', reduction='none'),
        ignore_index=255,
        future_discount=1.0,
    ):
        super().__init__()

        self.gaussian_focal_loss = build_loss(focal_cfg)
        self.ignore_index = ignore_index
        self.future_discount = future_discount

    def clip_sigmoid(self, x, eps=1e-4):
        """Sigmoid function for input feature.

        Args:
            x (torch.Tensor): Input feature map with the shape of [B, N, H, W].
            eps (float): Lower bound of the range to be clamped to. Defaults
                to 1e-4.

        Returns:
            torch.Tensor: Feature map after sigmoid.
        """
        y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)

        return y

    def forward(self, prediction, target, frame_mask=None):
        b, s, c, h, w = prediction.shape
        prediction = prediction.view(b * s, h, w)
        target = target.view(b * s, h, w)

        assert frame_mask is not None
        frame_mask = frame_mask.contiguous().view(-1)
        valid_pred, valid_target = prediction[frame_mask], target[frame_mask]

        valid_pred = clip_sigmoid(valid_pred.float())
        loss = self.gaussian_focal_loss(
            pred=valid_pred, target=valid_target,
        )

        # compute discounts & normalizer
        future_discounts = self.future_discount ** torch.arange(
            s, device=loss.device, dtype=loss.dtype)
        future_discounts = future_discounts.view(1, s).repeat(b, 1).view(-1)
        future_discounts = future_discounts[frame_mask]

        num_pos = valid_target.eq(1).float().flatten(1).sum(1)
        num_pos *= future_discounts

        # multiply loss & future_discounts
        loss = loss * future_discounts.view(-1, 1, 1)
        loss = loss.sum() / torch.clamp_min(num_pos.sum(), 1.0)

        return loss


class SpatialRegressionLoss(nn.Module):
    def __init__(self, norm, ignore_index=255, future_discount=1.0):
        super(SpatialRegressionLoss, self).__init__()
        self.norm = norm
        self.ignore_index = ignore_index
        self.future_discount = future_discount
        # self.fp16_enabled = False

        if norm == 1:
            self.loss_fn = F.l1_loss
        elif norm == 2:
            self.loss_fn = F.mse_loss
        else:
            raise ValueError(f'Expected norm 1 or 2, but got norm={norm}')

    def forward(self, prediction, target, frame_mask=None):
        assert len(prediction.shape) == 5, 'Must be a 5D tensor'
        # frame filter
        assert frame_mask is not None
        # [b * s, c, h, w]
        b, s = prediction.shape[:2]
        prediction = prediction[frame_mask]
        target = target[frame_mask]
        future_discounts = self.future_discount ** torch.arange(
            s).type_as(prediction)
        future_discounts = future_discounts.view(1, s).repeat(b, 1)
        future_discounts = future_discounts[frame_mask]

        # reg filter
        reg_mask = torch.all(target != self.ignore_index, dim=1)

        if reg_mask.sum() == 0:
            return prediction.abs().sum().float() * 0.0

        # [num_frame, num_channel, h, w]
        loss = self.loss_fn(prediction, target, reduction='none')

        # Sum channel dimension
        loss = torch.sum(loss, dim=1)

        # multiply discounts
        loss *= future_discounts.view(-1, 1, 1)
        loss = loss[reg_mask].mean()

        return loss


class SpatialProbabilisticLoss(nn.Module):
    def __init__(self, foreground=False, bidirectional=False):
        super().__init__()

        self.foreground = foreground
        self.bidirectional = bidirectional

    def forward(self, output, foreground_mask=None, batch_valid_mask=None):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = future_log_sigma.exp()
        var_present = present_log_sigma.exp()

        kl_div = present_log_sigma - future_log_sigma - 1 + \
            ((future_mu - present_mu) ** 2 + var_future) / var_present
        kl_div *= 0.5

        # summation along the channels
        kl_loss = torch.sum(kl_div, dim=1)

        # [batch, sequence]
        kl_loss = kl_loss[batch_valid_mask]

        if self.foreground:
            assert foreground_mask is not None
            foreground_mask = (
                foreground_mask[batch_valid_mask].sum(dim=1) > 0).float()

            foreground_mask = F.interpolate(
                foreground_mask, size=kl_loss.shape[-2:], mode='nearest').squeeze(1)
            kl_loss = kl_loss[foreground_mask.bool()]

        # computing the distribution loss only for samples with complete temporal completeness
        if kl_loss.numel() > 0:
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = (kl_loss * 0).sum().float()

        return kl_loss


class ProbabilisticLoss(nn.Module):
    def __init__(self, foreground=False):
        super().__init__()

        self.foreground = foreground

    def forward(self, output, foreground_mask=None, batch_valid_mask=None):
        present_mu = output['present_mu']
        present_log_sigma = output['present_log_sigma']
        future_mu = output['future_mu']
        future_log_sigma = output['future_log_sigma']

        var_future = torch.exp(2 * future_log_sigma)
        var_present = torch.exp(2 * present_log_sigma)
        kl_div = (
            present_log_sigma - future_log_sigma - 0.5 + (var_future + (future_mu - present_mu) ** 2) / (
                2 * var_present)
        )

        if kl_div.ndim == 4:
            kl_loss = torch.sum(kl_div, dim=1)

        elif kl_div.ndim == 3:
            kl_loss = torch.sum(kl_div, dim=-1)

        # [batch, sequence]
        kl_loss = kl_loss[batch_valid_mask]

        if self.foreground:
            assert foreground_mask is not None
            foreground_mask = (
                foreground_mask[batch_valid_mask].sum(dim=1) > 0).float()

            foreground_mask = F.interpolate(
                foreground_mask, size=kl_loss.shape[-2:], mode='nearest').squeeze(1)
            kl_loss = kl_loss[foreground_mask.bool()]

        # computing the distribution loss only for samples with complete temporal completeness
        if kl_loss.numel() > 0:
            kl_loss = torch.mean(kl_loss)
        else:
            kl_loss = (kl_loss * 0).sum().float()

        return kl_loss
