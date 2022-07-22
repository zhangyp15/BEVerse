# Copyright (c) Junjie.huang. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from mmdet3d.models.builder import NECKS
from mmdet3d.models import builder
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_norm_layer
from mmdet3d.models.backbones.swin import SwinTransformer
from mmdet3d.ops.bev_pool import bev_pool

import pdb


@NECKS.register_module()
class TransformerLSS(BaseModule):
    def __init__(self, grid_conf=None, input_dim=None, init_cfg=None, numC_input=512,
                 numC_Trans=512, downsample=16, faster=False, use_bev_pool=True, **kwargs):

        super(TransformerLSS, self).__init__(init_cfg)
        if grid_conf is None:
            grid_conf = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [4.0, 45.0, 1.0], }

        self.grid_conf = grid_conf
        self.dx, self.bx, self.nx = gen_dx_bx(self.grid_conf['xbound'],
                                              self.grid_conf['ybound'],
                                              self.grid_conf['zbound'],
                                              )

        self.input_dim = input_dim
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(
            self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)

        self.use_bev_pool = use_bev_pool

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.input_dim
        fH, fW = ogfH // self.downsample, ogfW // self.downsample
        ds = torch.arange(
            *self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(
            0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH, fW)
        ys = torch.linspace(
            0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH, fW)

        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, rots, trans, intrins, post_rots, post_trans):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """

        B, S, N, _ = trans.shape
        BS = B * S

        # flatten (batch & sequence)
        rots = rots.flatten(0, 1)
        trans = trans.flatten(0, 1)
        intrins = intrins.flatten(0, 1).float()
        # inverse can only work for float32
        post_rots = post_rots.flatten(0, 1).float()
        post_trans = post_trans.flatten(0, 1)

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(BS, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            BS, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)

        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(BS, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(BS, N, 1, 1, 1, 3)

        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        bx = self.bx.type_as(geom_feats)
        dx = self.dx.type_as(geom_feats)
        nx = self.nx.type_as(geom_feats).long()

        geom_feats = ((geom_feats - (bx - dx / 2.)) / dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        geom_feats = geom_feats.type_as(x).long()

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        if self.use_bev_pool:
            final = bev_pool(x, geom_feats, B,
                             self.nx[2], self.nx[0], self.nx[1])
            final = final.transpose(dim0=-2, dim1=-1)
        else:
            # get tensors from the same voxel next to each other
            ranks = geom_feats[:, 0] * (nx[1] * nx[2] * B) \
                + geom_feats[:, 1] * (nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
            sorts = ranks.argsort()
            x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

            # cumsum trick
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

            # griddify (B x C x Z x X x Y)
            final = torch.zeros((B, C, nx[2], nx[1], nx[0]), device=x.device)
            final[geom_feats[:, 3], :, geom_feats[:, 2],
                  geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, input, flip_x=False, flip_y=False):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, S, N, C, H, W = x.shape
        # flatten (batch, seq, num_cam)
        x = x.view(B * S * N, C, H, W)
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        # [B * S, N, D, H, W, 3]
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)
        cvt_feature_list = [x[:, self.D:(self.D + self.numC_Trans)]]
        volume_channel_index = [0, ]
        for feature in cvt_feature_list:
            volume_channel_index.append(
                feature.shape[1]+volume_channel_index[-1])

        cvt_feature = torch.cat(cvt_feature_list, dim=1)

        volume = depth.unsqueeze(1) * cvt_feature.unsqueeze(2)
        volume = volume.view(B * S, N, volume_channel_index[-1], self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        if flip_x:
            geom[..., 0] = -geom[..., 0]
        if flip_y:
            geom[..., 1] = -geom[..., 1]

        bev_feat = self.voxel_pooling(geom, volume)
        bev_feat = bev_feat.view(B, S, *bev_feat.shape[1:])

        return bev_feat


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]])

    return dx, bx, nx


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None
