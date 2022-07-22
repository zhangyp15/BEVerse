# Copyright (c) Junjie.huang. All rights reserved.

import torch
import torch.nn as nn
from mmcv.runner import BaseModule
from ..builder import NECKS
from .. import builder
from mmdet.models.backbones.resnet import Bottleneck, BasicBlock
from mmcv.cnn import build_norm_layer
from ..backbones.swin import SwinTransformer

import pdb


def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


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


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, norm_cfg=dict(type='BN')):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)
        assert norm_cfg['type'] in ['BN', 'SyncBN']
        if norm_cfg['type'] == 'BN':
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, out_channels, postfix=0)[1],
                nn.ReLU(inplace=True)
            )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class BevEncode(nn.Module):
    def __init__(self, numC_input, numC_output, num_layer=[2, 2, 2], num_channels=None,
                 backbone_output_ids=None,  norm_cfg=dict(type='BN'),
                 bev_encode_block='BottleNeck', multiview_learning=False, feature_fuse_type='SUM',
                 bev_encoder_fpn_type='lssfpn'):
        super(BevEncode, self).__init__()

        # build downsample modules for multiview learning
        self.multiview_learning = multiview_learning
        if self.multiview_learning:
            downsample_conv_list = []
            for i in range(len(num_layer)-1):
                downsample_conv_list.append(
                    nn.Sequential(
                        nn.Conv2d(numC_input, numC_input * 2**(i+1),
                                  kernel_size=3, stride=2**(i+1), padding=1, bias=False),
                        build_norm_layer(norm_cfg, numC_input *
                                         2**(i+1), postfix=0)[1],
                        nn.ReLU(inplace=True)))
            self.downsample_conv_list = nn.Sequential(*downsample_conv_list)
        self.feature_fuse_type = feature_fuse_type

        # build backbone
        assert len(num_layer) >= 3
        num_channels = [numC_input*2**(i+1) for i in range(
            len(num_layer))] if num_channels is None else num_channels
        self.backbone_output_ids = range(len(
            num_layer)-3, len(num_layer)) if backbone_output_ids is None else backbone_output_ids
        layers = []
        if bev_encode_block == 'BottleNeck':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [Bottleneck(curr_numC, num_channels[i]//4, stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([Bottleneck(curr_numC, curr_numC//4,
                                         norm_cfg=norm_cfg) for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        elif bev_encode_block == 'Basic':
            curr_numC = numC_input
            for i in range(len(num_layer)):
                layer = [BasicBlock(curr_numC, num_channels[i], stride=2,
                                    downsample=nn.Conv2d(
                                        curr_numC, num_channels[i], 3, 2, 1),
                                    norm_cfg=norm_cfg)]
                curr_numC = num_channels[i]
                layer.extend([BasicBlock(curr_numC, curr_numC, norm_cfg=norm_cfg)
                             for _ in range(num_layer[i]-1)])
                layers.append(nn.Sequential(*layer))
        else:
            assert False
        self.layers = nn.Sequential(*layers)

        # build neck
        self.bev_encoder_fpn_type = bev_encoder_fpn_type
        if self.bev_encoder_fpn_type == 'lssfpn':
            self.up1 = Up(num_channels[-1] + num_channels[-3],
                          numC_output*2, scale_factor=4, norm_cfg=norm_cfg)
        elif self.bev_encoder_fpn_type == 'fpnv1':
            img_neck_cfg = dict(
                type='FPNv1',
                in_channels=num_channels[-3:],
                out_channels=numC_output*2,
                num_outs=1,
                start_level=0,
                out_ids=[0])
            self.up1 = builder.build_neck(img_neck_cfg)
        else:
            assert False
        assert norm_cfg['type'] in ['BN', 'SyncBN']
        if norm_cfg['type'] == 'BN':
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(numC_output * 2, numC_output,
                          kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(numC_output),
                nn.ReLU(inplace=True),
                nn.Conv2d(numC_output, numC_output, kernel_size=1, padding=0),
            )
        else:
            self.up2 = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear',
                            align_corners=True),
                nn.Conv2d(numC_output * 2, numC_output,
                          kernel_size=3, padding=1, bias=False),
                build_norm_layer(norm_cfg, numC_output, postfix=0)[1],
                nn.ReLU(inplace=True),
                nn.Conv2d(numC_output, numC_output, kernel_size=1, padding=0),
            )

    def forward(self, bev_feat_list):
        feats = []
        x_tmp = bev_feat_list[0]
        for lid, layer in enumerate(self.layers):
            x_tmp = layer(x_tmp)
            # x_tmp = checkpoint.checkpoint(layer,x_tmp)
            if lid in self.backbone_output_ids:
                feats.append(x_tmp)
            if lid < (len(self.layers)-1) and self.multiview_learning:
                if self.feature_fuse_type == 'SUM':
                    bev_feat_from_img_view = bev_feat_list[lid + 1]
                    bev_feat_from_img_view = self.downsample_conv_list[lid](
                        bev_feat_from_img_view)
                    x_tmp = x_tmp + bev_feat_from_img_view
                else:
                    assert False
        if self.bev_encoder_fpn_type == 'lssfpn':
            res = self.up1(feats[-1], feats[-3])
        elif self.bev_encoder_fpn_type == 'fpnv1':
            res = self.up1(feats)
        else:
            assert False
        res = self.up2(res)
        return res


def check_point(p, w, h):
    if p[0] < 0 or p[0] > w-1 or p[1] < 0 or p[1] > h-1:
        return False
    return True


@NECKS.register_module()
class TransformerLiftSplatShootMultiview(BaseModule):
    def __init__(self, grid_conf=None, data_aug_conf=None, init_cfg=None,
                 numC_input=512, numC_Trans=512, numC_output=512, downsample=16,
                 bev_encode=False, bev_encode_block='BottleNeck', bev_encoder_type='resnet18',
                 bev_encode_depth=[2, 2, 2], num_channels=None, backbone_output_ids=None,
                 norm_cfg=dict(type='BN'), bev_encoder_fpn_type='lssfpn', **kwargs):
        super(TransformerLiftSplatShootMultiview, self).__init__(init_cfg)
        if grid_conf is None:
            grid_conf = {
                'xbound': [-51.2, 51.2, 0.8],
                'ybound': [-51.2, 51.2, 0.8],
                'zbound': [-10.0, 10.0, 20.0],
                'dbound': [4.0, 45.0, 1.0], }
        self.grid_conf = grid_conf
        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        if data_aug_conf is None:
            data_aug_conf = {
                'resize_lim': (0.193, 0.225),
                'final_dim': (128, 352),
                'rot_lim': (-5.4, 5.4),
                'H': 900, 'W': 1600,
                'rand_flip': True,
                'bot_pct_lim': (0.0, 0.22),
                'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                         'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                'Ncams': 5, }
        self.data_aug_conf = data_aug_conf
        self.downsample = downsample

        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.numC_input = numC_input
        self.numC_output = numC_output
        self.numC_Trans = numC_Trans
        self.depthnet = nn.Conv2d(
            self.numC_input, self.D + self.numC_Trans, kernel_size=1, padding=0)

        self.bev_encode = bev_encode
        if self.bev_encode:
            if bev_encoder_type == 'resnet18':
                self.bev_encode_module = BevEncode(numC_input=self.numC_Trans, numC_output=self.numC_output,
                                                   num_channels=num_channels,
                                                   backbone_output_ids=backbone_output_ids,
                                                   num_layer=bev_encode_depth, bev_encode_block=bev_encode_block,
                                                   norm_cfg=norm_cfg,
                                                   bev_encoder_fpn_type=bev_encoder_fpn_type)
            else:
                assert False, 'Unknown bev encoder type: %s' % bev_encoder_type

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']
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
        B, N, _ = trans.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)
        points = torch.inverse(post_rots).view(
            B, N, 1, 1, 1, 3, 3).matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)

        combine = rots.matmul(torch.inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W
        nx = self.nx.to(torch.long)
        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
            & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
            & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # [nx, ny, nz, n_batch]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
            + geom_feats[:, 1] * (self.nx[2] * B) \
            + geom_feats[:, 2] * B \
            + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)

        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, nx[2], nx[0], nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2],
              geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward_without_bevencoder(self, input):
        x, rots, trans, intrins, post_rots, post_trans = input
        B, N, C, H, W = x.shape
        x = x.view(B * N, C, H, W)
        x = self.depthnet(x)
        depth = self.get_depth_dist(x[:, :self.D])
        geom = self.get_geometry(rots, trans, intrins, post_rots, post_trans)

        cvt_feature_list = [x[:, self.D:(self.D + self.numC_Trans)]]

        volume_channel_index = [0, ]
        for feature in cvt_feature_list:
            volume_channel_index.append(
                feature.shape[1]+volume_channel_index[-1])

        cvt_feature = torch.cat(cvt_feature_list, dim=1)

        volume = depth.unsqueeze(1) * cvt_feature.unsqueeze(2)
        volume = volume.view(B, N, volume_channel_index[-1], self.D, H, W)
        volume = volume.permute(0, 1, 3, 4, 5, 2)

        bev_feat = self.voxel_pooling(geom, volume)
        bev_feat_list = [bev_feat[:, volume_channel_index[i]:volume_channel_index[i+1], ...]
                         for i in range(len(cvt_feature_list))]
        return bev_feat_list

    def forward(self, input):
        bev_feat_list = self.forward_without_bevencoder(input)
        if self.bev_encode:
            bev_feat = self.bev_encode_module(bev_feat_list)
        else:
            assert len(bev_feat_list) == 1
            bev_feat = bev_feat_list[0]
        return [bev_feat]
