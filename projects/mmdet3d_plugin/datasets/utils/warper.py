import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from pyquaternion import Quaternion
from .geometry import pose_vec2mat, mat2pose_vec, invert_pose_matrix

def gen_dx_bx(xbound, ybound, zbound):
    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor(
        [row[0] + row[2]/2.0 for row in [xbound, ybound, zbound]])
    nx = torch.Tensor([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]])

    return dx, bx, nx

class FeatureWarper(object):
    def __init__(self, grid_conf=None, input_dim=None):
        self.grid_conf = grid_conf

        # dx 间隔，bx 起始点, nx 尺寸
        dx, bx, nx = gen_dx_bx(
            grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound'])

        self.dx = dx.numpy()
        self.bx = bx.numpy()
        self.nx = nx.long().numpy()

        self.spatial_extent = (grid_conf['xbound'][1], grid_conf['ybound'][1])

        # generate bev meshgrid
        xs = torch.linspace(
            self.bx[0], self.spatial_extent[0], self.nx[0], dtype=torch.float).view(1, -1)
        ys = torch.linspace(
            self.bx[1], self.spatial_extent[1], self.nx[1], dtype=torch.float).view(-1, 1)

        self.bev_grid = torch.stack(
            (xs.expand(self.nx[1], self.nx[0]), ys.expand(self.nx[1], self.nx[0])), dim=2)
        self.bev_grid = nn.Parameter(self.bev_grid, requires_grad=False)

    def warp_features(self, x, flow, mode='nearest'):
        """ Applies a rotation and translation to feature map x.
        Args:
            x: (b, c, h, w) feature map
            flow: (b, 6) 6DoF vector (only uses the xy poriton)
            mode: use 'nearest' when dealing with categorical inputs
        Returns:
            in plane transformed feature map
        """

        # transform coordinates
        flow = flow.float().inverse()
        # [b, 2, 3]
        xy_flow = flow[..., :2, [0, 1, 3]]
        # [h, w, 3, 1]
        points = torch.cat(
            (self.bev_grid, torch.ones_like(self.bev_grid)[..., :1]), dim=-1).unsqueeze(-1)

        # [b, 1, 1, 2, 3] @ [1, h, w, 3, 1] ==> [b, h, w, 2, 1]
        trans_points = xy_flow.view(-1, 1, 1, 2,
                                    3) @ points.unsqueeze(0).type_as(xy_flow)
        trans_points = trans_points.squeeze(-1)

        # normalize points: [b, h, w, 2]
        trans_points = trans_points[..., :2]
        trans_points[..., 0] /= self.spatial_extent[0]
        trans_points[..., 1] /= self.spatial_extent[1]

        warped_x = F.grid_sample(x, trans_points.float(), mode=mode,
                                 padding_mode='zeros', align_corners=True)

        return warped_x

    def cumulative_warp_features(self, x, flow, mode='nearest', bev_transform=None):
        """ Warps a sequence of feature maps by accumulating incremental 2d flow.

        x[:, -1] remains unchanged
        x[:, -2] is warped using flow[:, -2]
        x[:, -3] is warped using flow[:, -3] @ flow[:, -2]
        ...
        x[:, 0] is warped using flow[:, 0] @ ... @ flow[:, -3] @ flow[:, -2]

        Args:
            x: (b, t, c, h, w) sequence of feature maps
            flow: (b, t, 6) sequence of 6 DoF pose
                from t to t+1 (only uses the xy poriton)
        """

        sequence_length = x.shape[1]
        if sequence_length == 1:
            return x

        flow = pose_vec2mat(flow)

        out = [x[:, -1]]
        cum_flow = flow[:, -2]
        for t in reversed(range(sequence_length - 1)):
            if bev_transform is not None:
                warp_flow = bev_transform @ cum_flow @ bev_transform.inverse()
            else:
                warp_flow = cum_flow.clone()

            out.append(self.warp_features(
                x[:, t], warp_flow, mode=mode))
            # @ is the equivalent of torch.bmm
            cum_flow = flow[:, t - 1] @ cum_flow

        return torch.stack(out[::-1], 1)

    def cumulative_warp_features_reverse(self, x, flow, mode='nearest', bev_transform=None):
        # flow vector (6 DoF) ==> flow matrix (4 x 4)
        flow = pose_vec2mat(flow)
        out = [x[:, 0]]

        for i in range(1, x.shape[1]):
            if i == 1:
                cum_flow = invert_pose_matrix(flow[:, 0])
            else:
                cum_flow = cum_flow @ invert_pose_matrix(flow[:, i-1])

            # cum_flow only represents the ego_motion, while bev_transform needs extra processing
            if bev_transform is not None:
                # points 先做 inverse_bev_transform，再做 motion 变换，再做 bev_transform
                # warp_flow = bev_transform @ cum_flow @ bev_transform.inverse()
                warp_flow = bev_transform @ cum_flow @ bev_transform.inverse()
            else:
                warp_flow = cum_flow.clone()

            out.append(self.warp_features(x[:, i], warp_flow, mode))

        return torch.stack(out, 1)
