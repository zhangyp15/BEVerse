import torch
import torch.nn as nn
import torch.nn.functional as F
from .basic_modules import Bottleneck, SpatialGRU, ConvBlock, GRUCell

import pdb
import copy


class DistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
            self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.encoder = DistributionEncoder(
            in_channels,
            self.compress_dim,
        )
        self.last_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(
                self.compress_dim, out_channels=2 * self.latent_dim, kernel_size=1)
        )

        self.fp16_enabled = False

    def forward(self, s_t):
        b, s = s_t.shape[:2]
        assert s == 1
        encoding = self.encoder(s_t[:, 0])

        mu_log_sigma = self.last_conv(encoding).view(b, 1, 2 * self.latent_dim)
        mu = mu_log_sigma[:, :, :self.latent_dim]
        log_sigma = mu_log_sigma[:, :, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(
            log_sigma, self.min_log_sigma, self.max_log_sigma)
        return mu, log_sigma


class SpatialDistributionModule(nn.Module):
    """
    A convolutional net that parametrises a diagonal Gaussian distribution.
    """

    def __init__(
            self, in_channels, latent_dim, min_log_sigma, max_log_sigma):
        super().__init__()
        self.compress_dim = in_channels // 2
        self.latent_dim = latent_dim
        self.min_log_sigma = min_log_sigma
        self.max_log_sigma = max_log_sigma

        self.encoder = DistributionEncoder(
            in_channels,
            self.compress_dim,
        )

        # difference: remove global average pooling
        self.last_conv = nn.Sequential(
            nn.Conv2d(self.compress_dim, out_channels=2 *
                      self.latent_dim, kernel_size=1)
        )
        self.fp16_enabled = False

    def forward(self, s_t):
        b, s = s_t.shape[:2]
        assert s == 1
        encoding = self.encoder(s_t[:, 0])

        # [batch, latent_dim, h, w]
        mu_log_sigma = self.last_conv(encoding)
        mu = mu_log_sigma[:, :self.latent_dim]
        log_sigma = mu_log_sigma[:, self.latent_dim:]

        # clip the log_sigma value for numerical stability
        log_sigma = torch.clamp(
            log_sigma, self.min_log_sigma, self.max_log_sigma)

        return mu, log_sigma


class DistributionEncoder(nn.Module):
    """Encodes s_t or (s_t, y_{t+1}, ..., y_{t+H}).
    """

    def __init__(self, in_channels, out_channels, num_layer=2):
        super().__init__()

        layers = []
        for _ in range(num_layer):
            layers.append(Bottleneck(in_channels=in_channels,
                          out_channels=out_channels, downsample=True))
            in_channels = out_channels

        self.model = nn.Sequential(*layers)
        self.fp16_enabled = False

    def forward(self, s_t):
        return self.model(s_t)


class FuturePrediction(torch.nn.Module):
    def __init__(self, in_channels, latent_dim, n_gru_blocks=3, n_res_layers=3):
        super().__init__()
        self.n_gru_blocks = n_gru_blocks

        # Convolutional recurrent model with z_t as an initial hidden state and inputs the sample
        # from the probabilistic model. The architecture of the model is:
        # [Spatial GRU - [Bottleneck] x n_res_layers] x n_gru_blocks
        self.spatial_grus = []
        self.res_blocks = []

        # 第一个 gru_block 将低维的 random_variable 转为高维的 特征
        for i in range(self.n_gru_blocks):
            gru_in_channels = latent_dim if i == 0 else in_channels
            self.spatial_grus.append(SpatialGRU(gru_in_channels, in_channels))
            self.res_blocks.append(torch.nn.Sequential(*[Bottleneck(in_channels)
                                                         for _ in range(n_res_layers)]))

        self.spatial_grus = torch.nn.ModuleList(self.spatial_grus)
        self.res_blocks = torch.nn.ModuleList(self.res_blocks)

    def forward(self, x, hidden_state):
        # x 为隐分布, hidden_state 为当前时刻的 bev_features
        # 在每个 gru_block 中，x 为 future_states, hidden_state 为当前的特征
        # grucell 实现了 future_states 沿着时序方向的 updates
        # res_blocks 则是在空域上对特征进行进一步的提取

        # x has shape (b, n_future, c_latent_dim, h, w), hidden_state (b, c, h, w)
        for i in range(self.n_gru_blocks):
            x = self.spatial_grus[i](x, hidden_state, flow=None)
            b, n_future, c, h, w = x.shape

            x = self.res_blocks[i](x.view(b * n_future, c, h, w))
            x = x.view(b, n_future, c, h, w)

        return x


class ResFuturePrediction(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 n_future,
                 detach_state=True,
                 n_gru_blocks=3,
                 flow_warp=True,
                 prob_each_future=False,
                 ):
        super().__init__()
        self.n_future = n_future
        self.detach_state = detach_state
        self.flow_warp = flow_warp
        self.prob_each_future = prob_each_future

        # 每个时刻，都以 sample_distribution 和 当前帧的 bev features 作为输入
        # 1. offset prediction: 预测 feature flow ==> warp features
        # 2. gru_cell: reset & update
        # 3. spatial convolution: 输出当前帧特征

        # offset prediction
        if self.flow_warp:
            self.offset_conv = ConvBlock(in_channels=latent_dim + in_channels)
            self.offset_pred = nn.Conv2d(
                latent_dim + in_channels, 2, kernel_size=1, padding=0)

        # gru_cell
        self.gru_cells = nn.ModuleList()
        gru_in_channels = in_channels + latent_dim
        for _ in range(n_gru_blocks):
            self.gru_cells.append(
                GRUCell(input_size=gru_in_channels, hidden_size=in_channels))
            gru_in_channels = in_channels

        # spatial conv
        self.spatial_conv = ConvBlock(in_channels=in_channels)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # init weights
        self.init_weights()

    def init_weights(self):
        if self.flow_warp:
            self.offset_pred.weight.data.normal_(0.0, 0.02)
            self.offset_pred.bias.data.fill_(0)

    def forward(self, sample_distribution, hidden_state):
        # x has shape (b, c_latent_dim, h, w), hidden_state (b, c, h, w)
        res = []
        current_state = hidden_state
        for i in range(self.n_future):

            if self.flow_warp:
                combine = torch.cat(
                    (sample_distribution, current_state), dim=1)
                flow = self.offset_pred(self.offset_conv(combine))
                warp_state = warp_with_flow(current_state, flow=flow)
                warp_state = torch.cat(
                    (warp_state, sample_distribution), dim=1)
            else:
                warp_state = torch.cat(
                    (sample_distribution, current_state), dim=1)

            for gru_cell in self.gru_cells:
                warp_state = gru_cell(warp_state, state=current_state)

            warp_state = self.spatial_conv(warp_state)
            res.append(warp_state)

            # updating current states
            if self.detach_state:
                current_state = warp_state.detach()
            else:
                current_state = warp_state.clone()

        return torch.stack(res, dim=1)


class ResFuturePredictionV2(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 latent_dim,
                 n_future,
                 detach_state=True,
                 n_gru_blocks=3,
                 flow_warp=True,
                 prob_each_future=False,
                 with_state_refine=True,
                 ):
        super().__init__()
        self.n_future = n_future
        self.detach_state = detach_state
        self.flow_warp = flow_warp
        self.prob_each_future = prob_each_future
        self.with_state_refine = with_state_refine

        # 每个时刻，都以 sample_distribution 和 当前帧的 bev features 作为输入
        # 1. offset prediction: 预测 feature flow ==> warp features
        # 2. gru_cell: reset & update
        # 3. spatial convolution: 输出当前帧特征

        # offset prediction
        if self.flow_warp:
            flow_pred = nn.Sequential(
                ConvBlock(in_channels=latent_dim + in_channels),
                nn.Conv2d(latent_dim + in_channels, 2,
                          kernel_size=1, padding=0),
            )

        # gru_cell
        self.gru_cells = nn.ModuleList()
        gru_in_channels = in_channels + latent_dim
        for _ in range(n_gru_blocks):
            self.gru_cells.append(
                GRUCell(input_size=gru_in_channels, hidden_size=in_channels))
            gru_in_channels = in_channels

        # spatial conv
        spatial_conv = ConvBlock(in_channels=in_channels)

        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        if self.with_state_refine:
            if self.flow_warp:
                self.flow_preds = _get_clones(flow_pred, self.n_future)

            self.spatial_convs = _get_clones(spatial_conv, self.n_future)
        else:
            if self.flow_warp:
                self.flow_preds = nn.ModuleList(
                    [flow_pred for _ in range(self.n_future)])

            self.spatial_convs = nn.ModuleList(
                [spatial_conv for _ in range(self.n_future)])

        self.init_weights()

    def init_weights(self):
        if self.flow_warp:
            for flow_pred in self.flow_preds:
                flow_pred[1].weight.data.normal_(0.0, 0.02)
                flow_pred[1].bias.data.fill_(0)

    def forward(self, sample_distribution, hidden_state):
        # x has shape (b, c_latent_dim, h, w), hidden_state (b, c, h, w)
        res = []
        current_state = hidden_state
        for i in range(self.n_future):
            if self.flow_warp:
                combine = torch.cat(
                    (sample_distribution, current_state), dim=1)
                flow = self.flow_preds[i](combine)

                warp_state = warp_with_flow(current_state, flow=flow)
                warp_state = torch.cat(
                    (warp_state, sample_distribution), dim=1)
            else:
                warp_state = torch.cat(
                    (current_state, sample_distribution), dim=1)

            for gru_cell in self.gru_cells:
                warp_state = gru_cell(warp_state, state=current_state)

            warp_state = self.spatial_convs[i](warp_state)
            res.append(warp_state)

            # updating current states
            if self.detach_state:
                current_state = warp_state.detach()
            else:
                current_state = warp_state.clone()

        return torch.stack(res, dim=1)


class ResFuturePredictionV1(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        latent_dim,
        n_future,
        n_gru_blocks=3,
        flow_warp=True,
        detach_state=False,
        prob_each_future=False,
    ):
        super().__init__()
        self.n_future = n_future
        self.detach_state = detach_state
        self.flow_warp = flow_warp
        self.latent_dim = latent_dim
        self.hidden_size = in_channels
        self.n_gru_blocks = n_gru_blocks
        self.prob_each_future = prob_each_future

        # offset prediction
        if self.flow_warp:
            # 输入为上一个时刻的特征 + 采样的分布
            self.flow_pred = nn.Sequential(
                ConvBlock(in_channels=latent_dim + in_channels),
                nn.Conv2d(latent_dim + in_channels, 2,
                          kernel_size=1, padding=0),
            )

        # 为每个时刻都生成一个分布 ?
        self.spatial_grus = nn.ModuleList()
        self.spatial_convs = nn.ModuleList()

        gru_in_channels = in_channels + latent_dim
        for i in range(n_gru_blocks):
            if i == 0:
                self.spatial_grus.append(
                    GRUCell(input_size=gru_in_channels, hidden_size=in_channels))
            else:
                self.spatial_grus.append(
                    SpatialGRU(input_size=gru_in_channels, hidden_size=in_channels))

            gru_in_channels = in_channels
            # spatial conv
            self.spatial_convs.append(
                nn.Sequential(
                    ConvBlock(in_channels=in_channels),
                    nn.Conv2d(in_channels, in_channels,
                              kernel_size=1, padding=0),
                ),
            )

        # init weights
        self.init_weights()

    def init_weights(self):
        self.flow_pred[1].weight.data.normal_(0.0, 0.02)
        self.flow_pred[1].bias.data.fill_(0)

    def forward(self, sample_distribution, hidden_state):
        current_state = hidden_state
        future_states = [current_state]

        if self.prob_each_future:
            future_distributions = torch.split(
                sample_distribution, self.latent_dim, dim=1)
        else:
            future_distributions = [sample_distribution] * self.n_future

        # initialize hidden state
        b, _, h, w = hidden_state.shape
        rnn_state = torch.zeros(b, self.hidden_size, h,
                                w).type_as(hidden_state)
        for i in range(self.n_future):
            if self.flow_warp:
                combine = torch.cat(
                    (future_distributions[i], current_state), dim=1)
                flow = self.flow_pred(combine)
                current_state = warp_with_flow(current_state, flow=flow)

            combine = torch.cat(
                (future_distributions[i], current_state), dim=1)
            rnn_state = self.spatial_grus[0](combine, rnn_state)

            # update
            if self.detach_state:
                current_state = rnn_state.detach()
            else:
                current_state = rnn_state
            future_states.append(rnn_state)

        # [b, t, c, h, w]
        future_states = torch.stack(future_states, dim=1)
        b, t, c, h, w = future_states.shape
        future_states = self.spatial_convs[0](future_states.flatten(0, 1))
        future_states = future_states.view(b, t, c, h, w)

        # further updating
        for k in range(1, self.n_gru_blocks):
            future_states = self.spatial_grus[k](future_states)
            future_states = self.spatial_convs[k](
                future_states.flatten(0, 1)).view(b, t, c, h, w)

        return future_states


def warp_with_flow(x, flow):
    B, C, H, W = x.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), dim=1).float()
    flow += grid.type_as(flow)
    flow = flow.permute(0, 2, 3, 1)

    flow[..., 0] = flow[..., 0] / (W - 1) * 2 - 1.0
    flow[..., 1] = flow[..., 1] / (H - 1) * 2 - 1.0
    x = F.grid_sample(x, flow, mode='bilinear', align_corners=True)

    return x
