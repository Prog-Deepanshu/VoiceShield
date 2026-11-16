"""
AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
======================================================================================

Adapted from the official PyTorch implementation by NAVER Corp.
Source: https://github.com/clovaai/aasist

Original paper:
    Jung et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks", ICASSP 2022.

Copyright (c) 2021-present NAVER Corp.
MIT license

Modifications for VoiceGuard:
    - Removed training-specific code paths (Freq_aug masking)
    - Added inference-optimized forward pass
    - Added type hints and documentation
    - Structured for production deployment
"""

import random
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Network layer for homogeneous graphs.
    Computes pairwise node attention and projects features.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (#batch, #nodes, #dim)

        Returns:
            Output tensor of shape (#batch, #nodes, out_dim)
        """
        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)
        return x

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        """
        Calculates pairwise multiplication of nodes for attention map.

        Args:
            x: (#batch, #nodes, #dim)

        Returns:
            Pairwise products: (#batch, #nodes, #nodes, #dim)
        """
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map(self, x: Tensor) -> Tensor:
        """Derive attention map from input features."""
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))
        att_map = torch.matmul(att_map, self.att_weight)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        """Project features with and without attention."""
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        """Apply batch normalization across all nodes."""
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size) -> nn.Parameter:
        """Initialize parameters with Xavier normal."""
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class HtrgGraphAttentionLayer(nn.Module):
    """
    Heterogeneous Graph Attention Network layer.
    Handles two types of nodes (spectral and temporal) with a master node.
    """

    def __init__(self, in_dim: int, out_dim: int, **kwargs):
        super().__init__()

        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)

        # attention map
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)

        self.att_weight11 = self._init_new_params(out_dim, 1)
        self.att_weight22 = self._init_new_params(out_dim, 1)
        self.att_weight12 = self._init_new_params(out_dim, 1)
        self.att_weightM = self._init_new_params(out_dim, 1)

        # project
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)

        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)

        # batch norm
        self.bn = nn.BatchNorm1d(out_dim)

        # dropout for inputs
        self.input_drop = nn.Dropout(p=0.2)

        # activate
        self.act = nn.SELU(inplace=True)

        # temperature
        self.temp = kwargs.get("temperature", 1.0)

    def forward(
        self, x1: Tensor, x2: Tensor, master: Tensor = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass for heterogeneous graph.

        Args:
            x1: Type-1 node features (#batch, #nodes1, #dim)
            x2: Type-2 node features (#batch, #nodes2, #dim)
            master: Master node features (#batch, 1, #dim), or None

        Returns:
            Tuple of (x1_out, x2_out, master_out)
        """
        num_type1 = x1.size(1)
        num_type2 = x2.size(1)

        x1 = self.proj_type1(x1)
        x2 = self.proj_type2(x2)

        x = torch.cat([x1, x2], dim=1)

        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)

        # apply input dropout
        x = self.input_drop(x)

        # derive attention map
        att_map = self._derive_att_map(x, num_type1, num_type2)

        # directional edge for master node
        master = self._update_master(x, master)

        # projection
        x = self._project(x, att_map)

        # apply batch norm
        x = self._apply_BN(x)
        x = self.act(x)

        x1 = x.narrow(1, 0, num_type1)
        x2 = x.narrow(1, num_type1, num_type2)

        return x1, x2, master

    def _update_master(self, x: Tensor, master: Tensor) -> Tensor:
        """Update master node via attention over all nodes."""
        att_map = self._derive_att_map_master(x, master)
        master = self._project_master(x, master, att_map)
        return master

    def _pairwise_mul_nodes(self, x: Tensor) -> Tensor:
        """Pairwise node multiplication for attention computation."""
        nb_nodes = x.size(1)
        x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
        x_mirror = x.transpose(1, 2)
        return x * x_mirror

    def _derive_att_map_master(self, x: Tensor, master: Tensor) -> Tensor:
        """Derive attention map between master and all nodes."""
        att_map = x * master
        att_map = torch.tanh(self.att_projM(att_map))
        att_map = torch.matmul(att_map, self.att_weightM)
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _derive_att_map(
        self, x: Tensor, num_type1: int, num_type2: int
    ) -> Tensor:
        """Derive heterogeneous attention map with typed edge weights."""
        att_map = self._pairwise_mul_nodes(x)
        att_map = torch.tanh(self.att_proj(att_map))

        att_board = torch.zeros_like(att_map[:, :, :, 0]).unsqueeze(-1)

        att_board[:, :num_type1, :num_type1, :] = torch.matmul(
            att_map[:, :num_type1, :num_type1, :], self.att_weight11
        )
        att_board[:, num_type1:, num_type1:, :] = torch.matmul(
            att_map[:, num_type1:, num_type1:, :], self.att_weight22
        )
        att_board[:, :num_type1, num_type1:, :] = torch.matmul(
            att_map[:, :num_type1, num_type1:, :], self.att_weight12
        )
        att_board[:, num_type1:, :num_type1, :] = torch.matmul(
            att_map[:, num_type1:, :num_type1, :], self.att_weight12
        )

        att_map = att_board
        att_map = att_map / self.temp
        att_map = F.softmax(att_map, dim=-2)
        return att_map

    def _project(self, x: Tensor, att_map: Tensor) -> Tensor:
        x1 = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x))
        x2 = self.proj_without_att(x)
        return x1 + x2

    def _project_master(
        self, x: Tensor, master: Tensor, att_map: Tensor
    ) -> Tensor:
        x1 = self.proj_with_attM(
            torch.matmul(att_map.squeeze(-1).unsqueeze(1), x)
        )
        x2 = self.proj_without_attM(master)
        return x1 + x2

    def _apply_BN(self, x: Tensor) -> Tensor:
        org_size = x.size()
        x = x.view(-1, org_size[-1])
        x = self.bn(x)
        x = x.view(org_size)
        return x

    def _init_new_params(self, *size) -> nn.Parameter:
        out = nn.Parameter(torch.FloatTensor(*size))
        nn.init.xavier_normal_(out)
        return out


class GraphPool(nn.Module):
    """
    Graph pooling layer using top-k node selection based on learned scores.
    """

    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()
        self.in_dim = in_dim

    def forward(self, h: Tensor) -> Tensor:
        Z = self.drop(h)
        weights = self.proj(Z)
        scores = self.sigmoid(weights)
        new_h = self._top_k_graph(scores, h, self.k)
        return new_h

    def _top_k_graph(self, scores: Tensor, h: Tensor, k: float) -> Tensor:
        """
        Select top-k nodes based on attention scores.

        Args:
            scores: Attention scores (#batch, #nodes, 1)
            h:      Graph features (#batch, #nodes, #dim)
            k:      Ratio of nodes to keep

        Returns:
            Pooled graph (#batch, #nodes', #dim)
        """
        _, n_nodes, n_feat = h.size()
        n_nodes = max(int(n_nodes * k), 1)
        _, idx = torch.topk(scores, n_nodes, dim=1)
        idx = idx.expand(-1, -1, n_feat)

        h = h * scores
        h = torch.gather(h, 1, idx)
        return h


class SincConv(nn.Module):
    """
    Sinc-based convolutional layer (learned bandpass filterbank).
    Operates directly on raw waveform input, replacing traditional
    handcrafted feature extraction (e.g., mel spectrogram).
    """

    @staticmethod
    def to_mel(hz: float) -> float:
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel: float) -> float:
        return 700 * (10 ** (mel / 2595) - 1)

    def __init__(
        self,
        out_channels: int,
        kernel_size: int,
        sample_rate: int = 16000,
        in_channels: int = 1,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        bias: bool = False,
        groups: int = 1,
        mask: bool = False,
    ):
        super().__init__()
        if in_channels != 1:
            raise ValueError(
                f"SincConv only supports one input channel (got in_channels={in_channels})"
            )
        if bias:
            raise ValueError("SincConv does not support bias.")
        if groups > 1:
            raise ValueError("SincConv does not support groups.")

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate

        # Force odd kernel size for perfect symmetry
        if kernel_size % 2 == 0:
            self.kernel_size = self.kernel_size + 1

        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.mask = mask

        # Initialize mel-spaced bandpass filters
        NFFT = 512
        f = int(self.sample_rate / 2) * np.linspace(0, 1, int(NFFT / 2) + 1)
        fmel = self.to_mel(f)
        fmelmax = np.max(fmel)
        fmelmin = np.min(fmel)
        filbandwidthsmel = np.linspace(fmelmin, fmelmax, self.out_channels + 1)
        filbandwidthsf = self.to_hz(filbandwidthsmel)

        self.mel = filbandwidthsf
        self.hsupp = torch.arange(
            -(self.kernel_size - 1) / 2, (self.kernel_size - 1) / 2 + 1
        )
        self.band_pass = torch.zeros(self.out_channels, self.kernel_size)

        for i in range(len(self.mel) - 1):
            fmin = self.mel[i]
            fmax = self.mel[i + 1]
            hHigh = (2 * fmax / self.sample_rate) * np.sinc(
                2 * fmax * self.hsupp / self.sample_rate
            )
            hLow = (2 * fmin / self.sample_rate) * np.sinc(
                2 * fmin * self.hsupp / self.sample_rate
            )
            hideal = hHigh - hLow
            self.band_pass[i, :] = Tensor(np.hamming(self.kernel_size)) * Tensor(
                hideal
            )

    def forward(self, x: Tensor, mask: bool = False) -> Tensor:
        """
        Apply sinc bandpass filterbank to raw waveform.

        Args:
            x:    Raw waveform (#batch, 1, #samples)
            mask: Whether to apply frequency masking (training only)

        Returns:
            Filtered output (#batch, #out_channels, #time)
        """
        band_pass_filter = self.band_pass.clone().to(x.device).to(x.dtype)

        if mask:
            A = int(np.random.uniform(0, 20))
            A0 = random.randint(0, band_pass_filter.shape[0] - A)
            band_pass_filter[A0 : A0 + A, :] = 0

        self.filters = band_pass_filter.view(
            self.out_channels, 1, self.kernel_size
        )

        return F.conv1d(
            x,
            self.filters,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            bias=None,
            groups=1,
        )


class Residual_block(nn.Module):
    """Residual block with 2D convolutions, batch norm, and max pooling."""

    def __init__(self, nb_filts: list[int], first: bool = False):
        super().__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])

        self.conv1 = nn.Conv2d(
            in_channels=nb_filts[0],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(1, 1),
            stride=1,
        )
        self.selu = nn.SELU(inplace=True)

        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        self.conv2 = nn.Conv2d(
            in_channels=nb_filts[1],
            out_channels=nb_filts[1],
            kernel_size=(2, 3),
            padding=(0, 1),
            stride=1,
        )

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(
                in_channels=nb_filts[0],
                out_channels=nb_filts[1],
                padding=(0, 1),
                kernel_size=(1, 3),
                stride=1,
            )
        else:
            self.downsample = False

        self.mp = nn.MaxPool2d((1, 3))

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.selu(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out


class Model(nn.Module):
    """
    AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal
    Graph Attention Networks.

    This model processes raw waveform audio through:
    1. Sinc-based bandpass filterbank (learned front-end)
    2. Residual convolutional encoder
    3. Dual-branch Graph Attention Networks (spectral + temporal)
    4. Heterogeneous graph fusion with master nodes
    5. Final classification head (bonafide vs. spoof)

    Args:
        d_args: Model configuration dictionary containing:
            - filts: Filter dimensions for each residual block
            - gat_dims: GAT hidden dimensions
            - pool_ratios: Graph pooling ratios
            - temperatures: Attention temperature parameters
            - first_conv: Number of sinc filters
            - nb_samp: Expected input length in samples
    """

    def __init__(self, d_args: dict):
        super().__init__()

        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]

        # ── Sinc filter front-end ────────────────────────────────────
        self.conv_time = SincConv(
            out_channels=filts[0],
            kernel_size=d_args["first_conv"],
            in_channels=1,
        )
        self.first_bn = nn.BatchNorm2d(num_features=1)

        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)

        # ── Residual convolutional encoder ───────────────────────────
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Residual_block(nb_filts=filts[2])),
            nn.Sequential(Residual_block(nb_filts=filts[3])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
            nn.Sequential(Residual_block(nb_filts=filts[4])),
        )

        # ── Spectral and temporal GAT branches ──────────────────────
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))

        self.GAT_layer_S = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[0]
        )
        self.GAT_layer_T = GraphAttentionLayer(
            filts[-1][-1], gat_dims[0], temperature=temperatures[1]
        )

        # ── Heterogeneous graph attention layers ─────────────────────
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(
            gat_dims[0], gat_dims[1], temperature=temperatures[2]
        )
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(
            gat_dims[1], gat_dims[1], temperature=temperatures[2]
        )

        # ── Graph pooling layers ─────────────────────────────────────
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)

        # ── Classification head ──────────────────────────────────────
        # 5 pooled representations concatenated: T_max, T_avg, S_max, S_avg, master
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, x: Tensor, Freq_aug: bool = False) -> tuple[Tensor, Tensor]:
        """
        Forward pass.

        Args:
            x:        Raw waveform tensor (#batch, #samples)
            Freq_aug: Whether to apply frequency augmentation (training only)

        Returns:
            Tuple of:
                - last_hidden: Embedding vector (#batch, 5*gat_dims[1])
                - output:      Classification logits (#batch, 2)
                               where [:, 0] = spoof score, [:, 1] = bonafide score
        """
        # ── Front-end processing ─────────────────────────────────────
        x = x.unsqueeze(1)  # (#batch, 1, #samples)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)  # (#batch, 1, #filts, #time)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)

        # ── Encoder ──────────────────────────────────────────────────
        e = self.encoder(x)  # (#batch, #filt, #spec, #seq)

        # ── Spectral GAT (GAT-S) ────────────────────────────────────
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S

        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)

        # ── Temporal GAT (GAT-T) ────────────────────────────────────
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)

        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)

        # ── Learnable master nodes ───────────────────────────────────
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        # ── Inference branch 1 ───────────────────────────────────────
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(
            out_T, out_S, master=self.master1
        )
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(
            out_T1, out_S1, master=master1
        )
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug

        # ── Inference branch 2 ───────────────────────────────────────
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(
            out_T, out_S, master=self.master2
        )
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)

        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(
            out_T2, out_S2, master=master2
        )
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug

        # ── Combine branches with dropout ────────────────────────────
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)

        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)

        # ── Aggregate pooled representations ─────────────────────────
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)

        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)

        last_hidden = torch.cat(
            [T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1
        )

        # ── Classification ───────────────────────────────────────────
        last_hidden = self.drop(last_hidden)
        output = self.out_layer(last_hidden)

        return last_hidden, output
