"""Basic blocks of DINet_HD.

Structure and forward order are ported from the official open-source DINet
(https://github.com/MRzzm/DINet) and extended to the 512-channel "HD"
configuration used by HeyGem's ``dinet_v1_20240131.pth``.

Key naming convention (must match the checkpoint exactly):
  - SameBlock2d / SameBlock1d: ``{i}.conv.*`` + ``{i}.norm.*``
  - ResBlock2d / ResBlock1d:    ``{i}.conv1.*`` ``{i}.conv2.*``
                                ``{i}.norm1.*`` ``{i}.norm2.*``
  - DownBlock2d / DownBlock1d:  ``{i}.conv.*`` + ``{i}.norm.*``
"""

import torch
from torch import nn
from torch.nn import functional as F


def make_coordinate_grid_3d(spatial_size, ref: torch.Tensor):
    """Generate a 3D coordinate grid (used by AdaAT)."""
    d, h, w = spatial_size
    x = torch.arange(w, dtype=ref.dtype, device=ref.device)
    y = torch.arange(h, dtype=ref.dtype, device=ref.device)
    z = torch.arange(d, dtype=ref.dtype, device=ref.device)
    x = 2 * (x / (w - 1)) - 1
    y = 2 * (y / (h - 1)) - 1
    z = 2 * (z / (d - 1)) - 1
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)
    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3)], 3)
    return meshed, zz


class ResBlock1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv1d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(in_features, out_features, kernel_size, padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv1d(in_features, out_features, 1)
        self.norm1 = nn.BatchNorm1d(in_features)
        self.norm2 = nn.BatchNorm1d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + (self.channel_conv(x) if self.in_features != self.out_features else x)
        return out


class ResBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)
        if out_features != in_features:
            self.channel_conv = nn.Conv2d(in_features, out_features, 1)
        self.norm1 = nn.BatchNorm2d(in_features)
        self.norm2 = nn.BatchNorm2d(in_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = out + (self.channel_conv(x) if self.in_features != self.out_features else x)
        return out


class DownBlock1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, padding=padding, stride=2)
        self.norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1, stride=2):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding, stride=stride)
        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class UpBlock2d(nn.Module):
    """Upsample x2 then conv/norm/relu.

    Verified against the compiled original: feeding a (1,8,4,4) tensor through
    ``UpBlock2d(8,4)`` gives (1,4,8,8) and matches ``relu(norm(conv(interpolate(x,2))))``
    exactly, while ``interpolate(relu(norm(conv(x))),2)`` does NOT. Order matters:
    the convolution runs at the higher resolution.
    """

    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = F.interpolate(x, scale_factor=2)
        out = self.conv(out)
        out = self.norm(out)
        return self.relu(out)


class SameBlock1d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, padding):
        super().__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, padding=padding)
        self.norm = nn.BatchNorm1d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class SameBlock2d(nn.Module):
    def __init__(self, in_features, out_features, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))


class AdaAT(nn.Module):
    """Adaptive affine transformation operator (from open-source DINet)."""

    def __init__(self, para_ch, feature_ch):
        super().__init__()
        self.para_ch = para_ch
        self.feature_ch = feature_ch
        self.commn_linear = nn.Sequential(nn.Linear(para_ch, para_ch), nn.ReLU())
        self.scale = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Sigmoid())
        self.rotation = nn.Sequential(nn.Linear(para_ch, feature_ch), nn.Tanh())
        self.translation = nn.Sequential(nn.Linear(para_ch, 2 * feature_ch), nn.Tanh())

    def forward(self, feature_map, para_code):
        batch, d, h, w = feature_map.size(0), feature_map.size(1), feature_map.size(2), feature_map.size(3)
        para_code = self.commn_linear(para_code)
        scale = self.scale(para_code).unsqueeze(-1) * 2
        angle = self.rotation(para_code).unsqueeze(-1) * 3.14159
        rotation_matrix = torch.cat([torch.cos(angle), -torch.sin(angle),
                                     torch.sin(angle), torch.cos(angle)], -1)
        rotation_matrix = rotation_matrix.view(batch, self.feature_ch, 2, 2)
        translation = self.translation(para_code).view(batch, self.feature_ch, 2)
        grid_xy, grid_z = make_coordinate_grid_3d((d, h, w), feature_map)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch, 1, 1, 1, 1)
        grid_z = grid_z.unsqueeze(0).repeat(batch, 1, 1, 1)
        scale = scale.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        rotation_matrix = rotation_matrix.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1, 1)
        translation = translation.unsqueeze(2).unsqueeze(3).repeat(1, 1, h, w, 1)
        trans_grid = torch.matmul(rotation_matrix, grid_xy.unsqueeze(-1)).squeeze(-1) * scale + translation
        full_grid = torch.cat([trans_grid, grid_z.unsqueeze(-1)], -1)
        return F.grid_sample(feature_map.unsqueeze(1), full_grid, align_corners=False).squeeze(1)
