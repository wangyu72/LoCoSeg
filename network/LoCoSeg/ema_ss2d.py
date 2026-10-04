

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    MAMBA_AVAILABLE = True
except ImportError:
    MAMBA_AVAILABLE = False
    selective_scan_fn = None


def cross_scan(x: torch.Tensor):

    B, C, H, W = x.shape
    L = H * W
    x_hw = x.view(B, C, L)
    x_wh = x.transpose(2, 3).contiguous().view(B, C, L)
    x_rev_hw = x_hw.flip(dims=[-1])
    x_rev_wh = x_wh.flip(dims=[-1])
    y = torch.stack([x_hw, x_wh, x_rev_hw, x_rev_wh], dim=1)
    return y


def cross_merge(y: torch.Tensor, H, W):

    B, K, C, L = y.shape
    y_hw, y_wh, y_rev_hw, y_rev_wh = torch.unbind(y, dim=1)
    y_rev_hw = y_rev_hw.flip(dims=[-1])
    y_rev_wh = y_rev_wh.flip(dims=[-1])
    y_wh = y_wh.view(B, C, W, H).transpose(2, 3).contiguous().view(B, C, L)
    y_rev_wh = y_rev_wh.view(B, C, W, H).transpose(2, 3).contiguous().view(B, C, L)
    x = y_hw + y_wh + y_rev_hw + y_rev_wh
    return x.view(B, C, H, W)


class SS2DCore(nn.Module):

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=3,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dropout=0.,
        bias=False,
        **kwargs
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            self.d_inner, self.d_inner, groups=self.d_inner,
            bias=True, kernel_size=d_conv, padding=(d_conv - 1) // 2
        )
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    def forward_core(self, x: torch.Tensor):

        B, C, H, W = x.shape
        L, K = H * W, 4
        xs = cross_scan(x)

        x_dbl = self.x_proj(xs.permute(0, 1, 3, 2).contiguous().view(-1, C))
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)

        dts = self.dt_proj(dts).view(B, K, L, -1).permute(0, 1, 3, 2).contiguous().view(B * K, -1, L)
        Bs = Bs.view(B, K, L, -1).permute(0, 1, 3, 2).contiguous().view(B * K, -1, L)
        Cs = Cs.view(B, K, L, -1).permute(0, 1, 3, 2).contiguous().view(B * K, -1, L)
        xs = xs.view(B * K, -1, L)

        As = -torch.exp(self.A_log.float())
        Ds = self.D.float()

        if MAMBA_AVAILABLE:
            ys = selective_scan_fn(xs.float(), dts.float(), As, Bs.float(), Cs.float(), Ds, delta_softplus=True)
        else:
            ys = self._fallback_scan(xs, dts, As, Bs, Cs, Ds)

        y = ys.view(B, K, -1, L)
        return cross_merge(y, H, W)

    def _fallback_scan(self, xs, dts, As, Bs, Cs, Ds):

        return xs

    def forward(self, x: torch.Tensor):

        B, C, H, W = x.shape
        x_permuted = x.permute(0, 2, 3, 1)
        x_proj = self.in_proj(x_permuted)
        x_feat, z = x_proj.chunk(2, dim=-1)

        x_feat = x_feat.permute(0, 3, 1, 2).contiguous()
        x_feat = self.act(self.conv2d(x_feat))

        y = self.forward_core(x_feat)

        y = y.permute(0, 2, 3, 1)
        y = y * F.silu(z)
        out = self.out_proj(y)
        out = self.dropout(out)

        return out.permute(0, 3, 1, 2)


class EMA_SS2D(nn.Module):

    def __init__(
        self,
        channels: int,
        factor: int = 8,
        groups: int = 4,
        unshuffle_factor: int = 2,
    ):
        super().__init__()
        self.groups = groups
        self.unshuffle_factor = unshuffle_factor

        assert channels % groups == 0, f"channels {channels} must be divisible by groups {groups}"
        self.group_channels = channels // groups

        self.softmax = nn.Softmax(dim=-1)
        self.agp = nn.AdaptiveAvgPool2d((1, 1))

        self.x_local_conv1 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=1),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True)
        )
        self.x_local_conv2 = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, groups=channels // 2),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True)
        )

        unshuffle_channels = (channels // 2) * (unshuffle_factor ** 2)

        self.y_pre_proj = nn.Conv2d(unshuffle_channels, channels // factor, kernel_size=1)

        self.ss2d = SS2DCore(
            d_model=channels // factor,
            d_state=8,
            expand=1,
            d_conv=3,
        )

        self.y_post_proj = nn.Conv2d(channels // factor, channels // 2, kernel_size=1)

        self.y_spatial = nn.Sequential(
            nn.Conv2d(channels // 2, channels // 2, kernel_size=3, padding=1, groups=channels // 2),
            nn.BatchNorm2d(channels // 2),
            nn.SiLU(inplace=True)
        )

        self.cross_proj = nn.Conv2d(channels // 2, channels // 2, kernel_size=1)

        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        B, C, H, W = x.shape

        x_half = x[:, :C // 2, :, :]
        y_half = x[:, C // 2:, :, :]

        x_local = self.x_local_conv1(x_half)
        x_local = self.x_local_conv2(x_local)

        y_unshuffle = F.pixel_unshuffle(y_half, self.unshuffle_factor)

        y_proj = self.y_pre_proj(y_unshuffle)
        y_ss2d = self.ss2d(y_proj)
        y_ss2d = self.y_post_proj(y_ss2d)

        y_global = F.interpolate(y_ss2d, size=(H, W), mode='bilinear', align_corners=False)

        y_global = self.y_spatial(y_global)

        x_gap = self.agp(x_local)
        y_gap = self.agp(y_global)

        x_gap_flat = x_gap.view(B, self.groups, -1)
        y_gap_flat = y_gap.view(B, self.groups, -1)

        attn = self.softmax(x_gap_flat @ y_gap_flat.transpose(-1, -2))

        y_global_grouped = y_global.view(B, self.groups, -1, H, W)
        y_attended = torch.einsum('bgk,bkchw->bgchw', attn, y_global_grouped)
        y_attended = y_attended.view(B, C // 2, H, W)

        y_final = self.cross_proj(y_attended)

        out = torch.cat([x_local, y_final], dim=1)
        out = self.out_proj(out)

        return out + x


class EMABlock(nn.Module):

    def __init__(
        self,
        channels: int,
        factor: int = 8,
        groups: int = 4,
        ffn_ratio: float = 4.0,
        drop_path: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(1, channels)
        self.ema = EMA_SS2D(channels, factor=factor, groups=groups)

        self.norm2 = nn.GroupNorm(1, channels)
        ffn_dim = int(channels * ffn_ratio)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, ffn_dim, 1),
            nn.GELU(),
            nn.Conv2d(ffn_dim, channels, 1),
        )

        from timm.models.layers import DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.ema(self.norm1(x)))
        x = x + self.drop_path(self.ffn(self.norm2(x)))
        return x


class ECAAttention(nn.Module):
    def __init__(self, channels, gamma=2, b=1):
        super(ECAAttention, self).__init__()
        k = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = k if k % 2 else k + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)

        y = y.squeeze(-1).transpose(-1, -2)
        y = self.conv(y)
        y = y.transpose(-1, -2).unsqueeze(-1)

        return x * self.sigmoid(y)


class StripSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super(StripSpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_x = x
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_map = self.conv1(x_cat)
        return input_x * self.sigmoid(x_map)

class Dynamic_MSAA(nn.Module):

    def __init__(self, in_channels, out_channels=None, factor=4.0):
        super(Dynamic_MSAA, self).__init__()
        out_channels = out_channels or in_channels
        dim = int(out_channels // factor)
        dim = max(dim, 16)

        self.down = nn.Conv2d(in_channels, dim, kernel_size=1)

        self.branch1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)

        self.branch2 = nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim)

        self.branch3 = nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim)
        self.pw_mix = nn.Conv2d(dim, dim, kernel_size=1)

        self.branch_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(1),
            nn.Linear(dim, dim * 3),
        )

        self.spatial_attention = StripSpatialAttention()
        self.channel_attention = ECAAttention(dim)

        self.up = nn.Conv2d(dim, out_channels, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x):
        identity = x

        x_in = self.down(x)

        b1 = self.branch1(x_in)
        b2 = self.branch2(x_in)
        b3 = self.branch3(x_in)

        attn_logits = self.branch_attn(x_in)
        attn_logits = attn_logits.view(-1, 3, b1.size(1))
        attn_weights = attn_logits.softmax(dim=1)
        w1 = attn_weights[:, 0, :].unsqueeze(-1).unsqueeze(-1)
        w2 = attn_weights[:, 1, :].unsqueeze(-1).unsqueeze(-1)
        w3 = attn_weights[:, 2, :].unsqueeze(-1).unsqueeze(-1)
        x_fused = b1 * w1 + b2 * w2 + b3 * w3

        x_fused = self.pw_mix(x_fused)

        x_spatial = self.spatial_attention(x_fused)

        x_refined = self.channel_attention(x_spatial)

        out = self.up(x_refined)
        out = self.act(out)

        if identity.shape[1] == out.shape[1]:
            out = out + identity

        return out
