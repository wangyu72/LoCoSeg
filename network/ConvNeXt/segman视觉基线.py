import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from typing import Optional, List, Tuple
from timm.models.layers import DropPath, trunc_normal_

# Dynamic_MSAA 替代 VSSM/SS2D
from .ema_ss2d import Dynamic_MSAA

# ConvNeXt 编码器
from ..ConvNeXt.ConvNeXt import ConvNeXt
# from network.VMamba.vmamba_model import VSSM

# ====================================
# Helper Modules (from official repo)
# ====================================
class LayerNorm2d(nn.LayerNorm):
    """Channel-first LayerNorm for 2D feature maps"""
    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)
        
    def forward(self, x):
        # x: (B, C, H, W) -> (B, H, W, C) -> LayerNorm -> (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class MLP(nn.Module):
    """
    Linear Embedding: github.com/NVlabs/SegFormer
    将高维特征投影到统一维度
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, H*W, C)
        x = self.proj(x)  # (B, H*W, embed_dim)
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)  # (B, embed_dim, H, W)
        return x


class ConvModule(nn.Module):
    """Conv + BN + Act 模块 (替代 MMSeg 的 ConvModule)"""
    def __init__(self, in_channels, out_channels, kernel_size=1, 
                 stride=1, padding=0, groups=1, bias=False):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, 
                              stride, padding, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class LayerScale(nn.Module):
    """Layer Scale for residual scaling"""
    def __init__(self, dim, init_value=1e-5, enable_bias=True):
        super().__init__()
        self.dim = dim
        self.init_value = init_value
        self.enable_bias = enable_bias
        self.weight = nn.Parameter(torch.ones(dim, 1, 1, 1) * init_value, requires_grad=True)
        if enable_bias:
            self.bias = nn.Parameter(torch.zeros(dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x):
        x = F.conv2d(x, weight=self.weight, bias=self.bias, groups=x.shape[1])
        return x


class FFN(nn.Module):
    """Feed-Forward Network (官方结构)"""
    def __init__(self, embed_dim, ffn_dim, act_layer=nn.GELU, dropout=0):
        super().__init__()
        self.fc1 = nn.Conv2d(embed_dim, ffn_dim, kernel_size=1)
        self.act_layer = act_layer()
        self.dwconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.fc2 = nn.Conv2d(ffn_dim, embed_dim, kernel_size=1)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_layer(x)
        x = x + self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ChannelAttention(nn.Module):
    """Simple Squeeze-and-Excitation style channel attention"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        mid = max(1, channels // reduction)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, mid, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.fc(self.avg(x))
        return x * w


# ====================================
# MSAABlock: 替代官方的 VSSMBlock
# 官方 VSSMBlock 使用 VSSM (SS2D) 作为 token_mixer
# 本实现使用 Dynamic_MSAA 替代
# ====================================
class MSAABlock(nn.Module):
    """
    替代官方的 VSSMBlock
    
    官方 VSSMBlock 结构:
    - CPE (Conditional Position Encoding)
    - VSSM (Selective Scan State Space Model) as token_mixer
    - LayerScale
    - FFN (Feed Forward Network)
    - DropPath
    
    本实现使用 Dynamic_MSAA 替代 VSSM，无需 CUDA selective_scan
    """
    def __init__(
        self,
        embed_dim: int = 192,
        expansion_ratio: float = 1.0,
        drop_path: float = 0.0,
        layerscale: bool = False,
        layer_init_values: float = 1e-6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.layerscale = layerscale
        
        # CPE (Conditional Position Encoding)
        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, 
                              padding=1, groups=embed_dim)
        
        # Norm + Token Mixer (Dynamic_MSAA 替代 VSSM)
        self.norm1 = LayerNorm2d(embed_dim)
        self.token_mixer = Dynamic_MSAA(embed_dim, embed_dim, factor=4.0)
        
        # CPE2 + Norm + FFN
        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3, 
                              padding=1, groups=embed_dim)
        self.norm2 = LayerNorm2d(embed_dim)
        self.mlp = FFN(embed_dim, embed_dim * 4)
        
        # DropPath
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        # LayerScale (optional)
        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()
    
    def forward(self, x):
        # CPE1 + Token Mixer (官方结构)
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))
        
        # CPE2 + FFN (官方结构)
        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))
        
        return x


# ====================================
# SegMANDecoder: 完全按照官方仓库结构
# ====================================
class SegMANDecoder(nn.Module):
    """
    SegMAN Decoder - 完全按照官方仓库 segman_decoder.py 结构实现
    
    官方结构:
    1. forward_mlp_decoder: MLP 线性融合 c2, c3, c4
    2. forward_winssm: 
       - short_path + image_pool 分支
       - 多尺度下采样 (conv_downsample_2, conv_downsample_4)
       - Pixel Unshuffle 统一分辨率到 H/4 × W/4
       - reduce_channels 降维
       - VSSMBlock (替换为 MSAABlock)
       - 残差加回 c2, c3, c4
       - Concat 所有分支输出
    """
    def __init__(
        self,
        in_channels: List[int] = [96, 192, 384, 768],
        embed_dim: int = 256,
        feat_proj_dim: int = 256,
        drop_path: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        self.feat_proj_dim = feat_proj_dim
        
        # ========== MLP Decoder (官方: forward_mlp_decoder) ==========
        # 将 c2, c3, c4 投影到 feat_proj_dim
        self.linear_c4 = MLP(in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(in_channels[1], feat_proj_dim)
        
        # 融合 c2, c3, c4 -> embed_dim
        self.linear_fuse = ConvModule(
            in_channels=feat_proj_dim * 3,
            out_channels=embed_dim,
            kernel_size=1,
        )
        
        # ========== 确定性上采样层 (ConvTranspose2d 替代 F.interpolate) ==========
        # _c4: H/32 -> H/8 (4x 上采样)
        self.upsample_c4 = nn.ConvTranspose2d(
            feat_proj_dim, feat_proj_dim,
            kernel_size=4, stride=4, padding=0,
            groups=feat_proj_dim
        )
        # _c3: H/16 -> H/8 (2x 上采样)
        self.upsample_c3 = nn.ConvTranspose2d(
            feat_proj_dim, feat_proj_dim,
            kernel_size=2, stride=2, padding=0,
            groups=feat_proj_dim
        )
        
        # ========== WinSSM 分支 (官方: forward_winssm) ==========
        # 下采样卷积 (官方: conv_downsample_2, conv_downsample_4)
        self.conv_downsample_2 = ConvModule(
            embed_dim, embed_dim * 2,
            kernel_size=3, stride=2, padding=1,
        )
        self.conv_downsample_4 = ConvModule(
            embed_dim, embed_dim * 4,
            kernel_size=5, stride=4, padding=1,
        )
        
        # reduce_channels: 将 unshuffle 后的高通道特征降维 (官方结构)
        # x_4: embed_dim * 4 -> embed_dim
        # x_2_unshuffle: embed_dim * 2 * 4 = embed_dim * 8 -> embed_dim (pixel_unshuffle factor=2)
        # x_unshuffle: embed_dim * 16 -> embed_dim (pixel_unshuffle factor=4)
        self.reduce_channels = nn.ModuleList([
            ConvModule(embed_dim * 4, embed_dim, kernel_size=1),      # x_4
            ConvModule(embed_dim * 8, embed_dim, kernel_size=1),      # x_2_unshuffle
            ConvModule(embed_dim * 16, embed_dim, kernel_size=1),     # x_unshuffle
        ])

        # 通道注意力模块（在 pixel_unshuffle 后对高维通道进行加权再降维）
        # 减少一次性大幅压缩导致的信息丢失
        self.ca_modules = nn.ModuleList([
            ChannelAttention(embed_dim * 4, reduction=16),
            ChannelAttention(embed_dim * 8, reduction=16),
            ChannelAttention(embed_dim * 16, reduction=16),
        ])
        
        # MSAABlock 替代 VSSMBlock (官方: self.vssm = VSSMBlock)
        vssm_dim = embed_dim * 3  # concat(x_unshuffle, x_2_unshuffle, x_4)
        self.vssm = MSAABlock(
            embed_dim=vssm_dim,
            expansion_ratio=1,
            drop_path=drop_path,
            layerscale=False,
        )
        
        # 确定性上采样：VSSM 输出 H/4 -> H (4x 上采样)
        self.upsample_vssm = nn.ConvTranspose2d(
            vssm_dim, vssm_dim,
            kernel_size=4, stride=4, padding=0,
            groups=vssm_dim
        )
        
        # short_path 分支 (官方结构)
        self.short_path = ConvModule(
            embed_dim, embed_dim, kernel_size=1,
        )
        
        # image_pool 分支 (官方: 全局上下文)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim, kernel_size=1),
        )
        
        # proj_out: 将 VSSM 输出投影回 feat_proj_dim (官方结构)
        self.proj_out = ConvModule(
            vssm_dim, feat_proj_dim, kernel_size=1,
        )
        
        # 最终 concat 融合 (官方: feat_concat_dim)
        # out = [short_path, image_pool, vssm_out, c2+proj, c3+proj, c4+proj]
        # 通道数: embed_dim + embed_dim + vssm_dim + feat_proj_dim*3
        feat_concat_dim = embed_dim * 2 + vssm_dim + feat_proj_dim * 3
        self.cat = ConvModule(
            feat_concat_dim, embed_dim, kernel_size=1,
        )
    
    def forward_mlp_decoder(self, inputs):
        """
        MLP Decoder: 融合 c2, c3, c4 多尺度特征 (官方结构)
        
        Args:
            inputs: [c1, c2, c3, c4] 编码器特征
            
        Returns:
            _c: 融合后的特征 (B, embed_dim, H/8, W/8)
            _c2, _c3, _c4: 投影后的特征
        """
        c1, c2, c3, c4 = inputs
        
        # 投影到统一维度
        _c4 = self.linear_c4(c4)  # (B, feat_proj_dim, H/32, W/32)
        _c3 = self.linear_c3(c3)  # (B, feat_proj_dim, H/16, W/16)
        _c2 = self.linear_c2(c2)  # (B, feat_proj_dim, H/8, W/8)
        
        # 上采样到 c2 分辨率 (H/8) - 使用确定性 ConvTranspose2d
        _c4 = self.upsample_c4(_c4)  # 4x 上采样
        _c3 = self.upsample_c3(_c3)  # 2x 上采样
        
        # 融合
        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))
        
        return _c, _c2, _c3, _c4
    
    def forward_winssm(self, x, c2, c3, c4):
        """
        WinSSM 分支 (官方核心，使用 MSAABlock 替代 VSSMBlock)
        
        Args:
            x: 融合后的特征 (B, embed_dim, H/8, W/8)
            c2, c3, c4: 投影后的多尺度特征
            
        Returns:
            out: 解码器输出 (B, embed_dim, H/8, W/8)
        """
        B, C, H, W = x.shape
        
        # ===== 分支1: short_path (官方) =====
        out_short = self.short_path(x)  # (B, embed_dim, H, W)
        
        # ===== 分支2: image_pool (全局上下文，官方) =====
        out_pool = self.image_pool(x)  # (B, embed_dim, 1, 1)
        # 确定性广播：使用 expand 替代 F.interpolate
        out_pool = out_pool.expand(-1, -1, H, W)  # (B, embed_dim, H, W)
        
        # ===== 分支3: 多尺度 VSSM (官方核心，用 MSAABlock 替代) =====
        # 多尺度特征提取
        x_2 = self.conv_downsample_2(x)  # (B, embed_dim*2, H/2, W/2)
        x_4 = self.conv_downsample_4(x)  # (B, embed_dim*4, H/4, W/4)
        
        # Pixel Unshuffle 统一分辨率到 H/4 (官方结构)
        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)  # (B, embed_dim*8, H/4, W/4)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)      # (B, embed_dim*16, H/4, W/4)
        
        # reduce_channels 降维 (官方结构)
        # 在降维前先做通道注意力加权，缓解一次性大幅压缩的信息丢失
        x_4 = self.ca_modules[0](x_4) if hasattr(self, 'ca_modules') else x_4
        x_2_unshuffle = self.ca_modules[1](x_2_unshuffle) if hasattr(self, 'ca_modules') else x_2_unshuffle
        x_unshuffle = self.ca_modules[2](x_unshuffle) if hasattr(self, 'ca_modules') else x_unshuffle

        # reduce_channels 降维 (官方结构)
        x_4 = self.reduce_channels[0](x_4)           # (B, embed_dim, H/4, W/4)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)  # (B, embed_dim, H/4, W/4)
        x_unshuffle = self.reduce_channels[2](x_unshuffle)      # (B, embed_dim, H/4, W/4)
        
        # Concat 多尺度特征 (官方结构)
        multi_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)  # (B, embed_dim*3, H/4, W/4)
        
        # MSAABlock (官方: self.vssm(multi_x))
        _out = self.vssm(multi_x)  # (B, embed_dim*3, H/4, W/4)
        
        # 确定性上采样：使用 ConvTranspose2d 替代 F.interpolate (4x 上采样)
        _out = self.upsample_vssm(_out)  # (B, embed_dim*3, H, W)
        
        # 投影并残差加回 c2, c3, c4 (官方结构)
        _out_ = self.proj_out(_out)  # (B, feat_proj_dim, H, W)
        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_
        
        # ===== Concat 所有分支 (官方结构) =====
        # [short_path, image_pool, vssm_out, c2, c3, c4]
        out = torch.cat([out_short, out_pool, _out, c2, c3, c4], dim=1)
        out = self.cat(out)
        
        return out
    
    def forward(self, inputs):
        """
        Args:
            inputs: [c1, c2, c3, c4] 编码器特征
                   c1: (B, C1, H/4, W/4)
                   c2: (B, C2, H/8, W/8)
                   c3: (B, C3, H/16, W/16)
                   c4: (B, C4, H/32, W/32)
        
        Returns:
            out: (B, embed_dim, H/8, W/8)
        """
        x, c2, c3, c4 = self.forward_mlp_decoder(inputs)
        out = self.forward_winssm(x, c2, c3, c4)
        return out


# ====================================
# SegMAN UNet: Full Segmentation Network
# ====================================
class SegMANUNet(nn.Module):
    """
    SegMAN UNet: Full Segmentation Network
    
    Architecture (完全按照官方 yunxiangfu2001/SegMAN 仓库结构):
    - Encoder: ConvNeXt backbone (可替换)
    - Decoder: SegMANDecoder (官方结构, VSSM→Dynamic_MSAA)
    - Head: 8x upsampling segmentation head
    
    Key Changes vs Official:
    - VSSM (SS2D selective scan) → Dynamic_MSAA (纯 PyTorch)
    - 编码器使用 ConvNeXt (官方使用 SegMAN Encoder)
    - 移除 MMSeg 依赖，独立实现
    """
    def __init__(
        self,
        in_chans: int = 3,
        num_classes: int = 1,
        embed_dims: List[int] = [96, 192, 384, 768],
        depths: List[int] = [3, 3, 9, 3],
        decoder_embed_dim: int = 256,
        decoder_feat_proj_dim: int = 256,
        drop_path_rate: float = 0.1,
        deep_supervision: bool = False,
    ):
    
        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.embed_dims = embed_dims
        
        # Encoder: ConvNeXt
        self.encoder = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=embed_dims,
            drop_path_rate=drop_path_rate,
        )
        
        # Decoder: SegMANDecoder (官方结构)
        self.decoder = SegMANDecoder(
            in_channels=embed_dims,
            embed_dim=decoder_embed_dim,
            feat_proj_dim=decoder_feat_proj_dim,
            drop_path=drop_path_rate,
        )
        
        # Segmentation Head (8x upsampling: H/8 -> H)
        # 使用 ConvTranspose2d 替代 Upsample 以确保确定性
        self.head = nn.Sequential(
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim, 3, padding=1),
            nn.BatchNorm2d(decoder_embed_dim),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim, decoder_embed_dim, kernel_size=2, stride=2),  # H/8 -> H/4
            nn.Conv2d(decoder_embed_dim, decoder_embed_dim // 2, 3, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 2),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim // 2, decoder_embed_dim // 2, kernel_size=2, stride=2),  # H/4 -> H/2
            nn.Conv2d(decoder_embed_dim // 2, decoder_embed_dim // 4, 3, padding=1),
            nn.BatchNorm2d(decoder_embed_dim // 4),
            nn.GELU(),
            nn.ConvTranspose2d(decoder_embed_dim // 4, decoder_embed_dim // 4, kernel_size=2, stride=2),  # H/2 -> H
            nn.Conv2d(decoder_embed_dim // 4, num_classes, 1),
        )
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入图像
            
        Returns:
            out: (B, num_classes, H, W) 分割输出
        """
        # Encoder: ConvNeXt 返回 (features[::-1], pos_embed)
        # features[::-1] = [c4, c3, c2, c1] (从深到浅)
        encoder_out, _ = self.encoder(x)
        
        # 翻转为 [c1, c2, c3, c4] (从浅到深)
        features = encoder_out[::-1]  # [c1(H/4), c2(H/8), c3(H/16), c4(H/32)]
        
        # Decoder
        dec_out = self.decoder(features)  # (B, embed_dim, H/8, W/8)
        
        # Head
        out = self.head(dec_out)
        
        return out
    
    def load_from(self, config):
        """Load pretrained ConvNeXt encoder weights from config"""
        pretrained_path = config.MODEL.PRETRAIN_CKPT if hasattr(config.MODEL, 'PRETRAIN_CKPT') else None
        if pretrained_path:
            print(f"Loading ConvNeXt pretrained weights from: {pretrained_path}")
            self.encoder.load_from(pretrained_path)
    
    def load_pretrained(self, path: str):
        """Load pretrained encoder weights"""
        if path and path != "":
            print(f"Loading ConvNeXt pretrained weights from: {path}")
            self.encoder.load_from(path)


# ====================================
# Model Factory Function
# ====================================
def build_segman(
    num_classes: int = 1,
    in_chans: int = 3,
    drop_path_rate: float = 0.1,         # 保留兼容性（已在 MSAABlock 中使用 Dynamic_MSAA）
    deep_supervision: bool = False,  # 保留兼容性
    pretrained: str = None,
    **kwargs
):
    """
    Build SegMAN model with ConvNeXt-Tiny encoder
    
    Architecture (完全按照官方 yunxiangfu2001/SegMAN 仓库):
    - Encoder: ConvNeXt-Tiny [96, 192, 384, 768], depths [3, 3, 9, 3]
    - Decoder: SegMANDecoder (官方结构)
        - forward_mlp_decoder: MLP 融合 c2, c3, c4
        - forward_winssm: 多尺度 pixel_unshuffle + MSAABlock
    - MSAABlock: Dynamic_MSAA 替代官方 VSSM (SS2D)
    
    Args:
        num_classes: Number of output classes
        in_chans: Number of input channels
        drop_path_rate: Stochastic depth rate
        use_ema: 兼容参数（Dynamic_MSAA 默认启用）
        deep_supervision: 是否启用深度监督
        pretrained: Path to pretrained ConvNeXt weights
        
    Returns:
        SegMANUNet model
    """
    model = SegMANUNet(
        num_classes=num_classes,
        in_chans=in_chans,
        embed_dims=[96, 192, 384, 768],  # ConvNeXt-Tiny dims
        depths=[3, 3, 9, 3],              # ConvNeXt-Tiny depths
        decoder_embed_dim=256,            # 官方默认值
        decoder_feat_proj_dim=256,        # 官方默认值
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
    )
    
    if pretrained:
        model.load_pretrained(pretrained)
    
    return model
