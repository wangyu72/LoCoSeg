import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List
from timm.models.layers import DropPath, trunc_normal_

from .ema_ss2d import Dynamic_MSAA

from ..ConvNeXt.ConvNeXt import ConvNeXt

class LayerNorm2d(nn.LayerNorm):

    def __init__(self, dim):
        super().__init__(normalized_shape=dim, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)
        return x.contiguous()


class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        n, _, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 1).reshape(n, -1, h, w)
        return x


class ConvModule(nn.Module):

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


class MSAABlock(nn.Module):

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

        self.cpe1 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3,
                              padding=1, groups=embed_dim)

        self.norm1 = LayerNorm2d(embed_dim)
        self.token_mixer = Dynamic_MSAA(embed_dim, embed_dim, factor=4.0)

        self.cpe2 = nn.Conv2d(embed_dim, embed_dim, kernel_size=3,
                              padding=1, groups=embed_dim)
        self.norm2 = LayerNorm2d(embed_dim)
        self.mlp = FFN(embed_dim, embed_dim * 4)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

        if layerscale:
            self.layer_scale1 = LayerScale(embed_dim, init_value=layer_init_values)
            self.layer_scale2 = LayerScale(embed_dim, init_value=layer_init_values)
        else:
            self.layer_scale1 = nn.Identity()
            self.layer_scale2 = nn.Identity()

    def forward(self, x):
        x = x + self.cpe1(x)
        token_mix_feat = self.token_mixer(self.norm1(x))
        x = x + self.drop_path(self.layer_scale1(token_mix_feat))

        x = x + self.cpe2(x)
        x = x + self.drop_path(self.layer_scale2(self.mlp(self.norm2(x))))

        return x


class LoCoSegDecoder(nn.Module):

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

        self.linear_c4 = MLP(in_channels[3], feat_proj_dim)
        self.linear_c3 = MLP(in_channels[2], feat_proj_dim)
        self.linear_c2 = MLP(in_channels[1], feat_proj_dim)

        self.linear_fuse = ConvModule(
            in_channels=feat_proj_dim * 3,
            out_channels=embed_dim,
            kernel_size=1,
        )

        self.upsample_c4 = nn.ConvTranspose2d(
            feat_proj_dim, feat_proj_dim,
            kernel_size=4, stride=4, padding=0,
            groups=feat_proj_dim
        )
        self.upsample_c3 = nn.ConvTranspose2d(
            feat_proj_dim, feat_proj_dim,
            kernel_size=2, stride=2, padding=0,
            groups=feat_proj_dim
        )

        self.conv_downsample_2 = ConvModule(
            embed_dim, embed_dim * 2,
            kernel_size=3, stride=2, padding=1,
        )
        self.conv_downsample_4 = ConvModule(
            embed_dim, embed_dim * 4,
            kernel_size=5, stride=4, padding=1,
        )

        self.reduce_channels = nn.ModuleList([
            ConvModule(embed_dim * 4, embed_dim, kernel_size=1),
            ConvModule(embed_dim * 8, embed_dim, kernel_size=1),
            ConvModule(embed_dim * 16, embed_dim, kernel_size=1),
        ])

        self.ca_modules = nn.ModuleList([
            ChannelAttention(embed_dim * 4, reduction=16),
            ChannelAttention(embed_dim * 8, reduction=16),
            ChannelAttention(embed_dim * 16, reduction=16),
        ])

        vssm_dim = embed_dim * 3
        self.vssm = MSAABlock(
            embed_dim=vssm_dim,
            expansion_ratio=1,
            drop_path=drop_path,
            layerscale=False,
        )

        self.upsample_vssm = nn.ConvTranspose2d(
            vssm_dim, vssm_dim,
            kernel_size=4, stride=4, padding=0,
            groups=vssm_dim
        )

        self.short_path = ConvModule(
            embed_dim, embed_dim, kernel_size=1,
        )

        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvModule(embed_dim, embed_dim, kernel_size=1),
        )

        self.proj_out = ConvModule(
            vssm_dim, feat_proj_dim, kernel_size=1,
        )

        feat_concat_dim = embed_dim * 2 + vssm_dim + feat_proj_dim * 3
        self.cat = ConvModule(
            feat_concat_dim, embed_dim, kernel_size=1,
        )

    def forward_mlp_decoder(self, inputs):

        c1, c2, c3, c4 = inputs

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)

        _c4 = self.upsample_c4(_c4)
        _c3 = self.upsample_c3(_c3)

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))

        return _c, _c2, _c3, _c4

    def forward_winssm(self, x, c2, c3, c4):

        B, C, H, W = x.shape

        out_short = self.short_path(x)

        out_pool = self.image_pool(x)
        out_pool = out_pool.expand(-1, -1, H, W)

        x_2 = self.conv_downsample_2(x)
        x_4 = self.conv_downsample_4(x)

        x_2_unshuffle = F.pixel_unshuffle(x_2, downscale_factor=2)
        x_unshuffle = F.pixel_unshuffle(x, downscale_factor=4)

        x_4 = self.ca_modules[0](x_4)
        x_2_unshuffle = self.ca_modules[1](x_2_unshuffle)
        x_unshuffle = self.ca_modules[2](x_unshuffle)

        x_4 = self.reduce_channels[0](x_4)
        x_2_unshuffle = self.reduce_channels[1](x_2_unshuffle)
        x_unshuffle = self.reduce_channels[2](x_unshuffle)

        multi_x = torch.cat([x_unshuffle, x_2_unshuffle, x_4], dim=1)

        _out = self.vssm(multi_x)

        _out = self.upsample_vssm(_out)

        _out_ = self.proj_out(_out)
        c2 = c2 + _out_
        c3 = c3 + _out_
        c4 = c4 + _out_

        out = torch.cat([out_short, out_pool, _out, c2, c3, c4], dim=1)
        out = self.cat(out)

        return out

    def forward(self, inputs):

        x, c2, c3, c4 = self.forward_mlp_decoder(inputs)
        out = self.forward_winssm(x, c2, c3, c4)
        return out


class OmniSegController(nn.Module):

    def __init__(self, feat_dim=768, num_tasks=6, head_channels=8):
        super().__init__()
        self.feat_dim = feat_dim
        self.num_tasks = num_tasks
        self.head_channels = head_channels

        weight_nums = [
            head_channels * head_channels,
            head_channels * head_channels,
            head_channels * 1,
        ]
        bias_nums = [head_channels, head_channels, 1]
        total_params = sum(weight_nums) + sum(bias_nums)

        self.controller = nn.Sequential(
            nn.Linear(feat_dim + num_tasks, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, total_params),
        )

    def encoding_task(self, task_id):

        N = task_id.shape[0]
        task_encoding = torch.zeros(N, self.num_tasks, device=task_id.device)
        task_encoding.scatter_(1, task_id.long().unsqueeze(1), 1)
        return task_encoding

    def parse_dynamic_params(self, params):

        B = params.size(0)
        head_c = self.head_channels

        w0 = params[:, :64].reshape(B * head_c, head_c, 1, 1)
        w1 = params[:, 64:128].reshape(B * head_c, head_c, 1, 1)
        w2 = params[:, 128:136].reshape(B * 1, head_c, 1, 1)

        b0 = params[:, 136:144].reshape(-1)
        b1 = params[:, 144:152].reshape(-1)
        b2 = params[:, 152:153].reshape(-1)

        return [w0, w1, w2], [b0, b1, b2]

    def forward(self, head_inputs, global_feat, task_id):

        B = global_feat.size(0)

        task_encoding = self.encoding_task(task_id)

        controller_input = torch.cat([global_feat, task_encoding], dim=1)
        params = self.controller(controller_input)

        weights, biases = self.parse_dynamic_params(params)

        x = head_inputs.reshape(1, -1, head_inputs.size(2), head_inputs.size(3))
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(x, w, bias=b, stride=1, padding=0, groups=B)
            if i < len(weights) - 1:
                x = F.relu(x)

        logits = x.reshape(B, 1, head_inputs.size(2), head_inputs.size(3))
        return logits

class OmniSegControllerMulti(nn.Module):

    def __init__(self, feat_dim=768, num_tasks=6, head_channels=8):
        super().__init__()
        self.num_tasks = num_tasks
        self.single = OmniSegController(
            feat_dim=feat_dim,
            num_tasks=num_tasks,
            head_channels=head_channels,
        )

    def forward(self, head_inputs, global_feat):

        B = global_feat.size(0)
        all_logits = []
        for t in range(self.num_tasks):
            tids = torch.full((B,), t, dtype=torch.long, device=global_feat.device)
            logits_t = self.single(head_inputs, global_feat, tids)
            all_logits.append(logits_t)
        return torch.cat(all_logits, dim=1)


class SkipSGLCModule(nn.Module):

    def __init__(
        self,
        deep_channels: int,
        shallow_channels: int,
        num_classes: int,
        kernel_size: int = 7,
        learnable_kernel: bool = True,
        normalize_kernel: bool = True,
        use_global_baseline: bool = True,
        global_momentum: float = 0.01,
        delta: float = 0.05,
        lambda_clip: float = 4.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.kernel_size = kernel_size
        self.learnable_kernel = learnable_kernel
        self.normalize_kernel = normalize_kernel
        self.use_global_baseline = use_global_baseline
        self.global_momentum = global_momentum
        self.delta = delta
        self.lambda_clip = lambda_clip
        self.eps = 1e-6

        self.cls_proj = nn.Conv2d(deep_channels, num_classes, kernel_size=1)

        padding = kernel_size // 2
        self.spatial_diffusion = nn.Conv2d(
            num_classes, num_classes,
            kernel_size=kernel_size,
            padding=padding,
            groups=num_classes,
            bias=False
        )
        self.register_buffer(
            "gauss_base", self._build_gaussian_kernel(kernel_size), persistent=False
        )
        self.reset_diffusion_kernel()

        self.relation = nn.Parameter(torch.randn(num_classes, num_classes) * 0.05)

        off_diag = 1.0 - torch.eye(num_classes)
        self.register_buffer("off_diag", off_diag, persistent=False)
        self.register_buffer(
            "global_co", off_diag / max(num_classes - 1, 1), persistent=True
        )

        self.shallow_detail_extractor = nn.Sequential(
            nn.Conv2d(shallow_channels, 1, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, 1),
            nn.Sigmoid()
        )
        self.shallow_gain = nn.Parameter(torch.zeros(1))

        self.attn_proj = nn.Conv2d(num_classes, deep_channels, kernel_size=1, bias=False)
        trunc_normal_(self.attn_proj.weight, std=0.02)
        self.gamma = nn.Parameter(torch.full((deep_channels,), 1e-3))
        self.prior_beta = nn.Parameter(torch.full((num_classes,), 1e-3))

    @staticmethod
    def _build_gaussian_kernel(kernel_size: int, sigma_scale: float = 3.0) -> torch.Tensor:

        sigma = max(kernel_size / sigma_scale, 1e-6)
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, x, indexing='ij')
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)

    def reset_diffusion_kernel(self):

        with torch.no_grad():
            self.spatial_diffusion.weight.copy_(
                self.gauss_base.expand_as(self.spatial_diffusion.weight)
            )
        self.spatial_diffusion.weight.requires_grad_(self.learnable_kernel)

    def forward(self, x_deep: torch.Tensor, x_shallow: torch.Tensor,
                return_prior: bool = False):

        B, C_d, H_d, W_d = x_deep.shape
        _, _, H, W = x_shallow.shape
        K = self.num_classes

        if K < 2:
            prior = torch.zeros(B, K, H, W, device=x_deep.device, dtype=x_deep.dtype)
            return (x_deep, prior) if return_prior else x_deep

        if (H_d, W_d) != (H, W):
            x_deep = F.interpolate(x_deep, size=(H, W), mode='bilinear', align_corners=False)

        P = torch.sigmoid(self.cls_proj(x_deep))

        kernel = self.spatial_diffusion.weight
        if self.normalize_kernel:
            kernel = F.relu(kernel)
            kernel = kernel / (kernel.sum(dim=(1, 2, 3), keepdim=True) + self.eps)
        P_tilde = F.conv2d(P, kernel, padding=self.kernel_size // 2, groups=K)

        co_occurrence = 0.5 * (P.unsqueeze(2) * P_tilde.unsqueeze(1)
                               + P_tilde.unsqueeze(2) * P.unsqueeze(1))
        off = self.off_diag.view(1, K, K, 1, 1)
        co_occurrence = co_occurrence * off

        row_sum = co_occurrence.sum(dim=2, keepdim=True)
        R = (co_occurrence + self.delta) / (row_sum + (K - 1) * self.delta + self.eps)

        if self.use_global_baseline:
            with torch.no_grad():
                if self.training:
                    cur = (R.detach() * off).mean(dim=(0, 3, 4))
                    self.global_co.mul_(1.0 - self.global_momentum).add_(
                        self.global_momentum * cur
                    )
            base = self.global_co.clamp(min=self.delta).view(1, K, K, 1, 1)
            Lambda = torch.log(R.clamp_min(1e-12)) - torch.log(base)
        else:
            Lambda = torch.log(R.clamp_min(1e-12))
        Lambda = Lambda.clamp(-self.lambda_clip, self.lambda_clip) * off

        u = torch.einsum("bijhw,ij->bihw", Lambda, self.relation) * P

        spatial_detail_gate = self.shallow_detail_extractor(x_shallow)
        u = u * (1.0 + self.shallow_gain * (2.0 * spatial_detail_gate - 1.0))

        gate = self.attn_proj(torch.tanh(u))
        out = x_deep * (1.0 + self.gamma.view(1, C_d, 1, 1) * gate)

        prior = torch.tanh(u)

        return (out, prior) if return_prior else out


class LoCoSegUNet(nn.Module):

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
        head_hidden_dim: int = 8,
        multi_task_output: bool = True,
        use_sglc: bool = True,
        sglc_kernel_size: int = 7,
        sglc_learnable_kernel: bool = True,
        sglc_use_global_baseline: bool = True,
        sglc_global_momentum: float = 0.01,
        sglc_delta: float = 0.05,
        sglc_use_prior: bool = True,
    ):

        super().__init__()
        self.num_classes = num_classes
        self.deep_supervision = deep_supervision
        self.embed_dims = embed_dims
        self.head_hidden_dim = head_hidden_dim
        self.multi_task_output = multi_task_output

        self.encoder = ConvNeXt(
            in_chans=in_chans,
            depths=depths,
            dims=embed_dims,
            drop_path_rate=drop_path_rate,
        )

        self.decoder = LoCoSegDecoder(
            in_channels=embed_dims,
            embed_dim=decoder_embed_dim,
            feat_proj_dim=decoder_feat_proj_dim,
            drop_path=drop_path_rate,
        )

        self.use_sglc = use_sglc
        self.sglc_use_prior = sglc_use_prior
        if use_sglc:
            self.sglc = SkipSGLCModule(
                deep_channels=decoder_embed_dim,
                shallow_channels=embed_dims[0],
                num_classes=num_classes,
                kernel_size=sglc_kernel_size,
                learnable_kernel=sglc_learnable_kernel,
                use_global_baseline=sglc_use_global_baseline,
                global_momentum=sglc_global_momentum,
                delta=sglc_delta,
            )

        upsample_specs = [
            (decoder_embed_dim, decoder_embed_dim),
            (decoder_embed_dim, decoder_embed_dim // 2),
            (decoder_embed_dim // 2, decoder_embed_dim // 4),
        ] if not use_sglc else [
            (decoder_embed_dim, decoder_embed_dim // 2),
            (decoder_embed_dim // 2, decoder_embed_dim // 4),
        ]
        upsample_path = []
        for in_ch, out_ch in upsample_specs:
            upsample_path += [
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                nn.ConvTranspose2d(out_ch, out_ch, kernel_size=2, stride=2),
            ]
        self.upsample_path = nn.Sequential(*upsample_path)
        feat_in_ch = decoder_embed_dim // 4

        self.norm_encoder_out = LayerNorm2d(embed_dims[3])
        self.norm_decoder_out = LayerNorm2d(feat_in_ch)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.feat_proj = nn.Conv2d(feat_in_ch, self.head_hidden_dim, kernel_size=1, bias=False)

        if multi_task_output:
            self.controller = OmniSegControllerMulti(
                feat_dim=embed_dims[3],
                num_tasks=num_classes,
                head_channels=head_hidden_dim,
            )
        else:
            self.controller = OmniSegController(
                feat_dim=embed_dims[3],
                num_tasks=num_classes,
                head_channels=head_hidden_dim,
            )

        self.apply(self._init_weights)

        if self.use_sglc:
            self.sglc.reset_diffusion_kernel()

    def _init_weights(self, m):

        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, task_id=None):

        if not self.multi_task_output and task_id is None:
            raise ValueError("task_id must be provided for single-task Omni-Seg head")

        encoder_out, _ = self.encoder(x)

        c4 = encoder_out[0]

        features = encoder_out[::-1]

        dec_out = self.decoder(features)

        sglc_prior = None
        if self.use_sglc:
            c1 = features[0]
            if self.sglc_use_prior:
                dec_out, sglc_prior = self.sglc(dec_out, c1, return_prior=True)
            else:
                dec_out = self.sglc(dec_out, c1)

        up_out = self.upsample_path(dec_out)

        encoder_out_norm = self.norm_encoder_out(c4)
        global_feat = self.gap(encoder_out_norm)
        global_feat = global_feat.squeeze(-1).squeeze(-1)

        feat_norm = self.norm_decoder_out(up_out)
        head_inputs = self.feat_proj(feat_norm)

        if self.multi_task_output:
            logits = self.controller(head_inputs, global_feat)
        else:
            logits = self.controller(head_inputs, global_feat, task_id)

        if sglc_prior is not None:
            if sglc_prior.shape[-2:] != logits.shape[-2:]:
                sglc_prior = F.interpolate(
                    sglc_prior, size=logits.shape[-2:], mode='bilinear', align_corners=False
                )
            if self.multi_task_output:
                logits = logits + self.sglc.prior_beta.view(1, -1, 1, 1) * sglc_prior
            else:
                bs = sglc_prior.size(0)
                tid = task_id.long().view(bs, 1, 1, 1).expand(bs, 1, *sglc_prior.shape[-2:])
                prior_t = sglc_prior.gather(1, tid)
                beta_t = self.sglc.prior_beta[task_id.long()].view(bs, 1, 1, 1)
                logits = logits + beta_t * prior_t

        return logits

    def load_from(self, config):

        pretrained_path = config.MODEL.PRETRAIN_CKPT if hasattr(config.MODEL, 'PRETRAIN_CKPT') else None
        if pretrained_path:
            print(f"Loading ConvNeXt pretrained weights from: {pretrained_path}")
            self.encoder.load_from(pretrained_path)

    def load_pretrained(self, path: str):

        if path and path != "":
            print(f"Loading ConvNeXt pretrained weights from: {path}")
            self.encoder.load_from(path)


def build_locoseg(
    num_classes: int = 1,
    in_chans: int = 3,
    drop_path_rate: float = 0.1,
    deep_supervision: bool = False,
    pretrained: str = None,
    head_hidden_dim: int = 8,
    multi_task_output: bool = True,
    use_sglc: bool = True,
    sglc_kernel_size: int = 7,
    sglc_learnable_kernel: bool = True,
    sglc_use_global_baseline: bool = True,
    sglc_global_momentum: float = 0.01,
    sglc_delta: float = 0.05,
    sglc_use_prior: bool = True,
    **kwargs
):

    model = LoCoSegUNet(
        num_classes=num_classes,
        in_chans=in_chans,
        embed_dims=[96, 192, 384, 768],
        depths=[3, 3, 9, 3],
        decoder_embed_dim=256,
        decoder_feat_proj_dim=256,
        drop_path_rate=drop_path_rate,
        deep_supervision=deep_supervision,
        head_hidden_dim=head_hidden_dim,
        multi_task_output=multi_task_output,
        use_sglc=use_sglc,
        sglc_kernel_size=sglc_kernel_size,
        sglc_learnable_kernel=sglc_learnable_kernel,
        sglc_use_global_baseline=sglc_use_global_baseline,
        sglc_delta=sglc_delta,
        sglc_global_momentum=sglc_global_momentum,
        sglc_use_prior=sglc_use_prior,
    )

    if pretrained:
        model.load_pretrained(pretrained)

    return model
