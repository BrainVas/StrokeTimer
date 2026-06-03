# -*- coding: utf-8 -*-
"""
ResNeXt3D backbone + disentanglement (q(z|x,s), decoder g) + semantic head.
NOW with an MLP projector + L2 normalization for supervised contrastive learning.

- Stage-1: normal forward (reconstruction + style) to fill caches for runner's losses.
- Stage-2/3: set cls_only=True to skip reconstruction/style branches & caches.

Forward returns:
    (feat_semantic, fmap)  # feat_semantic for CE; runner calls projector() to get contrastive features.

Caches (only when cls_only=False):
    cache_z, cache_mu, cache_logvar, cache_x_hat
    cache_s_vec (semantic vector before projector/L2) [for orthogonality]
"""

from __future__ import annotations
from collections import OrderedDict
from contextlib import nullcontext
from os import path
from typing import Optional, Tuple, Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------- basic 3D block ----------------
class ConvBNAct3D(nn.Module):
    def __init__(self, c_in, c_out, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv3d(c_in, c_out, k, s, p, bias=False)
        self.bn = nn.InstanceNorm3d(c_out, affine=True)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ---------------- ResNeXt bottleneck 3D ----------------
class ResNeXtBottleneck3D(nn.Module):
    expansion = 4

    def __init__(
        self, inplanes: int, planes: int,
        stride=(1, 1, 1), downsample: Optional[nn.Module] = None,
        base_width: int = 4, cardinality: int = 32
    ):
        super().__init__()
        width = int(planes * (base_width / 64.0)) * cardinality

        self.conv1 = nn.Conv3d(inplanes, width, 1, bias=False)
        self.bn1 = nn.InstanceNorm3d(width, affine=True)

        self.conv2 = nn.Conv3d(width, width, 3, stride=stride, padding=1,
                               groups=cardinality, bias=False)
        self.bn2 = nn.InstanceNorm3d(width, affine=True)

        self.conv3 = nn.Conv3d(width, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.InstanceNorm3d(planes * self.expansion, affine=True)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.relu(out + identity)
        return out


# ---------------- ResNeXt3D backbone ----------------
class ResNeXt3D(nn.Module):
    """
    Spatial OS=16, depth stride=1 to preserve D.
    Output: C5 (B,2048,D',H',W')
    """
    def __init__(self, in_ch=1, layers=(3, 4, 6, 3), base_width=4, cardinality=32):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(in_ch, 64, 7, stride=(1, 2, 2), padding=3, bias=False)
        self.bn1 = nn.InstanceNorm3d(64, affine=True)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(3, stride=(1, 2, 2), padding=1)

        self.layer1 = self._make_layer(ResNeXtBottleneck3D, 64, layers[0], (1, 1, 1), base_width, cardinality)
        self.layer2 = self._make_layer(ResNeXtBottleneck3D, 128, layers[1], (1, 2, 2), base_width, cardinality)
        self.layer3 = self._make_layer(ResNeXtBottleneck3D, 256, layers[2], (1, 2, 2), base_width, cardinality)
        self.layer4 = self._make_layer(ResNeXtBottleneck3D, 512, layers[3], (1, 1, 1), base_width, cardinality)

    def _make_layer(self, block, planes, blocks, stride, base_width, cardinality):
        downsample = None
        if stride != (1, 1, 1) or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.InstanceNorm3d(planes * block.expansion, affine=True),
            )

        layers = [block(self.inplanes, planes, stride, downsample, base_width, cardinality)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, (1, 1, 1), None, base_width, cardinality))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)   # (B,2048,D',H',W')
        return x


# ---------------- Modality encoder q(z|x,s) ----------------
class ModalityEncoder3D(nn.Module):
    def __init__(self, in_ch_x=1, in_ch_s=512, hid=256, z_dim=128):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBNAct3D(in_ch_x + in_ch_s, hid, 3, 1, 1),
            nn.MaxPool3d(2),
            ConvBNAct3D(hid, hid, 3, 1, 1),
            nn.AdaptiveAvgPool3d(1),
        )
        self.fc_mu = nn.Linear(hid, z_dim)
        self.fc_lv = nn.Linear(hid, z_dim)

    def forward(self, x_small, s_feat):
        # x_small, s_feat: (B, C, D, H, W)
        h = self.enc(torch.cat([x_small, s_feat], dim=1)).flatten(1)
        mu = self.fc_mu(h)
        lv = self.fc_lv(h).clamp(-6.0, 2.0)
        std = (0.5 * lv).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, lv


# ---------------- Decoder (memory-safe) ----------------
class Decoder3D(nn.Module):
    def __init__(self, in_ch=128, z_dim=128, out_ch=1, out_scale=0.5, amp=True, mode="trilinear"):
        super().__init__()
        assert 0.0 < out_scale <= 1.0
        self.out_scale = float(out_scale)
        self.amp = bool(amp)
        self.mode = str(mode)

        self.to_gamma = nn.Linear(z_dim, in_ch)
        self.to_beta = nn.Linear(z_dim, in_ch)

        self.refine = nn.Sequential(
            ConvBNAct3D(in_ch, 256, 3, 1, 1),
            ConvBNAct3D(256, 128, 3, 1, 1),
        )
        self.head = nn.Conv3d(128, out_ch, 1, bias=True)

    def forward(self, s_feat, z, out_dhw):
        # s_feat: (B,in_ch,D',H',W'), z: (B,z_dim), out_dhw: (D,H,W)
        gamma = self.to_gamma(z).view(z.size(0), -1, 1, 1, 1)
        beta = self.to_beta(z).view(z.size(0), -1, 1, 1, 1)
        y = s_feat * (1.0 + gamma) + beta

        cm = torch.amp.autocast("cuda", dtype=torch.float16) if (self.amp and y.is_cuda) else nullcontext()
        with cm:
            y = self.refine(y)
            D, H, W = out_dhw
            if self.out_scale < 1.0:
                out_small = (
                    max(1, int(D * self.out_scale)),
                    max(1, int(H * self.out_scale)),
                    max(1, int(W * self.out_scale)),
                )
            else:
                out_small = out_dhw
            try:
                y = F.interpolate(
                    y,
                    size=out_small,
                    mode=self.mode,
                    align_corners=False if self.mode != "nearest" else None,
                )
            except RuntimeError:
                y = F.interpolate(y, size=out_small, mode="nearest")
            x_hat_small = self.head(y)

        if self.out_scale < 1.0 and out_small != out_dhw:
            x_hat = F.interpolate(
                x_hat_small.float(),
                size=out_dhw,
                mode="trilinear",
                align_corners=False,
            )
        else:
            x_hat = x_hat_small.float()
        return x_hat


# ---------------- Full model ----------------
class ResNeXt3DOrthogonal(nn.Module):
    """
    Exposes:
      - head_s: semantic MLP head -> feat_dim (for CE)
      - proj_head: MLP projector for contrastive features
      - projector(feat): proj_head + L2-normalization (for supervised contrastive / mean-shift)
      - caches for disent losses when cls_only=False
    """

    def __init__(
        self,
        in_channels: int = 1,
        feat_dim: int = 256,
        z_dim: int = 128,
        con_dim: int = 128,
        use_bn_head: bool = True,
        base_width: int = 4,
        cardinality: int = 32,
        stage2_classifier_only: bool = False,
        decoder_out_scale: float = 0.5,
        decoder_amp: bool = True,
        decoder_mode: str = "trilinear",
        # orthogonality helper dims (kept for compatibility)
        ortho_dim: int = 64,
        ortho_mode: str = "xcov",
        ortho_center: bool = True,
        ortho_unitnorm: bool = True,
    ):
        super().__init__()

        self._cls_only_default = bool(stage2_classifier_only)

        # backbone
        self.backbone = ResNeXt3D(
            in_ch=in_channels,
            layers=(3, 4, 6, 3),
            base_width=base_width,
            cardinality=cardinality,
        )

        # neck and twin convs
        self.neck_reduce = nn.Sequential(
            nn.Conv3d(2048, 256, 1, bias=False),
            nn.InstanceNorm3d(256, affine=True),
            nn.ReLU(inplace=True),
        )
        self.cb_block = ConvBNAct3D(256, 256, 3, 1, 1)
        self.rb_block = ConvBNAct3D(256, 256, 3, 1, 1)

        self.avgpool = nn.AdaptiveAvgPool3d(1)

        # semantic head -> feat_dim (for CE)
        if use_bn_head:
            self.head_s = nn.Sequential(
                nn.Linear(512, 512, bias=False),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_dim),
            )
        else:
            self.head_s = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, feat_dim),
            )

        # channel gate from semantic to fmap
        self.s_gate = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.Sigmoid(),
        )

        # -------- NEW: MLP projector for supervised contrastive features --------
        # Output dim = con_dim, which can be <= feat_dim (e.g., 128).
        self.con_dim = int(con_dim)
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, self.con_dim),
        )

        # s reducer for decoder
        self.s_reducer = nn.Sequential(
            nn.Conv3d(512, 128, 1, bias=False),
            nn.InstanceNorm3d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        # modality encoder and decoder
        self.modenc = ModalityEncoder3D(
            in_ch_x=in_channels,
            in_ch_s=512,
            hid=256,
            z_dim=z_dim,
        )
        self.decoder = Decoder3D(
            in_ch=128,
            z_dim=z_dim,
            out_ch=in_channels,
            out_scale=decoder_out_scale,
            amp=decoder_amp,
            mode=decoder_mode,
        )

        # expose sizes
        self.out_dim = feat_dim
        self.z_dim = z_dim

        # caches
        self.cache_z = None
        self.cache_mu = None
        self.cache_logvar = None
        self.cache_x_hat = None
        self.cache_s_vec = None

        self._last_fmap = None
        self._last_sfeat128 = None

    # ---- trunk ----
    def _trunk(self, x):
        x = self.backbone(x)       # (B,2048,.,.,.)
        x = self.neck_reduce(x)    # (B,256,.,.,.)
        return x

    # ---- helpers ----
    @staticmethod
    def l2_normalize(z: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        """L2-normalize along feature dimension."""
        n = torch.norm(z, p=2, dim=1, keepdim=True).clamp_min(eps)
        return z / n

    def projector(self, feat: torch.Tensor) -> torch.Tensor:
        """
        MLP projector + L2 normalization.

        Input:
            feat : semantic feature from head_s, shape (B, feat_dim).

        Output:
            z    : contrastive feature on the unit sphere, shape (B, con_dim).
        """
        z = self.proj_head(feat)
        z = self.l2_normalize(z)
        return z

    # ---- forward ----
    def forward(self, x, *, cls_only: bool = False, **kwargs):
        """
        Args:
            x: (B, C, D, H, W)
            cls_only: if True, skip modality encoder / decoder and only compute semantic branch.
        """
        B, _, D, H, W = x.shape

        f = self._trunk(x)
        f_cb = self.cb_block(f)
        f_rb = self.rb_block(f)
        fmap = torch.cat([f_cb, f_rb], dim=1)    # (B,512,.,.,.)
        self._last_fmap = fmap

        g = self.avgpool(fmap).flatten(1)        # (B,512)
        feat_s = self.head_s(g)                  # (B,feat_dim)
        # cache raw semantic vector for orthogonality loss
        self.cache_s_vec = feat_s.detach()

        if cls_only or self._cls_only_default:
            # skip reconstruction/style branches completely
            self.cache_z = None
            self.cache_mu = None
            self.cache_logvar = None
            self.cache_x_hat = None
            self._last_sfeat128 = None
            return feat_s, fmap

        # Stage-1 path: reconstruction + style disentanglement
        gate = self.s_gate(feat_s).view(B, 512, 1, 1, 1)
        fmap_s = fmap * gate

        Dp, Hp, Wp = fmap.shape[-3:]
        x_small = F.adaptive_avg_pool3d(x, output_size=(Dp, Hp, Wp))
        z, mu, logvar = self.modenc(x_small, fmap_s)

        sfeat128 = self.s_reducer(fmap_s)
        x_hat = self.decoder(sfeat128, z, out_dhw=(D, H, W))

        # caches for disentanglement losses
        self.cache_z = z
        self.cache_mu = mu
        self.cache_logvar = logvar
        self.cache_x_hat = x_hat
        self._last_sfeat128 = sfeat128

        return feat_s, fmap

    @torch.no_grad()
    def reencode_z(self):

        if (self.cache_x_hat is None) or (self._last_sfeat128 is None) or (self._last_fmap is None):
            return None

        xh = self.cache_x_hat
        Dp, Hp, Wp = self._last_fmap.shape[-3:]
        xh_small = F.adaptive_avg_pool3d(xh, output_size=(Dp, Hp, Wp))

        # reuse reduced s features; pad back to 512 channels for modenc
        sfeat128 = self._last_sfeat128.detach()
        up = F.pad(sfeat128, (0, 0, 0, 0, 0, 0, 0, 384))   # (B,512,.,.,.)
        z_re, _, _ = self.modenc(xh_small, up)
        return z_re

    @torch.no_grad()
    def feat_dim(self) -> int:
        """Return semantic feature dimension (before projector)."""
        return self.out_dim


# ---------------- Factory ----------------
def create_model(
    in_channels: int = 1,
    feat_dim: int = 256,
    z_dim: int = 128,
    con_dim: int = 128,
    use_bn_head: bool = True,
    pretrain: bool = False,
    model_dir: Optional[str] = None,
    stage1_weights: bool = False,
    fix: bool = False,
    stage2_classifier_only: bool = False,
    base_width: int = 4,
    cardinality: int = 32,
    decoder_out_scale: float = 0.5,
    decoder_amp: bool = True,
    decoder_mode: str = "trilinear",
    ortho_dim: int = 64,
    ortho_mode: str = "xcov",
    ortho_center: bool = True,
    ortho_unitnorm: bool = True,
    *args,
    **kwargs,
):
    """
    Factory function used by your runner/config. Keeps existing arguments for compatibility.
    """
    print("Loading ResNeXt3DOrthogonal (backbone + semantic head + projector + q/decoder).")
    net = ResNeXt3DOrthogonal(
        in_channels=in_channels,
        feat_dim=feat_dim,
        z_dim=z_dim,
        con_dim=con_dim,
        use_bn_head=use_bn_head,
        base_width=base_width,
        cardinality=cardinality,
        stage2_classifier_only=stage2_classifier_only,
        decoder_out_scale=decoder_out_scale,
        decoder_amp=decoder_amp,
        decoder_mode=decoder_mode,
        ortho_dim=ortho_dim,
        ortho_mode=ortho_mode,
        ortho_center=ortho_center,
        ortho_unitnorm=ortho_unitnorm,
    )

    # Optional load from previous checkpoint (Stage-1, etc.)
    ckpt_path = None
    if model_dir:
        ckpt_path = model_dir if path.isfile(model_dir) else path.join(model_dir, "final_model_checkpoint.pth")

    if pretrain and ckpt_path and path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("state_dict_best", ckpt)
        sd = sd.get("feat_model", sd)

        new_sd = OrderedDict()
        for k, v in sd.items():
            k2 = k[7:] if k.startswith("module.") else k
            if k2 in net.state_dict() and net.state_dict()[k2].shape == v.shape:
                new_sd[k2] = v

        net.load_state_dict(new_sd, strict=False)
        print("[ResNeXt3DOrthogonal] Loaded pretrain weights partially.")
    else:
        if pretrain:
            print(f"[ResNeXt3DOrthogonal] Pretrain asked but not found: {ckpt_path}")

    if fix:
        for p in net.parameters():
            p.requires_grad = False
        print("[ResNeXt3DOrthogonal] All parameters are frozen (fix=True).")

    return net
