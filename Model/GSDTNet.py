"""GSDTNet: Geometry-aware Semantic-Detail Transformer Network.

This file keeps the best-performing v4b_res_region path from the uploaded
ablation code and removes all other ablation branches. The retained model
matches the manuscript terminology:

    RGB + pseudo-depth -> PseudoRGBDAdapter -> EfficientNet-B0 encoder
    -> GSDT semantic-detail-geometric recalibration -> CGG-style decoder.

No model logic from the v4b_res_region path is changed: the pseudo-RGB-D
adapter, EfficientNet-B0 feature levels, GSDT geometry gate, zero-initialized
semantic region residual branch, detail injection scale, and decoder are kept.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.EfficientNet import EfficientNet_B0
from Model.modules import BasicConv2d, DepthwiseSeparableConv


class PseudoRGBDAdapter(nn.Module):
    """Lightweight 1x1 adapter A([I, D]) for pseudo-RGB-D input."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(4, 3, kernel_size=1, bias=False),
            nn.BatchNorm2d(3),
            nn.GELU(),
        )

    def forward(self, rgb, depth):
        return self.proj(torch.cat([rgb, depth], dim=1))


class FeatureProjector(nn.Module):
    """Lightweight channel projection P_i used before decoder fusion."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Sequential(
            BasicConv2d(in_channels, out_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class GSDT(nn.Module):
    """Geometry-aware Semantic-Detail Transformer.

    The implementation is the final v4b_res_region branch:
      - shallow detail feature from F1 and F2;
      - high-level semantic feature from F4 and F5;
      - semantic-detail cross attention;
      - pseudo-depth-derived geometry gate (DGG);
      - zero-initialized semantic region residual compensation (RRC).

    The zero-initialized `region_scale` keeps the initial behavior identical to
    the boundary-gated GSDT path and lets the region branch become active only
    when training learns it is useful.
    """

    def __init__(self, f1_channels, f2_channels, f4_channels, f5_channels,
                 out_channels=32, num_heads=4, ffn_ratio=2, eps=1e-6):
        super().__init__()
        self.eps = eps

        self.detail_f1 = BasicConv2d(
            f1_channels, out_channels, kernel_size=3, stride=2, padding=1
        )
        self.detail_f2 = BasicConv2d(
            f2_channels, out_channels, kernel_size=1
        )
        self.detail_fuse = nn.Sequential(
            BasicConv2d(out_channels * 2, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.semantic_f4 = BasicConv2d(
            f4_channels, out_channels, kernel_size=1
        )
        self.semantic_f5 = BasicConv2d(
            f5_channels, out_channels, kernel_size=1
        )
        self.semantic_fuse = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.q_norm = nn.LayerNorm(out_channels)
        self.kv_norm = nn.LayerNorm(out_channels)
        self.attn = nn.MultiheadAttention(
            embed_dim=out_channels,
            num_heads=num_heads,
            batch_first=True,
        )
        hidden_channels = out_channels * ffn_ratio
        self.ffn_norm = nn.LayerNorm(out_channels)
        self.ffn = nn.Sequential(
            nn.Linear(out_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, out_channels),
        )

        self.attn_proj = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.depth_guided_geometry_gate = nn.Sequential(
            nn.Conv2d(1, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

        self.semantic_region_gate = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid(),
        )

        self.out_proj = nn.Sequential(
            BasicConv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )

        self.attn_scale = nn.Parameter(torch.tensor(0.10))
        self.region_scale = nn.Parameter(torch.tensor(0.0))

        sobel_x = torch.tensor(
            [[[-1.0, 0.0, 1.0],
              [-2.0, 0.0, 2.0],
              [-1.0, 0.0, 1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        sobel_y = torch.tensor(
            [[[-1.0, -2.0, -1.0],
              [ 0.0,  0.0,  0.0],
              [ 1.0,  2.0,  1.0]]],
            dtype=torch.float32,
        ).unsqueeze(0)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_y, persistent=False)

    def _depth_geometry(self, depth, size):
        kx = self.sobel_x.to(device=depth.device, dtype=depth.dtype)
        ky = self.sobel_y.to(device=depth.device, dtype=depth.dtype)

        gx = F.conv2d(depth, kx, padding=1)
        gy = F.conv2d(depth, ky, padding=1)
        grad = torch.sqrt(gx * gx + gy * gy + self.eps)
        grad = grad / (grad.amax(dim=(2, 3), keepdim=True) + self.eps)
        grad = F.interpolate(grad, size=size, mode='bilinear', align_corners=False)
        return grad

    @staticmethod
    def _flatten_tokens(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _unflatten_tokens(tokens, height, width):
        return tokens.transpose(1, 2).contiguous().view(
            tokens.shape[0], tokens.shape[2], height, width
        )

    def forward(self, f1, f2, f4, f5, depth, return_aux=False):
        detail_f1 = self.detail_f1(f1)
        if detail_f1.shape[2:] != f2.shape[2:]:
            detail_f1 = F.interpolate(detail_f1, size=f2.shape[2:], mode='bilinear', align_corners=False)
        detail_f2 = self.detail_f2(f2)
        detail = self.detail_fuse(torch.cat([detail_f1, detail_f2], dim=1))

        semantic_f4 = self.semantic_f4(f4)
        semantic_f5 = self.semantic_f5(f5)
        semantic_f5 = F.interpolate(semantic_f5, size=semantic_f4.shape[2:], mode='bilinear', align_corners=False)
        semantic = self.semantic_fuse(semantic_f4 + semantic_f5)

        detail_low = F.adaptive_avg_pool2d(detail, output_size=semantic.shape[2:])
        _, _, hs, ws = semantic.shape

        q = self.q_norm(self._flatten_tokens(detail_low))
        kv = self.kv_norm(self._flatten_tokens(semantic))
        attn_tokens, _ = self.attn(q, kv, kv, need_weights=False)
        attn_tokens = attn_tokens + self.ffn(self.ffn_norm(attn_tokens))
        attn_map = self._unflatten_tokens(attn_tokens, hs, ws)

        attn_map = F.interpolate(attn_map, size=detail.shape[2:], mode='bilinear', align_corners=False)
        semantic_guided_detail = self.attn_proj(attn_map)

        geometry_map = self._depth_geometry(depth, size=detail.shape[2:])
        geometry_gate = self.depth_guided_geometry_gate(geometry_map)
        geometry_residual = self.attn_scale.to(dtype=detail.dtype) * semantic_guided_detail * geometry_gate

        semantic_up = F.interpolate(semantic, size=detail.shape[2:], mode='bilinear', align_corners=False)
        region_gate = self.semantic_region_gate(semantic_up)
        region_residual = self.region_scale.to(dtype=detail.dtype) * semantic_guided_detail * region_gate

        refined = detail + geometry_residual + region_residual
        delta_f = self.out_proj(refined)

        if return_aux:
            aux = {
                "detail_feature": detail.detach(),
                "semantic_feature": semantic.detach(),
                "semantic_guided_detail": semantic_guided_detail.detach(),
                "pseudo_depth_geometry": geometry_map.detach(),
                "depth_guided_geometry_gate": geometry_gate.detach(),
                "semantic_region_gate": region_gate.detach(),
                "geometry_residual": geometry_residual.detach(),
                "region_residual": region_residual.detach(),
                "gsdt_residual": delta_f.detach(),
                "attn_scale": self.attn_scale.detach().float().view(1),
                "region_scale": self.region_scale.detach().float().view(1),
            }
            return delta_f, aux

        return delta_f


class CGGDecoder(nn.Module):
    """Lightweight top-down decoder with mask and edge heads.

    This keeps the decoder logic from the v4b_res_region code path. The module
    name follows the manuscript, where CGG denotes the lightweight refinement
    block used after top-down feature aggregation.
    """

    def __init__(self, channels=(32, 64, 96, 160), fpn_channels=64, use_edge_refine=True):
        super().__init__()
        c1, c2, c3, c4 = channels
        self.use_edge_refine = use_edge_refine
        self.l4 = BasicConv2d(c4, fpn_channels, kernel_size=1)
        self.l3 = BasicConv2d(c3, fpn_channels, kernel_size=1)
        self.l2 = BasicConv2d(c2, fpn_channels, kernel_size=1)
        self.l1 = BasicConv2d(c1, fpn_channels, kernel_size=1)
        self.s4 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s3 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s2 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.s1 = DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1)
        self.edge_proj = nn.Sequential(
            nn.Conv2d(1, fpn_channels, 1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )
        self.refine1 = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
        )
        self.mask1 = nn.Conv2d(fpn_channels, 1, 1)
        self.mask2 = nn.Conv2d(fpn_channels, 1, 1)
        self.mask3 = nn.Conv2d(fpn_channels, 1, 1)
        self.mask4 = nn.Conv2d(fpn_channels, 1, 1)
        self.edge_head = nn.Sequential(
            DepthwiseSeparableConv(fpn_channels, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.GELU(),
            nn.Conv2d(fpn_channels, 1, 1),
        )

    def forward(self, feats, rect_edge, out_size):
        f2, f3, f4, f5 = feats
        p5 = self.s4(self.l4(f5))
        p4 = self.s3(self.l3(f4) + F.interpolate(p5, size=f4.shape[2:], mode='bilinear', align_corners=False))
        p3 = self.s2(self.l2(f3) + F.interpolate(p4, size=f3.shape[2:], mode='bilinear', align_corners=False))
        p2 = self.s1(self.l1(f2) + F.interpolate(p3, size=f2.shape[2:], mode='bilinear', align_corners=False))
        edge1 = F.interpolate(rect_edge, size=p2.shape[2:], mode='bilinear', align_corners=False)
        if self.use_edge_refine:
            p2 = self.refine1(p2 + self.edge_proj(edge1))
        else:
            p2 = self.refine1(p2)
        m1 = F.interpolate(self.mask1(p2), size=out_size, mode='bilinear', align_corners=False)
        m2 = F.interpolate(self.mask2(p3), size=out_size, mode='bilinear', align_corners=False)
        m3 = F.interpolate(self.mask3(p4), size=out_size, mode='bilinear', align_corners=False)
        m4 = F.interpolate(self.mask4(p5), size=out_size, mode='bilinear', align_corners=False)
        edge = F.interpolate(self.edge_head(p2), size=out_size, mode='bilinear', align_corners=False)
        return m1, m2, m3, m4, edge


class GSDTNet(nn.Module):
    """Final GSDTNet model, retaining only the v4b_res_region path."""

    def __init__(self, pretrained=True, fuse_channels=(32, 64, 96, 160)):
        super().__init__()
        self.pseudo_rgbd_adapter = PseudoRGBDAdapter()
        self.encoder = EfficientNet_B0(pretrained=pretrained)
        stage_channels = self.encoder.get_stage_channels()

        # Decoder uses F2/F3/F4/F5, corresponding to EfficientNet stages r3/r4/r6/r8.
        projector_stage_indices = [1, 2, 3, 4]
        self.feature_projectors = nn.ModuleList([
            FeatureProjector(stage_channels[i], fuse_channels[j])
            for j, i in enumerate(projector_stage_indices)
        ])

        self.gsdt = GSDT(
            f1_channels=stage_channels[0],
            f2_channels=stage_channels[1],
            f4_channels=stage_channels[3],
            f5_channels=stage_channels[4],
            out_channels=fuse_channels[0],
            num_heads=4,
        )
        self.detail_inject_scale = nn.Parameter(torch.tensor(0.20))
        self.decoder = CGGDecoder(fuse_channels, 64, use_edge_refine=True)

    def _project_decoder_features(self, f2, f3, f4, f5):
        return (
            self.feature_projectors[0](f2),
            self.feature_projectors[1](f3),
            self.feature_projectors[2](f4),
            self.feature_projectors[3](f5),
        )

    def forward(self, rgb, depth, return_aux=False):
        rectified_depth = depth
        rect_edge = torch.zeros_like(depth)

        x = self.pseudo_rgbd_adapter(rgb, rectified_depth)
        f1, f2, f3, f4, f5 = self.encoder(x)

        dec_f2, dec_f3, dec_f4, dec_f5 = self._project_decoder_features(f2, f3, f4, f5)

        if return_aux:
            delta_f, gsdt_aux = self.gsdt(f1, f2, f4, f5, depth, return_aux=True)
        else:
            delta_f = self.gsdt(f1, f2, f4, f5, depth, return_aux=False)
            gsdt_aux = None

        dec_f2 = dec_f2 + self.detail_inject_scale.to(dtype=dec_f2.dtype) * delta_f
        m1, m2, m3, m4, edge_pred = self.decoder((dec_f2, dec_f3, dec_f4, dec_f5), rect_edge, rgb.shape[2:])

        if return_aux:
            gsdt_aux = dict(gsdt_aux)
            gsdt_aux.update({
                "detail_inject_scale": self.detail_inject_scale.detach().float().view(1),
                "enhanced_f2": dec_f2.detach(),
            })
            return m1, m2, m3, m4, edge_pred, rectified_depth, rect_edge, {"gsdt": gsdt_aux}

        return m1, m2, m3, m4, edge_pred
