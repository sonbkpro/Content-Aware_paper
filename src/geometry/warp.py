from __future__ import annotations
import torch
import torch.nn.functional as F


def warp_perspective(src: torch.Tensor, H_src_to_dst: torch.Tensor, out_h: int | None = None, out_w: int | None = None) -> torch.Tensor:
    """Pure PyTorch STN-style inverse warping.
    src: [B,C,H,W]. H maps source pixel coordinates to destination pixel coordinates.
    Returns source warped into the destination canvas.
    """
    if src.ndim != 4:
        raise ValueError('src must be [B,C,H,W]')
    b, _, h, w = src.shape
    out_h = h if out_h is None else out_h
    out_w = w if out_w is None else out_w
    ys, xs = torch.meshgrid(
        torch.arange(out_h, device=src.device, dtype=src.dtype),
        torch.arange(out_w, device=src.device, dtype=src.dtype),
        indexing='ij',
    )
    ones = torch.ones_like(xs)
    dst = torch.stack([xs, ys, ones], dim=-1).reshape(1, out_h * out_w, 3).repeat(b, 1, 1)
    H_inv = torch.linalg.inv(H_src_to_dst)
    src_xyw = dst @ H_inv.transpose(1, 2)
    x = src_xyw[..., 0] / src_xyw[..., 2].clamp_min(1e-8)
    y = src_xyw[..., 1] / src_xyw[..., 2].clamp_min(1e-8)
    x_norm = 2.0 * x / max(w - 1, 1) - 1.0
    y_norm = 2.0 * y / max(h - 1, 1) - 1.0
    grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b, out_h, out_w, 2)
    return F.grid_sample(src, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
