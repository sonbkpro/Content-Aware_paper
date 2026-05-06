from __future__ import annotations
from contextlib import nullcontext
import torch
import torch.nn.functional as F


_LOW_PRECISION_DTYPES = (torch.float16, torch.bfloat16)


def _geometry_dtype(*tensors: torch.Tensor) -> torch.dtype:
    dtype = tensors[0].dtype
    for tensor in tensors[1:]:
        dtype = torch.promote_types(dtype, tensor.dtype)
    return torch.float32 if dtype in _LOW_PRECISION_DTYPES else dtype


def _disable_autocast(device_type: str):
    if device_type in {'cuda', 'cpu'}:
        return torch.amp.autocast(device_type, enabled=False)
    return nullcontext()


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
    compute_dtype = _geometry_dtype(src, H_src_to_dst)
    with _disable_autocast(src.device.type):
        H = H_src_to_dst.to(compute_dtype)
        ys, xs = torch.meshgrid(
            torch.arange(out_h, device=src.device, dtype=compute_dtype),
            torch.arange(out_w, device=src.device, dtype=compute_dtype),
            indexing='ij',
        )
        ones = torch.ones_like(xs)
        dst = torch.stack([xs, ys, ones], dim=-1).reshape(1, out_h * out_w, 3).repeat(b, 1, 1)
        H_inv = torch.linalg.inv(H)
        src_xyw = dst @ H_inv.transpose(1, 2)
        x = src_xyw[..., 0] / src_xyw[..., 2].clamp_min(1e-8)
        y = src_xyw[..., 1] / src_xyw[..., 2].clamp_min(1e-8)
        x_norm = 2.0 * x / max(w - 1, 1) - 1.0
        y_norm = 2.0 * y / max(h - 1, 1) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).reshape(b, out_h, out_w, 2)
        warped = F.grid_sample(src.to(compute_dtype), grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    return warped.to(src.dtype)
