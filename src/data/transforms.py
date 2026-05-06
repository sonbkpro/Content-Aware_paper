from __future__ import annotations
import cv2
import numpy as np
import torch


def resize_if_needed(img: np.ndarray, min_h: int, min_w: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max(min_h / max(h, 1), min_w / max(w, 1), 1.0)
    if scale > 1.0:
        img = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_LINEAR)
    return img


def random_crop_pair(a: np.ndarray, b: np.ndarray, crop_h: int, crop_w: int, rng: np.random.Generator):
    a = resize_if_needed(a, crop_h, crop_w)
    b = resize_if_needed(b, crop_h, crop_w)
    h, w = a.shape[:2]
    if b.shape[:2] != (h, w):
        b = cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    y = int(rng.integers(0, h - crop_h + 1))
    x = int(rng.integers(0, w - crop_w + 1))
    return a[y:y+crop_h, x:x+crop_w], b[y:y+crop_h, x:x+crop_w]


def to_gray_float_tensor(img: np.ndarray) -> torch.Tensor:
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype(np.float32) / 255.0
    return torch.from_numpy(img).unsqueeze(0)
