from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .transforms import random_crop_pair, to_gray_float_tensor


class VideoFramePairDataset(Dataset):
    """Samples (frame_t, frame_t+k) from mp4 files, where k is random in [gap_min, gap_max]."""
    def __init__(self, video_dir: str, crop_h: int = 315, crop_w: int = 560, gap_min: int = 1, gap_max: int = 5,
                 pairs_per_epoch: int = 12000, seed: int = 42):
        self.video_dir = Path(video_dir)
        self.crop_h, self.crop_w = crop_h, crop_w
        self.gap_min, self.gap_max = int(gap_min), int(gap_max)
        self.pairs_per_epoch = int(pairs_per_epoch)
        self.seed = int(seed)
        self.videos = sorted([p for p in self.video_dir.glob('*.mp4')])
        if not self.videos:
            raise FileNotFoundError(f'No .mp4 videos found in {self.video_dir}')
        self.meta = []
        for p in self.videos:
            cap = cv2.VideoCapture(str(p))
            n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            if n > self.gap_max + 1:
                self.meta.append((p, n))
        if not self.meta:
            raise RuntimeError('No video has enough frames for the requested temporal gap')

    def __len__(self):
        return self.pairs_per_epoch

    @staticmethod
    def _read_frame(path: Path, idx: int):
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f'Could not read frame {idx} from {path}')
        return frame

    def __getitem__(self, index):
        rng = np.random.default_rng(self.seed + int(index) + torch.utils.data.get_worker_info().id * 100000 if torch.utils.data.get_worker_info() else self.seed + int(index))
        path, n = self.meta[int(rng.integers(0, len(self.meta)))]
        gap = int(rng.integers(self.gap_min, self.gap_max + 1))
        t = int(rng.integers(0, n - gap))
        a = self._read_frame(path, t)
        b = self._read_frame(path, t + gap)
        a, b = random_crop_pair(a, b, self.crop_h, self.crop_w, rng)
        return {'ia': to_gray_float_tensor(a), 'ib': to_gray_float_tensor(b), 'video': str(path), 't': t, 'gap': gap}


class LabeledPointPairsDataset(Dataset):
    """Loads validation .npy dictionaries with keys: path1, path2, matche_pts.
    matche_pts format example: [[(x1,y1),(x2,y2)], ...].
    """
    def __init__(self, npy_dir: str, image_root: str):
        self.npy_files = sorted(Path(npy_dir).glob('*.npy'))
        if not self.npy_files:
            raise FileNotFoundError(f'No .npy labels found in {npy_dir}')
        self.image_root = Path(image_root)

    def __len__(self): return len(self.npy_files)

    def __getitem__(self, idx):
        item = np.load(self.npy_files[idx], allow_pickle=True).item()
        p1, p2 = self.image_root / item['path1'], self.image_root / item['path2']
        img1, img2 = cv2.imread(str(p1)), cv2.imread(str(p2))
        if img1 is None or img2 is None:
            raise FileNotFoundError(f'Could not read {p1} or {p2}')
        pts_a = np.array([m[0] for m in item['matche_pts']], dtype=np.float32)
        pts_b = np.array([m[1] for m in item['matche_pts']], dtype=np.float32)
        return {
            'ia': to_gray_float_tensor(img1), 'ib': to_gray_float_tensor(img2),
            'pts_a': torch.from_numpy(pts_a), 'pts_b': torch.from_numpy(pts_b),
            'path1': str(p1), 'path2': str(p2)
        }
