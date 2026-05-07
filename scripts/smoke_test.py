#!/usr/bin/env python
from __future__ import annotations
import sys, tempfile
from pathlib import Path
import cv2, numpy as np, torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.data.video_pair_dataset import VideoFramePairDataset
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss


def make_video(path: Path, n=8, h=96, w=128):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    wr = cv2.VideoWriter(str(path), fourcc, 5, (w, h))
    for i in range(n):
        img = np.zeros((h, w, 3), np.uint8)
        cv2.rectangle(img, (10+i, 20), (50+i, 60), (255, 255, 255), -1)
        cv2.circle(img, (80, 40+i), 15, (128, 128, 128), -1)
        wr.write(img)
    wr.release()


def main():
    # Prevent excessive CPU thread oversubscription during local integrity checks.
    torch.set_num_threads(1)
    torch.manual_seed(0)
    with tempfile.TemporaryDirectory() as td:
        vdir = Path(td) / 'data/train'; vdir.mkdir(parents=True)
        make_video(vdir / '000001.mp4')
        ds = VideoFramePairDataset(str(vdir), crop_h=64, crop_w=96, gap_min=1, gap_max=3, pairs_per_epoch=2)
        batch = ds[0]
        ia = batch['ia'].unsqueeze(0)
        ib = batch['ib'].unsqueeze(0)
        model = ContentAwareHomographyNet()
        model.train()
        out = model(ia, ib, use_attention=False, bidirectional=True)
        criterion = ContentAwareTripletLoss()
        losses = criterion(ia, ib, out, model.feature)
        losses['loss'].backward()
        assert torch.isfinite(losses['loss']).item(), losses['loss']
        assert out['ab']['H'].shape == (1, 3, 3)
        print('SMOKE TEST PASSED')
        print({k: float(v) for k, v in losses.items() if torch.is_tensor(v) and v.ndim == 0})

if __name__ == '__main__': main()
