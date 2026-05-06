#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.utils.checkpoint import load_checkpoint
from src.data.video_pair_dataset import LabeledPointPairsDataset
from src.engine.evaluator import evaluate_labeled_points


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--npy_dir', required=True)
    ap.add_argument('--image_root', required=True)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ContentAwareHomographyNet().to(device)
    load_checkpoint(args.ckpt, model, map_location=device)
    ds = LabeledPointPairsDataset(args.npy_dir, args.image_root)
    print(evaluate_labeled_points(model, ds, device))

if __name__ == '__main__': main()
