#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
import cv2, torch
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.utils.checkpoint import load_checkpoint
from src.data.transforms import to_gray_float_tensor
from src.geometry.warp import warp_perspective
from src.utils.visualization import make_alignment_overlay, save_image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--image_a', required=True)
    ap.add_argument('--image_b', required=True)
    ap.add_argument('--out', default='alignment_overlay.png')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ContentAwareHomographyNet().to(device).eval()
    load_checkpoint(args.ckpt, model, map_location=device)
    ia = to_gray_float_tensor(cv2.imread(args.image_a)).unsqueeze(0).to(device)
    ib = to_gray_float_tensor(cv2.imread(args.image_b)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(ia, ib, use_attention=True, bidirectional=False)
        wa = warp_perspective(ia, out['ab']['H'], ib.shape[-2], ib.shape[-1])
    save_image(args.out, make_alignment_overlay(wa, ib))
    print('H_ab=')
    print(out['ab']['H'][0].detach().cpu().numpy())
    print(f'saved {args.out}')

if __name__ == '__main__': main()
