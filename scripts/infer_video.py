#!/usr/bin/env python
from __future__ import annotations
import argparse, sys, cv2, torch
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.utils.checkpoint import load_checkpoint
from src.data.transforms import to_gray_float_tensor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True); ap.add_argument('--video', required=True)
    ap.add_argument('--gap', type=int, default=1); ap.add_argument('--max_pairs', type=int, default=100)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    device = torch.device(args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu')
    model = ContentAwareHomographyNet().to(device).eval(); load_checkpoint(args.ckpt, model, map_location=device)
    cap = cv2.VideoCapture(args.video)
    frames = []
    while len(frames) < args.max_pairs + args.gap:
        ok, f = cap.read()
        if not ok: break
        frames.append(f)
    cap.release()
    for i in range(min(args.max_pairs, len(frames)-args.gap)):
        ia = to_gray_float_tensor(frames[i]).unsqueeze(0).to(device)
        ib = to_gray_float_tensor(frames[i+args.gap]).unsqueeze(0).to(device)
        with torch.no_grad(): H = model(ia, ib, use_attention=True, bidirectional=False)['ab']['H'][0].cpu().numpy()
        print(i, '->', i+args.gap, H.reshape(-1).tolist())

if __name__ == '__main__': main()
