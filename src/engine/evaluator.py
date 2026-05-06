from __future__ import annotations
import torch
from torch.utils.data import DataLoader
from src.geometry.homography_utils import point_l2_error


@torch.no_grad()
def evaluate_labeled_points(model, dataset, device='cuda'):
    model.eval()
    errors = []
    for sample in DataLoader(dataset, batch_size=1, shuffle=False):
        ia = sample['ia'].to(device).float()
        ib = sample['ib'].to(device).float()
        pts_a = sample['pts_a'].to(device).float()
        pts_b = sample['pts_b'].to(device).float()
        out = model(ia, ib, use_attention=True, bidirectional=False)
        err = point_l2_error(pts_a, pts_b, out['ab']['H']).item()
        errors.append(err)
    return {'point_l2_mean': float(sum(errors) / max(len(errors), 1)), 'num_pairs': len(errors)}
