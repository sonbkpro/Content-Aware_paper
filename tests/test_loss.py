import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss


def test_loss_backward():
    m = ContentAwareHomographyNet(max_corner_offset=16)
    ia = torch.rand(1,1,64,96)
    ib = torch.rand(1,1,64,96)
    out = m(ia, ib, use_attention=False, bidirectional=True)
    losses = ContentAwareTripletLoss()(ia, ib, out, m.feature)
    losses['loss'].backward()
    assert torch.isfinite(losses['loss']).item()
