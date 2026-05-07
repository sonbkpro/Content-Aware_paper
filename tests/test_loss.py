import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss


def test_loss_backward():
    m = ContentAwareHomographyNet()
    ia = torch.rand(1,1,64,96)
    ib = torch.rand(1,1,64,96)
    out = m(ia, ib, use_attention=False, bidirectional=True)
    losses = ContentAwareTripletLoss()(ia, ib, out, m.feature)
    losses['loss'].backward()
    assert torch.isfinite(losses['loss']).item()


def test_stage1_loss_ignores_collapsed_learned_masks():
    m = ContentAwareHomographyNet()
    ia = torch.rand(1,1,64,96)
    ib = torch.rand(1,1,64,96)
    out = m(ia, ib, use_attention=False, bidirectional=True)
    for direction in ('ab', 'ba'):
        out[direction]['Ma'] = torch.zeros_like(out[direction]['Ma'])
        out[direction]['Mb'] = torch.zeros_like(out[direction]['Mb'])
    criterion = ContentAwareTripletLoss()
    masked = criterion(ia, ib, out, m.feature, use_mask_weighting=True)['l_ab']
    unmasked = criterion(ia, ib, out, m.feature, use_mask_weighting=False)['l_ab']
    assert masked.item() == 0.0
    assert unmasked.item() > 0.0


def test_normalized_feature_loss_stays_scale_invariant_for_tiny_masks():
    criterion = ContentAwareTripletLoss()
    F_warp = torch.ones(1, 1, 4, 4)
    F_tgt = torch.zeros(1, 1, 4, 4)
    full = torch.ones(1, 1, 4, 4)
    tiny = torch.full((1, 1, 4, 4), 1e-6)
    full_loss = criterion.normalized_feature_loss(F_warp, F_tgt, full, full)
    tiny_loss = criterion.normalized_feature_loss(F_warp, F_tgt, tiny, tiny)
    assert torch.allclose(tiny_loss, full_loss)


def test_triplet_loss_is_nonnegative():
    criterion = ContentAwareTripletLoss()
    anchor = torch.zeros(1, 1, 4, 4)
    positive = torch.full((1, 1, 4, 4), 3.0)
    negative = torch.ones(1, 1, 4, 4)
    mask = torch.ones(1, 1, 4, 4)
    loss = criterion.normalized_triplet_loss(anchor, positive, negative, mask, mask)
    assert loss.item() >= 0.0
