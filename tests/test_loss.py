import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.data.transforms import make_h4p, make_patch_indices


def make_batch():
    org = torch.rand(1, 2, 96, 128)
    x, y, ph, pw = 16, 16, 64, 96
    return {
        'org_images': org,
        'input_tensors': org[:, :, y:y + ph, x:x + pw],
        'h4p': make_h4p(x, y, ph, pw).unsqueeze(0),
        'patch_indices': make_patch_indices(x, y, ph, pw, 128).unsqueeze(0),
    }


def test_loss_backward():
    m = ContentAwareHomographyNet()
    batch = make_batch()
    out = m.forward_oneline(**batch, use_attention=True, use_mask_weighting=False)
    losses = ContentAwareTripletLoss()(out)
    losses['loss'].backward()
    assert torch.isfinite(losses['loss']).item()


def test_stage1_uses_unit_loss_mask():
    m = ContentAwareHomographyNet()
    batch = make_batch()
    out = m.forward_oneline(**batch, use_attention=True, use_mask_weighting=False)
    assert torch.all(out['mask_ap'] == 1)


def test_triplet_loss_is_nonnegative():
    criterion = ContentAwareTripletLoss()
    anchor = torch.zeros(1, 1, 4, 4)
    positive = torch.full((1, 1, 4, 4), 3.0)
    negative = torch.ones(1, 1, 4, 4)
    loss_map = criterion.loss_map(anchor, positive, negative)
    assert torch.all(loss_map >= 0.0)
