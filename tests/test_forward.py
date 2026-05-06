import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import torch
from src.models.content_aware_homography import ContentAwareHomographyNet


def test_forward_shapes():
    m = ContentAwareHomographyNet()
    ia = torch.rand(1,1,64,96)
    ib = torch.rand(1,1,64,96)
    out = m(ia, ib, use_attention=True, bidirectional=True)
    assert out['ab']['H'].shape == (1,3,3)
    assert out['ab']['Ma'].shape == ia.shape
    assert out['ba']['H'].shape == (1,3,3)
