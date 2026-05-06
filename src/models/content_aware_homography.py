from __future__ import annotations
import torch
import torch.nn as nn
from .feature_extractor import FeatureExtractor
from .mask_predictor import MaskPredictor
from .homography_estimator import ResNet34HomographyEstimator


class ContentAwareHomographyNet(nn.Module):
    def __init__(self, feature_channels: int = 1, max_corner_offset: float | None = 128.0):
        super().__init__()
        self.feature = FeatureExtractor(1, feature_channels)
        self.mask = MaskPredictor(1)
        self.estimator = ResNet34HomographyEstimator(2 * feature_channels, max_corner_offset)

    def encode(self, x):
        F = self.feature(x)
        M = self.mask(x)
        return F, M

    def estimate(self, ia, ib, use_attention: bool = True):
        Fa, Ma = self.encode(ia)
        Fb, Mb = self.encode(ib)
        Ga = Fa * Ma if use_attention else Fa
        Gb = Fb * Mb if use_attention else Fb
        Hab, off = self.estimator(torch.cat([Ga, Gb], dim=1))
        return {'H': Hab, 'offsets': off, 'Fa': Fa, 'Fb': Fb, 'Ma': Ma, 'Mb': Mb, 'Ga': Ga, 'Gb': Gb}

    def forward(self, ia, ib, use_attention: bool = True, bidirectional: bool = True):
        ab = self.estimate(ia, ib, use_attention=use_attention)
        out = {'ab': ab}
        if bidirectional:
            out['ba'] = self.estimate(ib, ia, use_attention=use_attention)
        return out
