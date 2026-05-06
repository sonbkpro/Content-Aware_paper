from __future__ import annotations
import torch
import torch.nn as nn
from src.geometry.warp import warp_perspective
from src.geometry.homography_utils import inverse_consistency_loss


class ContentAwareTripletLoss(nn.Module):
    """Implements Eq. 4, 5, 6 from Content-Aware Unsupervised Deep Homography Estimation."""
    def __init__(self, lambda_triplet: float = 2.0, mu_inverse: float = 0.01, eps: float = 1e-6):
        super().__init__()
        self.lambda_triplet = lambda_triplet
        self.mu_inverse = mu_inverse
        self.eps = eps

    def normalized_feature_loss(self, F_warp, F_tgt, M_warp, M_tgt):
        weight = (M_warp * M_tgt).clamp_min(0.0)
        numer = (weight * (F_warp - F_tgt).abs()).sum(dim=(1, 2, 3))
        denom = weight.sum(dim=(1, 2, 3)).clamp_min(self.eps)
        return (numer / denom).mean()

    def forward(self, ia, ib, model_out, feature_extractor):
        ab, ba = model_out['ab'], model_out['ba']
        Hab, Hba = ab['H'], ba['H']
        ia_w = warp_perspective(ia, Hab, ib.shape[-2], ib.shape[-1])
        ib_w = warp_perspective(ib, Hba, ia.shape[-2], ia.shape[-1])
        Fa_w = feature_extractor(ia_w)
        Fb_w = feature_extractor(ib_w)
        Ma_w = warp_perspective(ab['Ma'], Hab, ib.shape[-2], ib.shape[-1])
        Mb_w = warp_perspective(ba['Ma'], Hba, ia.shape[-2], ia.shape[-1])
        l_ab = self.normalized_feature_loss(Fa_w, ab['Fb'], Ma_w, ab['Mb'])
        l_ba = self.normalized_feature_loss(Fb_w, ba['Fb'], Mb_w, ba['Mb'])
        discriminative = (ab['Fa'] - ab['Fb']).abs().mean()
        inv = inverse_consistency_loss(Hab, Hba)
        total = l_ab + l_ba - self.lambda_triplet * discriminative + self.mu_inverse * inv
        return {
            'loss': total,
            'l_ab': l_ab.detach(),
            'l_ba': l_ba.detach(),
            'discriminative': discriminative.detach(),
            'inverse': inv.detach(),
            'ia_warp': ia_w.detach(),
            'ib_warp': ib_w.detach(),
        }
