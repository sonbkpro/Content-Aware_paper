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
        weight = (M_warp.float() * M_tgt.float()).clamp_min(0.0)
        numer = (weight * (F_warp.float() - F_tgt.float()).abs()).sum(dim=(1, 2, 3))
        denom = weight.sum(dim=(1, 2, 3)).clamp_min(torch.finfo(weight.dtype).tiny)
        return (numer / denom).mean()

    def forward(self, ia, ib, model_out, feature_extractor, use_mask_weighting: bool = True):
        ab, ba = model_out['ab'], model_out['ba']
        Hab, Hba = ab['H'], ba['H']
        ia_w = warp_perspective(ia, Hab, ib.shape[-2], ib.shape[-1])
        ib_w = warp_perspective(ib, Hba, ia.shape[-2], ia.shape[-1])
        Fa_w = feature_extractor(ia_w)
        Fb_w = feature_extractor(ib_w)
        if use_mask_weighting:
            Ma = ab['Ma']
            Mb = ab['Mb']
            Mba_src = ba['Ma']
            Mba_tgt = ba['Mb']
        else:
            Ma = torch.ones_like(ab['Ma'])
            Mb = torch.ones_like(ab['Mb'])
            Mba_src = torch.ones_like(ba['Ma'])
            Mba_tgt = torch.ones_like(ba['Mb'])
        Ma_w = warp_perspective(Ma, Hab, ib.shape[-2], ib.shape[-1])
        Mb_w = warp_perspective(Mba_src, Hba, ia.shape[-2], ia.shape[-1])
        l_ab = self.normalized_feature_loss(Fa_w, ab['Fb'], Ma_w, Mb)
        l_ba = self.normalized_feature_loss(Fb_w, ba['Fb'], Mb_w, Mba_tgt)
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
