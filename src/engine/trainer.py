from __future__ import annotations
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.data.video_pair_dataset import VideoFramePairDataset, LabeledPointPairsDataset
from src.models.content_aware_homography import ContentAwareHomographyNet
from src.losses.triplet_homography_loss import ContentAwareTripletLoss
from src.utils.checkpoint import save_checkpoint, load_checkpoint
from .evaluator import evaluate_labeled_points


class Trainer:
    def __init__(self, cfg, resume=None):
        self.cfg = cfg
        requested = cfg.get('device', 'cuda')
        self.device = torch.device(requested if requested == 'cpu' or torch.cuda.is_available() else 'cpu')
        self.out_dir = Path(cfg['train']['out_dir']); self.out_dir.mkdir(parents=True, exist_ok=True)
        self.model = ContentAwareHomographyNet(cfg['model']['feature_channels'], cfg['model']['max_corner_offset']).to(self.device)
        self.criterion = ContentAwareTripletLoss(cfg['loss']['lambda_triplet'], cfg['loss']['mu_inverse'])
        self.optim = torch.optim.Adam(self.model.parameters(), lr=cfg['train']['lr'], betas=(cfg['train']['beta1'], cfg['train']['beta2']), eps=cfg['train']['eps'])
        self.sched = torch.optim.lr_scheduler.StepLR(self.optim, step_size=cfg['train']['lr_decay_every'], gamma=cfg['train']['lr_decay_gamma'])
        self.scaler = torch.amp.GradScaler('cuda', enabled=bool(cfg.get('amp', True) and self.device.type == 'cuda'))
        self.step = 0
        if resume:
            ckpt = load_checkpoint(resume, self.model, self.optim, self.sched, self.device)
            self.step = int(ckpt.get('step', 0))

    def train(self):
        dcfg = self.cfg['data']
        dataset = VideoFramePairDataset(
            dcfg['train_video_dir'], dcfg['crop_h'], dcfg['crop_w'], dcfg['temporal_gap_min'], dcfg['temporal_gap_max'],
            dcfg['pairs_per_epoch'], seed=self.cfg.get('seed', 42))
        loader = DataLoader(dataset, batch_size=dcfg['batch_size'], shuffle=True, num_workers=dcfg['num_workers'], pin_memory=self.device.type == 'cuda', drop_last=True)
        pbar = tqdm(total=self.cfg['train']['total_iters'], initial=self.step, dynamic_ncols=True)
        while self.step < self.cfg['train']['total_iters']:
            for batch in loader:
                if self.step >= self.cfg['train']['total_iters']: break
                ia = batch['ia'].to(self.device, non_blocking=True).float()
                ib = batch['ib'].to(self.device, non_blocking=True).float()
                use_attention = self.step >= self.cfg['train']['stage1_iters']
                self.optim.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=self.scaler.is_enabled()):
                    out = self.model(ia, ib, use_attention=use_attention, bidirectional=True)
                    losses = self.criterion(ia, ib, out, self.model.feature)
                    loss = losses['loss']
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optim)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10.0)
                self.scaler.step(self.optim); self.scaler.update(); self.sched.step()
                self.step += 1; pbar.update(1)
                if self.step % self.cfg['train']['log_every'] == 0:
                    pbar.set_postfix(loss=f'{loss.item():.4f}', stage='attn' if use_attention else 'ransac-only', lr=self.sched.get_last_lr()[0])
                if self.step % self.cfg['train']['ckpt_every'] == 0:
                    save_checkpoint(self.out_dir / f'ckpt_{self.step:07d}.pt', self.model, self.optim, self.sched, self.step, self.cfg)
                if self.step % self.cfg['train']['val_every'] == 0:
                    self._maybe_validate()
        save_checkpoint(self.out_dir / 'last.pt', self.model, self.optim, self.sched, self.step, self.cfg)
        pbar.close()

    def _maybe_validate(self):
        dcfg = self.cfg['data']
        try:
            ds = LabeledPointPairsDataset(dcfg['val_npy_dir'], dcfg['val_image_root'])
            metrics = evaluate_labeled_points(self.model, ds, self.device)
            print(f"\nvalidation: {metrics}")
        except Exception as e:
            print(f"\nvalidation skipped: {e}")
