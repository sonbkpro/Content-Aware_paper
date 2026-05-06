# Content-Aware Unsupervised Deep Homography Estimation — PyTorch Implementation

This repository implements the ECCV 2020 paper **Content-Aware Unsupervised Deep Homography Estimation** for small-baseline image/video pairs.

The implementation follows the paper method while adding practical engineering features:

- Pure PyTorch DLT homography solver.
- Pure PyTorch STN-style perspective warping with `grid_sample`.
- Video dataloader for `data/train/000001.mp4`, `data/train/000002.mp4`, ...
- Random frame-pair sampling: `frame_t` and `frame_t+k`, where `k ∈ [1, 5]` by default.
- Training crop size: `315 x 560`, matching the paper.
- Five supported semantic categories: regular, low-texture, low-light, small-foreground, large-foreground.
- Paper-style two-stage training:
  - Stage 1: mask used only in the normalized feature loss, not as attention.
  - Stage 2: mask used both as attention and as RANSAC-like loss weighting.
- Bidirectional homography prediction: `Hab` and `Hba`.
- Paper-style triplet loss with inverse consistency.
- Validation using manually labeled point correspondences stored in `.npy` files.
- Inference and visualization scripts.
- Smoke tests and unit tests.

---

## 1. Repository layout

```text
content_aware_deep_homography/
├── configs/
│   └── train_default.yaml
├── scripts/
│   ├── train.py
│   ├── infer_pair.py
│   ├── infer_video.py
│   ├── eval_labeled_points.py
│   ├── visualize_alignment.py
│   └── smoke_test.py
├── src/
│   ├── data/
│   ├── engine/
│   ├── geometry/
│   ├── losses/
│   ├── models/
│   └── utils/
└── tests/
```

---

## 2. Install

```bash
cd content_aware_deep_homography
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The implementation does **not** require Kornia. DLT and warping are implemented directly in PyTorch.

---

## 3. Data format

### Training videos

Place videos as:

```text
data/train/000001.mp4
data/train/000002.mp4
data/train/000003.mp4
...
```

The training dataloader samples a random video, then samples:

```text
Ia = frame_t
Ib = frame_t+k, where k ∈ [1, 5]
```

The pair is converted to grayscale and randomly cropped to:

```text
315 x 560
```

### Optional category organization

If you want to keep paper-style categories, you can organize externally as:

```text
data/train/regular/*.mp4
data/train/low-texture/*.mp4
data/train/low-light/*.mp4
data/train/small-foreground/*.mp4
data/train/large-foreground/*.mp4
```

The default loader currently expects `.mp4` directly under `data/train`. If you want recursive category loading, set videos under `data/train` or modify the glob line in `VideoFramePairDataset` from `*.mp4` to `**/*.mp4`.

---

## 4. Validation label format

The validation loader supports your `.npy` format:

```python
{
    'path1': '0000011_10001.jpg',
    'path2': '0000011_10005.jpg',
    'matche_pts': [
        [(349, 236), (357, 236)],
        [(397, 189), (401, 183)],
        ...
    ]
}
```

Put files as:

```text
data/val_labels/*.npy
data/val_images/0000011_10001.jpg
data/val_images/0000011_10005.jpg
```

Evaluate:

```bash
python scripts/eval_labeled_points.py \
  --ckpt runs/content_aware_homography/last.pt \
  --npy_dir data/val_labels \
  --image_root data/val_images
```

Metric:

```text
average L2 pixel error between predicted warped points and human-labeled target points
```

---

## 5. Train

Edit `configs/train_default.yaml` if needed, then run:

```bash
python scripts/train.py --config configs/train_default.yaml
```

Single-GPU safe defaults are used:

```yaml
batch_size: 4
num_workers: 4
amp: true
```

The paper used:

```text
120k iterations
Adam lr = 1e-4
batch size = 64 on 4 RTX 2080 Ti GPUs
lr decay by 0.8 every 12k iterations
lambda = 2.0
mu = 0.01
```

This repository keeps those algorithmic choices, but lowers batch size for one GPU.

---

## 6. Inference on two images

```bash
python scripts/infer_pair.py \
  --ckpt runs/content_aware_homography/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out alignment_overlay.png
```

The script prints `Hab` and saves an overlay where target/warped mismatch appears as color ghosts.

---

## 7. Visualize masks and alignment

```bash
python scripts/visualize_alignment.py \
  --ckpt runs/content_aware_homography/last.pt \
  --image_a path/to/a.jpg \
  --image_b path/to/b.jpg \
  --out_prefix vis/example
```

Outputs:

```text
vis/example_overlay.png
vis/example_mask_a.png
vis/example_mask_b.png
vis/example_warped_a.png
```

---

## 8. Smoke test

```bash
python scripts/smoke_test.py
pytest -q
```

Expected:

```text
SMOKE TEST PASSED
4 passed
```

---

## 9. Important implementation notes

1. **Feature extractor** follows Table 1(a): `Conv 1→4→8→1`, kernel 3, stride 1.
2. **Mask predictor** follows Table 1(b): `Conv 1→4→8→16→32→1`, kernel 3, stride 1, sigmoid output.
3. **Homography estimator** follows a ResNet-34-style backbone and predicts 8 corner offsets.
4. **Homography conversion** uses differentiable DLT from four canonical corners.
5. **Warping** uses inverse perspective sampling with `torch.nn.functional.grid_sample`.
6. **Loss** follows Eq. 4, Eq. 5, and Eq. 6:
   - normalized masked feature alignment loss,
   - feature discriminative term to avoid all-zero features,
   - inverse consistency `||Hab Hba - I||²`.
7. **Two-stage training** follows the paper:
   - first stage disables mask-as-attention,
   - second stage enables both mask roles.

---

## 10. Practical advice

For your videos, start with:

```yaml
batch_size: 2 or 4
pairs_per_epoch: 2000
val_every: 1000
ckpt_every: 2000
```

After confirming stable training and visualizations, increase iterations toward the paper-scale schedule.

This method is designed for **small-baseline** video pairs. It is not expected to work well for large-baseline panorama-style stitching.
