import sys, tempfile
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import cv2, numpy as np
from src.data.video_pair_dataset import VideoFramePairDataset


def test_video_dataset():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td); (d/'train').mkdir()
        wr = cv2.VideoWriter(str(d/'train/000001.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), 5, (64,48))
        for i in range(8): wr.write(np.full((48,64,3), i*20, np.uint8))
        wr.release()
        ds = VideoFramePairDataset(str(d/'train'), crop_h=32, crop_w=32, gap_min=1, gap_max=3, pairs_per_epoch=2)
        s = ds[0]
        assert s['ia'].shape == (1,32,32)
        assert 1 <= s['gap'] <= 3
