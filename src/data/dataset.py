import json
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# MediaPipe pose left/right pairs; layout is pose(33) + left hand(21) + right hand(21)
_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16),
          (17, 18), (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]
MIRROR_PERM = np.arange(75)
for a, b in _PAIRS:
    MIRROR_PERM[a], MIRROR_PERM[b] = b, a
for i in range(21):
    MIRROR_PERM[33 + i], MIRROR_PERM[54 + i] = 54 + i, 33 + i

LEFT, RIGHT = slice(33, 54), slice(54, 75)


def build_label_map(manifest_path):
    labels = sorted(pd.read_csv(manifest_path)["label"].unique())
    return {l: i for i, l in enumerate(labels)}


def fill_gaps(x, missing, max_gap=5):
    """Linearly interpolate short runs where a whole hand is missing.
    Gaps at the start/end of a clip, or longer than max_gap, are left as zeros."""
    x, missing = x.copy(), missing.copy()
    for s in (LEFT, RIGHT):
        gone = missing[:, s].all(1)
        t = 0
        while t < len(gone):
            if gone[t]:
                e = t
                while e < len(gone) and gone[e]:
                    e += 1
                if t > 0 and e < len(gone) and (e - t) <= max_gap:
                    for k in range(t, e):
                        w = (k - (t - 1)) / (e - (t - 1))
                        x[k, s] = (1 - w) * x[t - 1, s] + w * x[e, s]
                        missing[k, s] = False
                t = e
            else:
                t += 1
    return x, missing


def normalize(arr, fill=False, max_gap=5):
    """(T,225) -> (T,75,3) centered on shoulders, scaled by shoulder width.
    Undetected landmarks stay zero. Returns the normalized clip and the missing mask."""
    x = arr.reshape(len(arr), 75, 3).astype(np.float32)
    missing = (x == 0).all(-1)                          # (T,75)
    if fill:
        x, missing = fill_gaps(x, missing, max_gap)
    center = (x[:, 11] + x[:, 12]) / 2                  # shoulder midpoint
    scale = np.linalg.norm(x[:, 11, :2] - x[:, 12, :2], axis=-1).mean()
    scale = scale if scale > 1e-6 else 1.0
    x = (x - center[:, None]) / scale
    x[missing] = 0
    return x, missing


def temporal_idx(T, L, augment):
    lo, hi = 0, T - 1
    if augment:
        cut = int(0.1 * T)
        lo = np.random.randint(0, cut + 1)
        hi = T - 1 - np.random.randint(0, cut + 1)
    idx = np.round(np.linspace(lo, hi, L)).astype(int)
    return np.clip(idx, 0, T - 1)                       # nearest-frame: no blending with zeros


def augment_seq(x, missing, p_mirror, p_hand_drop):
    missing = missing.copy()
    if np.random.rand() < p_mirror:                     # flip x, swap left/right
        x = x.copy()
        x[..., 0] *= -1
        x, missing = x[:, MIRROR_PERM], missing[:, MIRROR_PERM]
    t = np.deg2rad(np.random.uniform(-10, 10))          # small rotation in the image plane
    R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=np.float32)
    x = x.copy()
    x[..., :2] = x[..., :2] @ R.T
    x *= np.random.uniform(0.9, 1.1)                    # scale
    x[..., :2] += np.random.uniform(-0.05, 0.05, size=2)  # shift
    x += np.random.normal(0, 0.005, x.shape)            # jitter
    if np.random.rand() < p_hand_drop:                  # simulate tracking loss on one hand
        s = LEFT if np.random.rand() < 0.5 else RIGHT
        T = len(x)
        a = np.random.randint(0, max(1, T - 12))
        b = min(T, a + np.random.randint(4, 12))
        missing[a:b, s] = True
    x[missing] = 0                                      # keep "missing" as zeros
    return x


class ISLDataset(Dataset):
    def __init__(self, manifest_path, split, label_map, seq_len=64, use_z=True,
                 augment=False, p_mirror=0.0, p_hand_drop=0.0, fill=False):
        df = pd.read_csv(manifest_path)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.label_map, self.seq_len = label_map, seq_len
        self.use_z, self.augment = use_z, augment
        self.p_mirror, self.p_hand_drop = p_mirror, p_hand_drop
        # cache normalized clips in memory (small: keypoints only)
        self.clips = [normalize(np.load(os.path.join(ROOT, p)), fill=fill)
                      for p in self.df["path"]]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        x, missing = self.clips[i]
        idx = temporal_idx(len(x), self.seq_len, self.augment)
        x, missing = x[idx], missing[idx]
        if self.augment:
            x = augment_seq(x, missing, self.p_mirror, self.p_hand_drop)
        if not self.use_z:
            x = x[..., :2]
        y = self.label_map[self.df.loc[i, "label"]]
        return torch.from_numpy(x.reshape(self.seq_len, -1).astype(np.float32)), y


if __name__ == "__main__":
    mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
    lm = build_label_map(mp)
    os.makedirs(os.path.join(ROOT, "configs"), exist_ok=True)
    json.dump(lm, open(os.path.join(ROOT, "configs", "labels.json"), "w"), indent=2)

    ds = ISLDataset(mp, "train", lm, augment=True, p_mirror=0.5, p_hand_drop=0.5, fill=True)
    x, y = ds[0]
    print(len(ds), "train clips | x:", tuple(x.shape), "| y:", y,
          "| NaNs:", bool(torch.isnan(x).any()))

    # how much does gap filling recover?
    df = pd.read_csv(mp)
    for name, fill in [("no fill", False), ("fill   ", True)]:
        rates = []
        for p in df["path"]:
            _, m = normalize(np.load(os.path.join(ROOT, p)), fill=fill)
            rates.append([m[:, LEFT].all(1).mean(), m[:, RIGHT].all(1).mean()])
        l, r = np.mean(rates, axis=0)
        print(f"{name}: frames with left hand missing {l:.3f} | right hand missing {r:.3f}")