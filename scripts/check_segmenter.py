import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.data.dataset import ROOT
from src.inference.segment import SignSegmenter

# (rest_line, pre_roll, post_pad)
CONFIGS = [(1.2, 4, 0), (1.6, 4, 0), (1.6, 12, 8), (2.0, 12, 8), (1.6, 20, 12)]
NOISE_FLOOR = 0.004      # ignore per-frame wrist displacement below this (image units)
L_WR, R_WR = 15, 16


def motion(frames, floor=0.0):
    """Total frame-to-frame wrist displacement, ignoring undetected frames and tiny wobble."""
    total = 0.0
    for w in (L_WR, R_WR):
        p = frames[:, w * 3:w * 3 + 2]
        ok = (frames[:, w * 3:w * 3 + 3] != 0).any(1)
        valid = ok[1:] & ok[:-1]
        d = np.linalg.norm(np.diff(p, axis=0), axis=1)[valid]
        total += d[d > floor].sum()
    return total


def segment(arr, rest_line, pre_roll, post_pad):
    seg = SignSegmenter(rest_line=rest_line, pre_roll=pre_roll, post_pad=post_pad)
    segs = []
    for f in arr:
        s = seg.push(f)
        if s is not None:
            segs.append(s)
    s = seg.flush()
    if s is not None:
        segs.append(s)
    return segs


df = pd.read_csv(os.path.join(ROOT, "data", "processed", "manifest_split.csv"))
df = df.iloc[::4]                                      # every 4th clip is plenty
clips = [np.load(os.path.join(ROOT, p)) for p in df["path"]]
print(f"clips checked: {len(clips)}\n")

for rl, pre, post in CONFIGS:
    counts, cover, raw_m, floor_m, bad = [], [], [], [], []
    for (_, r), arr in zip(df.iterrows(), clips):
        segs = segment(arr, rl, pre, post)
        counts.append(len(segs))
        cover.append(sum(len(s) for s in segs) / len(arr))
        t = motion(arr)
        tf = motion(arr, NOISE_FLOOR)
        if t > 0:
            raw_m.append(sum(motion(s) for s in segs) / t)
        if tf > 0:
            floor_m.append(sum(motion(s, NOISE_FLOOR) for s in segs) / tf)
        if len(segs) != 1:
            bad.append((r["video"], r["label"], len(segs)))
    counts, raw_m, floor_m = np.array(counts), np.array(raw_m), np.array(floor_m)
    print(f"rest_line {rl} | pre_roll {pre} | post_pad {post}")
    print(f"   1 segment {(counts == 1).mean():.1%} | 0 segments {(counts == 0).mean():.1%} | "
          f"2+ {(counts > 1).mean():.1%} | frames kept {np.median(cover):.2f}")
    print(f"   motion captured: raw {np.median(raw_m):.2f} | noise-filtered {np.median(floor_m):.2f} "
          f"(clips under 90%: {(floor_m < 0.9).mean():.1%})")
    if bad:
        print("   problem clips:", bad[:6])