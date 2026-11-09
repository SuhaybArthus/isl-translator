"""Compare landmarks recorded in the browser with the offline keypoints of the same video.

usage (project root, second venv):
  .venv-onnx\\Scripts\\python scripts\\compare_browser.py <browser.f32> <offline.npy>

It reports: frame counts, hand-detection rates, how the segmenter splits each version,
what the model predicts for each, how far apart the landmarks are, and whether left/right look swapped.
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnxruntime as ort

from src.inference.preprocess import normalize, preprocess_clip
from src.inference.segment import SignSegmenter

MODEL = os.environ.get("ISL_MODEL", "full_tf_s1")
meta = json.load(open(os.path.join(ROOT, "models", f"{MODEL}.meta.json"), encoding="utf-8"))
sess = ort.InferenceSession(os.path.join(ROOT, "models", f"{MODEL}.onnx"),
                            providers=["CPUExecutionProvider"])
LABELS = meta["labels"]


def top3(clip):
    x = preprocess_clip(clip, meta["seq_len"], use_z=meta["use_z"], fill=meta["fill_gaps"])
    logits = sess.run(None, {"x": x[None]})[0][0]
    p = np.exp(logits - logits.max())
    p /= p.sum()
    return [(LABELS[i], round(float(p[i]), 2)) for i in np.argsort(p)[::-1][:3]]


def segments(frames):
    seg, out = SignSegmenter(), []
    for f in frames:
        s = seg.push(f)
        if s is not None:
            out.append(s)
    s = seg.flush()
    if s is not None:
        out.append(s)
    return out


def hand_rates(a):
    return (float((a[:, 99:162] != 0).any(1).mean()), float((a[:, 162:225] != 0).any(1).mean()))


def strip_trailing_blank(a):
    keep = (a != 0).any(1)
    return a[: np.where(keep)[0][-1] + 1] if keep.any() else a


browser = strip_trailing_blank(np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 225))
offline = np.load(sys.argv[2]).astype(np.float32)

print(f"frames        browser {len(browser)} | offline {len(offline)}")
bl, br = hand_rates(browser)
ol, orr = hand_rates(offline)
print(f"hand detected browser left {bl:.2f} right {br:.2f} | offline left {ol:.2f} right {orr:.2f}")

for name, a in (("browser", browser), ("offline", offline)):
    segs = segments(a)
    print(f"segments      {name}: {len(segs)} -> lengths {[len(s) for s in segs]}")
    for i, s in enumerate(segs):
        print(f"   segment {i + 1} prediction: {top3(s)}")
    print(f"   whole clip prediction: {top3(a)}")

# landmark distance, in shoulder widths, over landmarks detected in both
n = min(len(browser), len(offline))
xb, mb = normalize(browser[:n])
xo, mo = normalize(offline[:n])


def dist(xa, ma, xb_, mb_, sl_a, sl_b):
    both = ~ma[:, sl_a] & ~mb_[:, sl_b]
    if both.sum() == 0:
        return float("nan")
    d = np.abs(xa[:, sl_a] - xb_[:, sl_b]).mean(-1)
    return float(d[both].mean())


P, L, R = slice(0, 33), slice(33, 54), slice(54, 75)
print("\nmean abs landmark difference (shoulder widths; smaller is closer)")
print(f"   pose {dist(xb, mb, xo, mo, P, P):.3f} | left hand {dist(xb, mb, xo, mo, L, L):.3f} "
      f"| right hand {dist(xb, mb, xo, mo, R, R):.3f}")
same = dist(xb, mb, xo, mo, L, L)
swapped = dist(xb, mb, xo, mo, L, R)
print(f"   left-vs-left {same:.3f} | browser-left vs offline-right {swapped:.3f}")
if np.isfinite(same) and np.isfinite(swapped) and swapped < same * 0.7:
    print("   -> LEFT AND RIGHT LOOK SWAPPED in the browser landmarks.")