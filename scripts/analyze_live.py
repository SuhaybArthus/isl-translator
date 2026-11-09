import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np
import onnxruntime as ort
from src.inference.preprocess import preprocess_clip
from src.inference.segment import SignSegmenter, hands_raised

MODEL = os.environ.get("ISL_MODEL", "full_tf_s1")
meta = json.load(open(os.path.join(ROOT, "models", f"{MODEL}.meta.json"), encoding="utf-8"))
sess = ort.InferenceSession(os.path.join(ROOT, "models", f"{MODEL}.onnx"), providers=["CPUExecutionProvider"])
a = np.fromfile(sys.argv[1], dtype=np.float32).reshape(-1, 225)

pose_ok = ((a[:, 33:36] != 0).any(1) & (a[:, 36:39] != 0).any(1)).mean()
raised = np.array([hands_raised(f, 1.6) for f in a])
rel = []                                   # height of the higher wrist, in shoulder widths below the shoulders
for f in a:
    ls, rs = f[33:36], f[36:39]
    if not (ls.any() and rs.any()):
        continue
    sw = np.linalg.norm(ls[:2] - rs[:2])
    if sw < 1e-6:
        continue
    mid = (ls[1] + rs[1]) / 2
    rel.append(min((f[45 + 1] - mid) / sw, (f[48 + 1] - mid) / sw))
print(f"frames {len(a)} | shoulders detected {pose_ok:.0%} | hands counted as raised {raised.mean():.0%}")
if rel:
    print("higher wrist, shoulder widths below shoulders (5th/50th/95th pct):",
          [round(float(np.percentile(rel, q)), 2) for q in (5, 50, 95)], "| rest line = 1.6")

seg, out = SignSegmenter(), []
for f in a:
    s = seg.push(f)
    if s is not None:
        out.append(s)
s = seg.flush()
if s is not None:
    out.append(s)
print(f"segments found: {len(out)}, lengths {[len(s) for s in out]}")
for i, s in enumerate(out, 1):
    x = preprocess_clip(s, meta["seq_len"], use_z=meta["use_z"], fill=meta["fill_gaps"])
    lg = sess.run(None, {"x": x[None]})[0][0]
    p = np.exp(lg - lg.max()); p /= p.sum()
    print(f"  segment {i}:", [(meta["labels"][j], round(float(p[j]), 2)) for j in np.argsort(p)[::-1][:3]])