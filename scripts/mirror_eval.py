"""Does the model still work when the sign is made with the other hand?
usage (project root, main venv):  python scripts\mirror_eval.py full_tf_s1
Evaluates the test set as recorded, then mirrored (x flipped, left and right swapped)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.dataset import ISLDataset, MIRROR_PERM, ROOT
from src.models.factory import build_model

name = sys.argv[1] if len(sys.argv) > 1 else "full_tf_s1"
ck = torch.load(os.path.join(ROOT, "models", f"{name}.pt"), map_location="cpu")
args, lm = ck["args"], ck["label_map"]
use_z = not args["no_z"]
ds = ISLDataset(os.path.join(ROOT, "data", "processed", "manifest_split.csv"), "test", lm,
                seq_len=args["seq_len"], use_z=use_z, fill=args.get("fill_gaps", False))
model = build_model(args.get("model", "bilstm"), ck["input_dim"], len(lm), hidden=args["hidden"])
model.load_state_dict(ck["state_dict"])
model.eval()

X = torch.stack([ds[i][0] for i in range(len(ds))]).numpy()
Y = np.array([ds[i][1] for i in range(len(ds))])
d = 3 if use_z else 2
M = X.reshape(len(X), -1, 75, d).copy()
M[..., 0] *= -1
M = M[:, :, MIRROR_PERM].reshape(X.shape)

with torch.no_grad():
    for label, arr in (("as recorded", X), ("mirrored (other hand)", M)):
        pred = model(torch.from_numpy(arr)).argmax(1).numpy()
        print(f"{name} | {label:<22}: top-1 {(pred == Y).mean():.3f} ({(pred == Y).sum()}/{len(Y)})")

# Stopgap used by the API: run the sign as-is and mirrored, keep whichever is more confident.
def probs(a):
    with torch.no_grad():
        return torch.softmax(model(torch.from_numpy(a)), 1).numpy()


pX, pM = probs(X), probs(M)


def tta(p1, p2):
    use = p2.max(1) > p1.max(1)
    return np.where(use[:, None], p2, p1).argmax(1)


for label, p1, p2 in (("as recorded", pX, pM), ("mirrored (other hand)", pM, pX)):
    pred = tta(p1, p2)
    print(f"{name} | with mirror stopgap, {label:<22}: top-1 {(pred == Y).mean():.3f} ({(pred == Y).sum()}/{len(Y)})")


# Auto-handedness: mirror a sign only when the LEFT wrist clearly does most of the moving.
# A left-handed signer then looks right-handed to the model, with no retraining.
K = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5


def mirror(a):
    m = a.reshape(len(a), -1, 75, d).copy()
    m[..., 0] *= -1
    return m[:, :, MIRROR_PERM].reshape(a.shape)


def travel(a):
    res = []
    for w in (15, 16):                                   # left wrist, right wrist
        p = a.reshape(len(a), -1, 75, d)[:, :, w, :2]
        ok = (p != 0).any(-1)
        step = np.linalg.norm(np.diff(p, axis=1), axis=-1) * (ok[:, 1:] & ok[:, :-1])
        res.append(step.sum(1))
    return np.stack(res, 1)


def auto(a):
    t = travel(a)
    flip = t[:, 0] > K * t[:, 1]
    use = np.where(flip[:, None, None], mirror(a), a)
    with torch.no_grad():
        return model(torch.from_numpy(use.astype(np.float32))).argmax(1).numpy(), flip


for label, arr in (("as recorded", X), ("mirrored (other hand)", M)):
    pred, flip = auto(arr)
    print(f"{name} | auto-handedness (K={K}), {label:<22}: top-1 {(pred == Y).mean():.3f} ({(pred == Y).sum()}/{len(Y)}) | mirrored {flip.mean():.0%} of clips")


# Sweep K, and add a hybrid: mirror when the left wrist clearly dominates, keep when the right does,
# and when it is unclear take whichever orientation the model is more confident about.
def hybrid(a, k):
    t = travel(a)
    flip, keep = t[:, 0] > k * t[:, 1], t[:, 1] > k * t[:, 0]
    amb = ~(flip | keep)
    pa, pm = probs(a), probs(mirror(a))
    use_m = np.where(amb, pm.max(1) > pa.max(1), flip)
    return np.where(use_m[:, None], pm, pa).argmax(1), use_m, amb


print()
for k in (1.0, 1.25, 1.5):
    K = k
    for label, arr in (("as recorded", X), ("mirrored", M)):
        p1, f1 = auto(arr)
        p2, f2, amb = hybrid(arr, k)
        print(f"{name} | K={k:<4} {label:<12}: motion rule {(p1 == Y).mean():.3f} (mirrors {f1.mean():.0%}) "
              f"| hybrid {(p2 == Y).mean():.3f} (mirrors {f2.mean():.0%}, unclear {amb.mean():.0%})")