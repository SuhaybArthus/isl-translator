"""Left-handed signers and still hands: numpy-only helpers shared by the API and the tests."""
import numpy as np

# left/right swap for mirroring a sign: pose pairs, then left hand <-> right hand
PERM = np.arange(75)
for _a, _b in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    PERM[_a], PERM[_b] = _b, _a
for _i in range(21):
    PERM[33 + _i], PERM[54 + _i] = 54 + _i, 33 + _i


def mirrored(x, use_z):
    """Mirror a preprocessed clip (T, 75*d): flip x and swap left and right."""
    d = 3 if use_z else 2
    m = x.reshape(len(x), 75, d).copy()
    m[..., 0] *= -1
    return m[:, PERM].reshape(x.shape)


def hand_share(seg):
    """Share of frames in a raw (T, 225) segment where at least one hand is tracked."""
    return float((seg[:, 99:225] != 0).any(1).mean()) if len(seg) else 0.0


def wrist_travel(x, use_z):
    """Wrist travel (left, right) in a preprocessed, shoulder-normalized clip."""
    a = x.reshape(len(x), 75, 3 if use_z else 2)
    out = []
    for w in (15, 16):
        p = a[:, w, :2]
        ok = (p != 0).any(-1)
        out.append(float((np.linalg.norm(np.diff(p, axis=0), axis=-1) * (ok[1:] & ok[:-1])).sum()))
    return out


def wrist_motion(seg):
    """Total wrist travel in a raw (T, 225) segment, in shoulder widths (ignores jitter under 0.004)."""
    sw = np.median(np.linalg.norm(seg[:, 33:35] - seg[:, 36:38], axis=1))
    if sw < 1e-6:
        return 0.0
    total = 0.0
    for c in (45, 48):
        ok = (seg[:, c:c + 2] != 0).any(1)
        d = np.linalg.norm(np.diff(seg[:, c:c + 2], axis=0), axis=1)[ok[1:] & ok[:-1]]
        total += d[d > 0.004].sum()
    return float(total / sw)