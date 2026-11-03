"""Shared preprocessing for training and serving.

Input : raw keypoints for ONE completed sign segment, shape (T, 225)
        layout = pose(33) + left hand(21) + right hand(21), each (x, y, z),
        all zeros for a landmark that was not detected.
Output: float32 array (seq_len, 225) or (seq_len, 150) without z.
"""
import numpy as np

N_LM = 75
LEFT, RIGHT = slice(33, 54), slice(54, 75)
L_SHOULDER, R_SHOULDER = 11, 12


def fill_gaps(x, missing, max_gap=5):
    """Linearly interpolate short runs where a whole hand is missing.
    Gaps at the start/end of a clip, or longer than max_gap, stay zero."""
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
    """(T,225) -> (T,75,3), centered on the shoulder midpoint and scaled by
    shoulder width. Undetected landmarks stay zero. Returns (x, missing)."""
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != N_LM * 3:
        raise ValueError(f"expected (T, {N_LM * 3}), got {arr.shape}")
    if len(arr) < 2:
        raise ValueError("need at least 2 frames")
    x = arr.reshape(len(arr), N_LM, 3)
    missing = (x == 0).all(-1)
    if fill:
        x, missing = fill_gaps(x, missing, max_gap)
    center = (x[:, L_SHOULDER] + x[:, R_SHOULDER]) / 2
    scale = np.linalg.norm(x[:, L_SHOULDER, :2] - x[:, R_SHOULDER, :2], axis=-1).mean()
    scale = scale if scale > 1e-6 else 1.0
    x = (x - center[:, None]) / scale
    x[missing] = 0
    return x, missing


def resample(x, seq_len=64):
    """Nearest-frame resampling to a fixed length (no blending with zeros)."""
    idx = np.round(np.linspace(0, len(x) - 1, seq_len)).astype(int)
    return x[np.clip(idx, 0, len(x) - 1)]


def preprocess_clip(arr, seq_len=64, use_z=True, fill=True, max_gap=5):
    """Full pipeline: raw (T,225) -> model input (seq_len, F), float32."""
    x, _ = normalize(arr, fill=fill, max_gap=max_gap)
    x = resample(x, seq_len)
    if not use_z:
        x = x[..., :2]
    return np.ascontiguousarray(x.reshape(seq_len, -1), dtype=np.float32)