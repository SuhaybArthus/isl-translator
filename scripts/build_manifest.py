import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROC = os.path.join(ROOT, "data", "processed")


def get_label(npy_path):
    d = os.path.dirname(npy_path)
    while os.path.abspath(d) != os.path.abspath(PROC):
        head, sep, tail = os.path.basename(d).partition(".")
        if sep and head.strip().isdigit():
            return tail.strip()
        d = os.path.dirname(d)
    return os.path.basename(os.path.dirname(npy_path))


rows = []
for f in sorted(glob.glob(os.path.join(PROC, "**", "*.npy"), recursive=True)):
    parts = os.path.relpath(f, PROC).split(os.sep)    # Zip_NofM / Category / NN. Sign / ... / clip.npy
    arr = np.load(f)
    rows.append({
        "path": os.path.relpath(f, ROOT),
        "category": parts[1] if len(parts) >= 4 else parts[0],
        "label": get_label(f),
        "video": os.path.basename(f),
        "frames": len(arr),
        "left_hand_rate": float((arr[:, 99:162] != 0).any(axis=1).mean()),
        "right_hand_rate": float((arr[:, 162:225] != 0).any(axis=1).mean()),
    })

df = pd.DataFrame(rows)
df.to_csv(os.path.join(PROC, "manifest.csv"), index=False)
print(len(df), "clips |", df.label.nunique(), "classes |", df.category.nunique(), "categories")