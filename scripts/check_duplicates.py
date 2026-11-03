import hashlib
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
df = pd.read_csv(os.path.join(ROOT, "data", "processed", "manifest_split.csv"))

seen = {}
for _, r in df.iterrows():
    h = hashlib.md5(np.load(os.path.join(ROOT, r["path"])).tobytes()).hexdigest()
    seen.setdefault(h, []).append((r["video"], r["label"], r["split"]))

dups = [v for v in seen.values() if len(v) > 1]
print(len(dups), "duplicate groups")
for d in dups:
    print(d)