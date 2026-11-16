"""Applies documented label corrections to data/processed/manifest_split.csv. Run it after make_splits.py.

configs/label_fixes.csv has the columns: category, wrong_label, video (file name without the extension), correct_label.
The raw data and the official split lists are never changed, and running it twice is safe.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MP = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
FX = os.path.join(ROOT, "configs", "label_fixes.csv")

df, fx = pd.read_csv(MP), pd.read_csv(FX)
stem = df["video"].str.replace(".npy", "", regex=False)
changed = 0
for _, r in fx.iterrows():
    wrong = (df["category"] == r["category"]) & (df["label"] == r["wrong_label"]) & (stem == r["video"])
    fixed = (df["category"] == r["category"]) & (df["label"] == r["correct_label"]) & (stem == r["video"])
    if r["correct_label"] not in set(df["label"]):
        print(f"WARNING: '{r['correct_label']}' is not an existing label, so {r['video']} was left alone")
        continue
    if wrong.any():
        print(f"{r['video']}: {r['wrong_label']} -> {r['correct_label']}  (split: {', '.join(df.loc[wrong, 'split'])})")
        df.loc[wrong, "label"] = r["correct_label"]
        changed += int(wrong.sum())
    else:
        print(("already fixed: " if fixed.any() else "NO MATCH, check the category and file name: ") + f"{r['category']} {r['wrong_label']} {r['video']}")
if changed:
    df.to_csv(MP, index=False)
print(f"{changed} label(s) changed in {MP}")