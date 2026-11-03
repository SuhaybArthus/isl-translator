import os
from collections import Counter, defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPLITS = os.path.join(ROOT, "data", "raw", "splits")


def clean(folder):
    head, sep, tail = folder.partition(".")
    return tail.strip() if sep and head.strip().isdigit() else folder.strip()


# official lists: Category/NN. Sign/clip.MOV
exp_n = Counter()
exp_cats = defaultdict(set)
for s in ("train", "val", "test"):
    for line in open(os.path.join(SPLITS, f"include_{s}.txt"), encoding="utf-8"):
        p = line.strip().replace("\\", "/").split("/")
        if len(p) < 3:
            continue
        lab = clean(p[1])
        exp_n[(p[0], lab)] += 1
        exp_cats[lab].add(p[0])

# what we actually have
df = pd.read_csv(os.path.join(ROOT, "data", "processed", "manifest.csv"))
have_n = Counter(zip(df["category"], df["label"]))
have_cats = defaultdict(set)
for (c, l) in have_n:
    have_cats[l].add(c)

print(f"official: {sum(exp_n.values())} clips, {len(exp_n)} (category, sign) pairs, {len(exp_cats)} distinct names")
print(f"have    : {sum(have_n.values())} clips, {len(have_n)} (category, sign) pairs, {len(have_cats)} distinct names")

print("\nNames used by more than one category (these merge into one class):")
for lab, cats in sorted(exp_cats.items()):
    if len(cats) > 1:
        print(f"  {lab!r}: {sorted(cats)}")

print("\nSigns where we have fewer clips than the official lists:")
short = 0
for key, n in sorted(exp_n.items()):
    if have_n.get(key, 0) < n:
        print(f"  {key}: expected {n}, have {have_n.get(key, 0)}")
        short += 1
if not short:
    print("  none")

print("\nSigns we have that the official lists don't:")
extra = [k for k in have_n if k not in exp_n]
print("  " + (", ".join(map(str, extra)) if extra else "none"))