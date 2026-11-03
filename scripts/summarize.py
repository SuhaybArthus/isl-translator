import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(ROOT, "outputs")

groups = defaultdict(list)
for f in sorted(glob.glob(os.path.join(OUT_DIR, "*_results.json"))):
    stem = os.path.basename(f).replace("_results.json", "")
    m = re.match(r"^(.*)_s(\d+)$", stem)
    if not m:                      # skip unseeded runs such as the old "bilstm"
        print(f"Skipping {os.path.basename(f)} (no _s<seed> suffix)")
        continue
    groups[m.group(1)].append(json.load(open(f)))


def fmt(vals):
    return f"{np.mean(vals):.3f} ± {np.std(vals):.3f}"


rows = []
header = f"{'config':<16}{'runs':>5}{'test top1':>16}{'test top5':>16}{'test macro-F1':>16}{'val top1':>16}"
print("\n" + header)
print("-" * len(header))
for name, runs in sorted(groups.items()):
    t = [r["test"] for r in runs]
    v = [r["val"]["top1"] for r in runs]
    line = (f"{name:<16}{len(runs):>5}{fmt([x['top1'] for x in t]):>16}"
            f"{fmt([x['top5'] for x in t]):>16}{fmt([x['macro_f1'] for x in t]):>16}{fmt(v):>16}")
    print(line)
    rows.append(line)

# markdown table for the README
md = ["| Config | Runs | Test top-1 | Test top-5 | Test macro-F1 |", "|---|---|---|---|---|"]
for name, runs in sorted(groups.items()):
    t = [r["test"] for r in runs]
    md.append(f"| {name} | {len(runs)} | {fmt([x['top1'] for x in t])} | "
              f"{fmt([x['top5'] for x in t])} | {fmt([x['macro_f1'] for x in t])} |")
out = os.path.join(OUT_DIR, "ablation_summary.md")
with open(out, "w", encoding="utf-8") as f:
    f.write("\n".join(md) + "\n")
print(f"\nMarkdown table saved to {out}")