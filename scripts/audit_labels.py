"""Lists clips whose label a trained model confidently disagrees with. These are CANDIDATES for wrong labels,
and a person has to watch each one before changing anything.

Limits: clips the model trained on can be memorized, so mistakes hide best in the 'train' split. Words that
are genuinely similar (Red and Pink) also show up, so look at the video, not just the list.
usage (project root, main venv):  python scripts\\audit_labels.py abl_drop_s1
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ISLDataset
from src.models.factory import build_model

name = sys.argv[1] if len(sys.argv) > 1 else "abl_drop_s1"
ck = torch.load(os.path.join(ROOT, "models", f"{name}.pt"), map_location="cpu")
args, lm = ck["args"], ck["label_map"]
names = {v: k for k, v in lm.items()}
model = build_model(args.get("model", "bilstm"), ck["input_dim"], len(lm), hidden=args["hidden"])
model.load_state_dict(ck["state_dict"])
model.eval()

mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
rows = []
for split in ("train", "val", "test"):
    ds = ISLDataset(mp, split, lm, seq_len=args["seq_len"], use_z=not args["no_z"], fill=args.get("fill_gaps", False))
    with torch.no_grad():
        probs = torch.cat([torch.softmax(model(x), 1) for x, _ in DataLoader(ds, batch_size=64)]).numpy()
    for i, p in enumerate(probs):
        lab = lm[ds.df.loc[i, "label"]]
        top = int(p.argmax())
        if top != lab:
            rows.append({"split": split, "category": ds.df.loc[i, "category"], "label": names[lab], "model_says": names[top],
                         "p_model": round(float(p[top]), 2), "p_label": round(float(p[lab]), 2), "video": ds.df.loc[i, "video"]})

out = pd.DataFrame(rows)
sus = out[(out.p_model >= 0.6) & (out.p_label <= 0.15)].sort_values("p_model", ascending=False)
os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
sus.to_csv(os.path.join(ROOT, "outputs", "label_audit.csv"), index=False)
print(f"{len(sus)} clips where the model is at least 60% sure of another word and gives the label 15% or less")
print(sus.head(60).to_string(index=False))
print("\nFull list saved to outputs\\label_audit.csv. Watch the videos before changing any label.")