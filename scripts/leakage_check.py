import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import ISLDataset, ROOT
from src.models.factory import build_model

NAME = sys.argv[1] if len(sys.argv) > 1 else "full_bilstm_s1"
WINDOW = 3

ck = torch.load(os.path.join(ROOT, "models", f"{NAME}.pt"), map_location="cpu")
args, lm = ck["args"], ck["label_map"]
mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")

ds = ISLDataset(mp, "test", lm, seq_len=args["seq_len"], use_z=not args["no_z"],
                fill=args.get("fill_gaps", False))
model = build_model(args.get("model", "bilstm"), ck["input_dim"], len(lm), hidden=args["hidden"])
model.load_state_dict(ck["state_dict"])
model.eval()
with torch.no_grad():
    pred = torch.cat([model(x).argmax(1) for x, _ in DataLoader(ds, batch_size=64)]).numpy()

num = lambda v: int(re.search(r"(\d+)", v).group(1))
train = pd.read_csv(mp).query("split == 'train'")
train_nums = train.groupby("label")["video"].apply(lambda s: [num(v) for v in s]).to_dict()

df = ds.df.copy()
df["ok"] = pred == df["label"].map(lm).values
df["near_train_take"] = [any(abs(num(v) - n) <= WINDOW for n in train_nums.get(l, []))
                         for v, l in zip(df["video"], df["label"])]

print(f"overall: {df.ok.mean():.3f} ({df.ok.sum()}/{len(df)})")
for flag, g in df.groupby("near_train_take"):
    tag = "has a near-identical train take" if flag else "no near-identical train take (session held out)"
    print(f"{tag}: {len(g)} clips | accuracy {g.ok.mean():.3f}")
df.to_csv(os.path.join(ROOT, "outputs", f"{NAME}_test_preds.csv"), index=False)