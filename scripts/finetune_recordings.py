"""Fine-tune an existing model on your own recorded takes, mixed with INCLUDE.

Your raw takes in data/recordings are only ever READ. Nothing is moved, changed or deleted, and the INCLUDE
manifest is not touched. A separate manifest is rebuilt from whatever is on disk on every run, so deleting a take,
or a whole word folder, removes it from the next run. The base checkpoint is never overwritten: the result is
saved under a new name.

usage (project root, main venv):
  python scripts\\finetune_recordings.py --base full_tf_s1 --name full_tf_ft1
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from src.data.dataset import ISLDataset
from src.models.factory import build_model
from src.training.train import evaluate

REC = os.path.join(ROOT, "data", "recordings")
STAMP = re.compile(r"_(\d{8}-\d{6})-\d{3}_([RLB])\.npy$")
SESSION_GAP = 30 * 60                      # takes more than 30 minutes apart count as different sessions


def scan(label_map, min_takes):
    """One row per usable take. Held-out takes: the latest session if there are several, else the last 25%."""
    rows, skipped = [], {}
    for d in sorted(glob.glob(os.path.join(REC, "*"))):
        if not os.path.isdir(d):
            continue
        word = os.path.basename(d)
        files = sorted(glob.glob(os.path.join(d, "*.npy")))
        if not files:
            continue
        if word not in label_map:
            skipped[word] = "not one of the model's words"
            continue
        if len(files) < min_takes:
            skipped[word] = f"only {len(files)} take(s), needs {min_takes}"
            continue
        takes = []
        for f in files:
            m = STAMP.search(f)
            t = time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S")) if m else os.path.getmtime(f)
            takes.append((t, f, m.group(2) if m else "?"))
        takes.sort()
        sessions, cur = [], [takes[0]]
        for a, b in zip(takes, takes[1:]):
            if b[0] - a[0] > SESSION_GAP:
                sessions.append(cur)
                cur = []
            cur.append(b)
        sessions.append(cur)
        n = len(takes)
        if len(sessions) > 1 and len(sessions[-1]) <= n // 2:
            test = {f for _, f, _ in sessions[-1]}
        elif n >= 4:
            test = {f for _, f, _ in takes[-max(1, round(n * 0.25)):]}
        else:
            test = set()
        for t, f, hand in takes:
            rows.append({"path": os.path.relpath(f, ROOT), "label": word, "video": os.path.basename(f),
                         "split": "rec_test" if f in test else "train", "source": "rec", "hand": hand})
    return pd.DataFrame(rows), skipped


def load_expanded(model, old_state, n_old):
    """Load old weights into a model with extra output classes: old classes keep their rows, new rows stay fresh."""
    new = model.state_dict()
    for k, v in old_state.items():
        if new[k].shape == v.shape:
            new[k] = v
        elif v.shape[0] == n_old:
            new[k][:n_old] = v
    model.load_state_dict(new)


def loader(ds, **kw):
    return DataLoader(ds, batch_size=32, num_workers=0, **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True, help="checkpoint in models/ to start from, e.g. full_tf_s1")
    ap.add_argument("--name", required=True, help="name for the new checkpoint (the base is left untouched)")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--rec_weight", type=float, default=5.0, help="how much more often a recorded take is sampled than an INCLUDE clip")
    ap.add_argument("--min_takes", type=int, default=3, help="skip words with fewer takes than this")
    ap.add_argument("--add_words", default="", help='new classes to add, comma separated, e.g. "No sign"')
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(os.path.join(ROOT, "models", f"{a.base}.pt"), map_location="cpu")
    args0, lm = ck["args"], ck["label_map"]
    n_old = len(lm)
    new = [w.strip() for w in a.add_words.split(",") if w.strip() and w.strip() not in lm]
    lm = dict(lm)
    for w in new:
        lm[w] = len(lm)

    rec, skipped = scan(lm, a.min_takes)
    if rec.empty:
        sys.exit(f"No usable takes in {REC}. Skipped: {skipped or 'nothing recorded yet'}")
    inc = pd.read_csv(os.path.join(ROOT, "data", "processed", "manifest_split.csv"))
    inc = inc[inc["label"].isin(lm)].assign(source="include")
    mp = os.path.join(ROOT, "outputs", f"{a.name}_finetune_manifest.csv")
    pd.concat([inc, rec], ignore_index=True).to_csv(mp, index=False)

    n_train, n_test = (rec["split"] == "train").sum(), (rec["split"] == "rec_test").sum()
    print(f"Recorded takes: {n_train} for training, {n_test} held out, across {rec['label'].nunique()} words.")
    for w, why in skipped.items():
        print(f"  skipped {w}: {why}")
    if n_test == 0:
        print("No held-out takes yet, so the change on your own signing cannot be measured.")

    common = dict(label_map=lm, seq_len=args0["seq_len"], use_z=not args0["no_z"], fill=args0.get("fill_gaps", False))
    train_ds = ISLDataset(mp, "train", augment=True, p_mirror=args0.get("p_mirror", 0.0),
                          p_hand_drop=args0.get("p_hand_drop", 0.0), **common)
    sets = {"INCLUDE val": ISLDataset(mp, "val", **common), "INCLUDE test": ISLDataset(mp, "test", **common)}
    if n_test:
        sets["your held-out takes"] = ISLDataset(mp, "rec_test", **common)

    model = build_model(args0.get("model", "bilstm"), ck["input_dim"], len(lm), hidden=args0["hidden"]).to(device)
    load_expanded(model, ck["state_dict"], n_old)
    before = None if new else {k: evaluate(model, loader(ds), device) for k, ds in sets.items()}   # the base has no new classes to score

    w = np.where(train_ds.df["source"].values == "rec", a.rec_weight, 1.0)
    sampler = WeightedRandomSampler(w, num_samples=len(train_ds), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=a.batch, sampler=sampler, num_workers=0)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    for ep in range(1, a.epochs + 1):
        model.train()
        total = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} | loss {total / len(train_ds):.3f}")

    after = {k: evaluate(model, loader(ds), device) for k, ds in sets.items()}   # last epoch, not best-on-test
    print(f"\n{'':<22}{'before':>16}{'after':>16}   (top-1, macro-F1)")
    for k in sets:
        f = after[k]
        b = f"{before[k]['top1']:>8.3f} {before[k]['macro_f1']:>6.3f}" if before else f"{'n/a':>15}"
        print(f"{k:<22}{b}{f['top1']:>10.3f} {f['macro_f1']:>6.3f}   n={len(sets[k])}")

    def predicted(ds):
        with torch.no_grad():
            return torch.cat([model(x.to(device)).argmax(1).cpu() for x, _ in loader(ds)]).numpy()
    for w in new:
        p_inc = predicted(sets["INCLUDE test"])
        print(f"\n'{w}': INCLUDE test clips wrongly called '{w}': {(p_inc == lm[w]).mean():.1%}")
        if "your held-out takes" in sets:
            ds = sets["your held-out takes"]
            mask = ds.df["label"].values == w
            if mask.any():
                print(f"'{w}': your held-out '{w}' takes called '{w}': {(predicted(ds)[mask] == lm[w]).mean():.0%} (n={mask.sum()})")

    out = {"base": a.base, "name": a.name, "recorded_train": int(n_train), "recorded_test": int(n_test),
           "skipped": skipped, "before": before, "after": after, "args": vars(a)}
    torch.save({"state_dict": model.state_dict(), "args": {**args0, "name": a.name, "finetuned_from": a.base},
                "label_map": lm, "input_dim": ck["input_dim"]}, os.path.join(ROOT, "models", f"{a.name}.pt"))
    json.dump(out, open(os.path.join(ROOT, "outputs", f"{a.name}_finetune.json"), "w"), indent=2)
    print(f"\nSaved models\\{a.name}.pt. The base model models\\{a.base}.pt is unchanged.")


if __name__ == "__main__":
    main()