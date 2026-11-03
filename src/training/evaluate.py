import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.data.dataset import ISLDataset, ROOT
from src.models.factory import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bilstm")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()

    ck = torch.load(os.path.join(ROOT, "models", f"{a.name}.pt"), map_location="cpu")
    args, lm = ck["args"], ck["label_map"]
    names = [None] * len(lm)
    for k, v in lm.items():
        names[v] = k

    mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
    ds = ISLDataset(mp, a.split, lm, seq_len=args["seq_len"], use_z=not args["no_z"],
                    fill=args.get("fill_gaps", False))
    dl = DataLoader(ds, batch_size=32)

    model = build_model(args.get("model", "bilstm"), ck["input_dim"], len(lm),
                        hidden=args["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    probs, labels = [], []
    with torch.no_grad():
        for x, y in dl:
            probs.append(torch.softmax(model(x), 1))
            labels.append(y)
    probs, labels = torch.cat(probs), torch.cat(labels).numpy()
    preds = probs.argmax(1).numpy()
    conf = probs.max(1).values.numpy()
    n = len(names)

    cm = confusion_matrix(labels, preds, labels=range(n))
    print(f"{a.split} accuracy: {(preds == labels).mean():.3f} ({(preds == labels).sum()}/{len(labels)})\n")

    # 1. misclassified clips
    print("Misclassified clips:")
    for i in np.where(preds != labels)[0]:
        print(f"  {ds.df.loc[i, 'video']:<22} true={names[labels[i]]:<15} "
              f"pred={names[preds[i]]:<15} conf={conf[i]:.2f}")

    # 2. most confused pairs
    off = cm.copy()
    np.fill_diagonal(off, 0)
    print("\nTop confused pairs (true -> predicted):")
    for idx in np.argsort(off, axis=None)[::-1][:10]:
        t, p = divmod(idx, n)
        if off[t, p] == 0:
            break
        print(f"  {names[t]} -> {names[p]}: {off[t, p]}")

    # 3. per-class report (only classes present in this split)
    present = sorted(set(labels) | set(preds))
    print("\n" + classification_report(labels, preds, labels=present,
                                       target_names=[names[i] for i in present],
                                       zero_division=0))

    # 4. heatmap (size capped so it still renders with 263 classes)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    size = min(max(6, n * 0.45), 30)
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    if n <= 80:
        fs = 8 if n <= 40 else 5
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(names, rotation=90, fontsize=fs)
        ax.set_yticklabels(names, fontsize=fs)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{a.name} - {a.split} confusion matrix (row-normalized)")
    if n <= 30:
        for i in range(n):
            for j in range(n):
                if cm[i, j]:
                    ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=8,
                            color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    out = os.path.join(ROOT, "outputs", f"{a.name}_{a.split}_confusion.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=100)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()