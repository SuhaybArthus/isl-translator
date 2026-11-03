import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

from src.data.dataset import ISLDataset, build_label_map, ROOT
from src.models.factory import build_model


def evaluate(model, loader, device, k=5):
    model.eval()
    preds, labels, topk, n = [], [], 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())
            kk = min(k, logits.shape[1])
            topk += (logits.topk(kk, 1).indices == y[:, None]).any(1).sum().item()
            n += len(y)
    p, l = torch.cat(preds).numpy(), torch.cat(labels).numpy()
    return {
        "top1": float((p == l).mean()),
        "top5": topk / n,
        "macro_f1": float(f1_score(l, p, average="macro")),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="bilstm")
    ap.add_argument("--model", default="bilstm", choices=["bilstm", "transformer"])
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=64)
    ap.add_argument("--p_mirror", type=float, default=0.0)
    ap.add_argument("--p_hand_drop", type=float, default=0.0)
    ap.add_argument("--fill_gaps", action="store_true")
    ap.add_argument("--no_z", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device, "|", vars(args))

    mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
    lm = build_label_map(mp)
    use_z = not args.no_z
    common = dict(label_map=lm, seq_len=args.seq_len, use_z=use_z, fill=args.fill_gaps)
    train_ds = ISLDataset(mp, "train", augment=True, p_mirror=args.p_mirror,
                          p_hand_drop=args.p_hand_drop, **common)
    val_ds = ISLDataset(mp, "val", **common)
    test_ds = ISLDataset(mp, "test", **common)
    train_dl = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=32, num_workers=0)

    input_dim = 225 if use_z else 150
    model = build_model(args.model, input_dim, len(lm), hidden=args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    ckpt = os.path.join(ROOT, "models", f"{args.name}.pt")
    best_val = -1.0

    for ep in range(1, args.epochs + 1):
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

        val = evaluate(model, val_dl, device)
        if val["top1"] > best_val:
            best_val = val["top1"]
            torch.save({"state_dict": model.state_dict(), "args": vars(args),
                        "label_map": lm, "input_dim": input_dim}, ckpt)
        if ep % 10 == 0 or ep == 1:
            print(f"ep {ep:3d} | loss {total / len(train_ds):.3f} | "
                  f"val top1 {val['top1']:.3f} f1 {val['macro_f1']:.3f}")

    # final: load best-by-val checkpoint, evaluate on test once
    model.load_state_dict(torch.load(ckpt, map_location=device)["state_dict"])
    val, test = evaluate(model, val_dl, device), evaluate(model, test_dl, device)
    print("\nBEST VAL :", val)
    print("TEST     :", test)
    with open(os.path.join(ROOT, "outputs", f"{args.name}_results.json"), "w") as f:
        json.dump({"args": vars(args), "val": val, "test": test}, f, indent=2)


if __name__ == "__main__":
    main()