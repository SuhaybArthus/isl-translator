import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import onnx
import onnxruntime as ort
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from src.data.dataset import ISLDataset, ROOT
from src.models.factory import build_model


def export(model, path, seq_len, feat):
    dummy = torch.randn(1, seq_len, feat)
    kw = dict(input_names=["x"], output_names=["logits"],
              dynamic_axes={"x": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17)
    try:
        torch.onnx.export(model, dummy, path, dynamo=False, **kw)   # legacy exporter
    except TypeError:
        torch.onnx.export(model, dummy, path, **kw)                 # older torch: no dynamo arg
    onnx.checker.check_model(onnx.load(path))


def bench(fn, x, warmup=20, runs=200):
    for _ in range(warmup):
        fn(x)
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn(x)
        t.append((time.perf_counter() - s) * 1000)
    return float(np.median(t)), float(np.percentile(t, 95))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="checkpoint name in models/, e.g. full_tf_s1")
    a = ap.parse_args()

    ck = torch.load(os.path.join(ROOT, "models", f"{a.name}.pt"), map_location="cpu")
    args, lm = ck["args"], ck["label_map"]
    seq_len, feat = args["seq_len"], ck["input_dim"]

    model = build_model(args.get("model", "bilstm"), feat, len(lm), hidden=args["hidden"])
    model.load_state_dict(ck["state_dict"])
    model.eval()

    fp32 = os.path.join(ROOT, "models", f"{a.name}.onnx")
    int8 = os.path.join(ROOT, "models", f"{a.name}_int8.onnx")
    export(model, fp32, seq_len, feat)
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8)
    print(f"Exported: {fp32} ({os.path.getsize(fp32) / 1e6:.2f} MB)")
    print(f"Quantized: {int8} ({os.path.getsize(int8) / 1e6:.2f} MB)")

    # metadata the API reads, so serving can't drift from training settings
    names = [None] * len(lm)
    for k, v in lm.items():
        names[v] = k
    meta = {"checkpoint": a.name, "model": args.get("model", "bilstm"), "seq_len": seq_len,
            "input_dim": feat, "use_z": not args["no_z"],
            "fill_gaps": bool(args.get("fill_gaps", False)), "labels": names}
    meta_path = os.path.join(ROOT, "models", f"{a.name}.meta.json")
    json.dump(meta, open(meta_path, "w", encoding="utf-8"), indent=2)
    print(f"Metadata: {meta_path}")

    so = ort.SessionOptions()
    s32 = ort.InferenceSession(fp32, so, providers=["CPUExecutionProvider"])
    s8 = ort.InferenceSession(int8, so, providers=["CPUExecutionProvider"])
    run32 = lambda x: s32.run(None, {"x": x})[0]
    run8 = lambda x: s8.run(None, {"x": x})[0]
    run_pt = lambda x: model(torch.from_numpy(x)).detach().numpy()

    # real clips whose labels this checkpoint knows (old checkpoints have fewer classes)
    mp = os.path.join(ROOT, "data", "processed", "manifest_split.csv")
    ds = ISLDataset(mp, "test", lm, seq_len=seq_len, use_z=not args["no_z"],
                    fill=args.get("fill_gaps", False))
    keep = [i for i in range(len(ds)) if ds.df.loc[i, "label"] in lm]
    result = {"checkpoint": a.name, "size_mb": {"fp32": os.path.getsize(fp32) / 1e6,
                                                 "int8": os.path.getsize(int8) / 1e6}}

    if keep:
        X = np.stack([ds[i][0].numpy() for i in keep]).astype(np.float32)
        Y = np.array([ds[i][1] for i in keep])
        with torch.no_grad():
            p_pt, p32, p8 = run_pt(X), run32(X), run8(X)
        diff = float(np.abs(p_pt - p32).max())
        acc = {k: float((p.argmax(1) == Y).mean()) for k, p in
               [("torch", p_pt), ("onnx_fp32", p32), ("onnx_int8", p8)]}
        result.update({"n_eval_clips": len(keep), "top1": acc,
                       "max_abs_logit_diff_torch_vs_onnx": diff,
                       "int8_agrees_with_fp32": float((p8.argmax(1) == p32.argmax(1)).mean())})
        print(f"\nEval clips: {len(keep)} | max |logit diff| torch vs ONNX fp32: {diff:.2e}")
        print("Top-1:", acc, "| INT8 agrees with FP32 on", f"{result['int8_agrees_with_fp32']:.1%}")
        x1 = X[:1]
    else:
        print("\nNo test clips match this checkpoint's labels; skipping accuracy check.")
        x1 = np.random.randn(1, seq_len, feat).astype(np.float32)

    lat = {}
    with torch.no_grad():
        for k, fn in [("torch", run_pt), ("onnx_fp32", run32), ("onnx_int8", run8)]:
            med, p95 = bench(fn, x1)
            lat[k] = {"median_ms": med, "p95_ms": p95}
            print(f"Latency (batch 1, {seq_len}x{feat}) {k:<10}: median {med:.2f} ms | p95 {p95:.2f} ms")
    result["latency"] = lat

    os.makedirs(os.path.join(ROOT, "outputs"), exist_ok=True)
    out = os.path.join(ROOT, "outputs", f"{a.name}_onnx_benchmark.json")
    json.dump(result, open(out, "w"), indent=2)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()