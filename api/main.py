"""Live sign recognition service.

WebSocket /ws receives raw landmark frames (225 floats: pose33 + left hand21 + right hand21,
xyz each) as binary float32 or JSON {"frame": [...]}. It segments signs, ignores segments where
the hands barely move, preprocesses each sign exactly like training, handles left-handed signers,
runs ONNX, and replies with top-k labels.
Commands (JSON): {"cmd": "reset"} clears the segmenter, {"cmd": "config", "rest_line": float} tunes it.
"""
import asyncio
import json
import mimetypes
import os
import re
import sys
import time

mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request, WebSocket
from fastapi.staticfiles import StaticFiles

from src.inference.preprocess import preprocess_clip
from src.inference.segment import SignSegmenter

MODEL = os.environ.get("ISL_MODEL", "full_tf_s1")
VARIANT = os.environ.get("ISL_VARIANT", "")                       # "" = fp32, "_int8" = quantized
CONF = float(os.environ.get("ISL_CONF", "0.5"))
MIN_MOTION = float(os.environ.get("ISL_MIN_MOTION", "1.0"))       # shoulder widths of wrist travel
HAND_K = float(os.environ.get("ISL_HAND_K", "1.25"))              # how clearly one wrist must lead
RECORDING = os.environ.get("ISL_RECORDING", "1") == "1"                # set to 0 on any shared or public server
TOPK, N_FEAT = 5, 225

meta = json.load(open(os.path.join(ROOT, "models", f"{MODEL}.meta.json"), encoding="utf-8"))
sess = ort.InferenceSession(os.path.join(ROOT, "models", f"{MODEL}{VARIANT}.onnx"),
                            providers=["CPUExecutionProvider"])
LABELS = meta["labels"]

app = FastAPI(title="ISL recognition")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL, "variant": VARIANT or "fp32",
            "classes": len(LABELS), "seq_len": meta["seq_len"]}


# left/right swap for mirroring a sign (pose pairs, then left hand <-> right hand)
PERM = np.arange(75)
for a_, b_ in [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16), (17, 18),
               (19, 20), (21, 22), (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)]:
    PERM[a_], PERM[b_] = b_, a_
for i_ in range(21):
    PERM[33 + i_], PERM[54 + i_] = 54 + i_, 33 + i_


def mirrored(x, use_z):
    d = 3 if use_z else 2
    m = x.reshape(len(x), 75, d).copy()
    m[..., 0] *= -1
    return m[:, PERM].reshape(x.shape)


def wrist_travel(x, use_z):
    """Wrist travel (left, right) in the preprocessed, shoulder-normalized clip."""
    a = x.reshape(len(x), 75, 3 if use_z else 2)
    out = []
    for w in (15, 16):
        p = a[:, w, :2]
        ok = (p != 0).any(-1)
        out.append(float((np.linalg.norm(np.diff(p, axis=0), axis=-1) * (ok[1:] & ok[:-1])).sum()))
    return out


def wrist_motion(seg):
    """Total wrist travel in a raw segment, in shoulder widths (ignores jitter under 0.004)."""
    sw = np.median(np.linalg.norm(seg[:, 33:35] - seg[:, 36:38], axis=1))
    if sw < 1e-6:
        return 0.0
    total = 0.0
    for c in (45, 48):
        ok = (seg[:, c:c + 2] != 0).any(1)
        d = np.linalg.norm(np.diff(seg[:, c:c + 2], axis=0), axis=1)[ok[1:] & ok[:-1]]
        total += d[d > 0.004].sum()
    return float(total / sw)


def run_probs(x):
    lg = sess.run(None, {"x": x[None]})[0][0]
    p = np.exp(lg - lg.max())
    return p / p.sum()


def predict(segment):
    t = time.perf_counter()
    uz = meta["use_z"]
    x = preprocess_clip(segment, meta["seq_len"], use_z=uz, fill=meta["fill_gaps"])
    mir = mirrored(x, uz)
    tl, tr = wrist_travel(x, uz)
    if tl > HAND_K * tr:                       # left hand leads: show the model a right-handed sign
        p, flipped = run_probs(mir), True
    elif tr > HAND_K * tl:
        p, flipped = run_probs(x), False
    else:                                      # unclear: keep the more confident reading
        pa, pm = run_probs(x), run_probs(mir)
        flipped = bool(pm.max() > pa.max())
        p = pm if flipped else pa
    idx = np.argsort(p)[::-1][:TOPK]
    return {"type": "prediction", "n_frames": int(len(segment)), "mirrored": flipped,
            "top": [{"label": LABELS[i], "p": round(float(p[i]), 4)} for i in idx],
            "confident": bool(p[idx[0]] >= CONF),
            "latency_ms": round((time.perf_counter() - t) * 1000, 2)}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    rest = 1.6
    seg, was_active = SignSegmenter(rest_line=rest), False
    while True:
        msg = await ws.receive()
        if msg["type"] == "websocket.disconnect":
            break
        try:
            if msg.get("bytes") is not None:
                frame = np.frombuffer(msg["bytes"], dtype=np.float32)
            else:
                data = json.loads(msg["text"])
                if data.get("cmd") == "config":
                    rest = float(np.clip(data.get("rest_line", rest), 0.5, 2.5))
                    seg, was_active = SignSegmenter(rest_line=rest), False
                    continue
                if data.get("cmd") == "reset":
                    seg.reset()
                    was_active = False
                    continue
                frame = np.asarray(data["frame"], dtype=np.float32)
        except Exception as e:
            await ws.send_json({"type": "error", "detail": f"bad message: {e}"})
            continue
        if frame.shape != (N_FEAT,) or not np.isfinite(frame).all():
            await ws.send_json({"type": "error", "detail": f"expected {N_FEAT} finite floats"})
            continue

        segment = seg.push(frame)
        if seg.active != was_active:
            was_active = seg.active
            await ws.send_json({"type": "state", "signing": was_active})
        if segment is not None:
            m = wrist_motion(segment)
            if m < MIN_MOTION:
                await ws.send_json({"type": "ignored", "motion": round(m, 2)})
            else:
                out = await asyncio.to_thread(predict, segment)
                out["motion"] = round(m, 2)
                await ws.send_json(out)


# ---- recording page: save your own takes in the same keypoint format as data/processed ----
REC_DIR = os.path.join(ROOT, "data", "recordings")
BAD_CHARS = re.compile(r'[\\/:*?"<>|]|\.\.')
LABEL_OK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ()\-]{0,39}$")


@app.get("/api/labels")
def labels():
    return {"labels": LABELS}


@app.get("/api/recordings")
def recordings():
    out = {}
    if os.path.isdir(REC_DIR):
        for d in sorted(os.listdir(REC_DIR)):
            p = os.path.join(REC_DIR, d)
            if os.path.isdir(p):
                out[d] = len([f for f in os.listdir(p) if f.endswith(".npy")])
    return out


@app.post("/api/recordings")
async def save_recording(request: Request, label: str, hand: str = "right"):
    if not RECORDING:
        raise HTTPException(403, "recording is turned off on this server")
    label = label.strip()
    if not (label in LABELS or LABEL_OK.match(label)) or BAD_CHARS.search(label) or hand not in ("right", "left", "both"):
        raise HTTPException(400, "invalid word or hand")
    raw = await request.body()
    if not raw or len(raw) % (N_FEAT * 4):
        raise HTTPException(400, "body must be float32 frames of 225 values")
    arr = np.frombuffer(raw, dtype=np.float32).reshape(-1, N_FEAT)
    if not 10 <= len(arr) <= 400 or not np.isfinite(arr).all():
        raise HTTPException(400, "need 10 to 400 frames of finite numbers")
    d = os.path.join(REC_DIR, label)
    os.makedirs(d, exist_ok=True)
    name = f"{label.replace(' ', '_')}_{time.strftime('%Y%m%d-%H%M%S')}-{int(time.time() * 1000) % 1000:03d}_{hand[0].upper()}.npy"
    np.save(os.path.join(d, name), arr)
    return {"saved": name, "frames": int(len(arr)), "count": len([f for f in os.listdir(d) if f.endswith(".npy")])}


# must come last so /health and /ws keep working
app.mount("/", StaticFiles(directory=os.path.join(ROOT, "web"), html=True), name="web")