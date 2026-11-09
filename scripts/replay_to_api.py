import asyncio
import json
import os
import sys

import numpy as np
import pandas as pd
import websockets

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


async def main(n):
    df = pd.read_csv(os.path.join(ROOT, "data", "processed", "manifest_split.csv"))
    df = df[df["split"] == "test"].sample(n, random_state=0)
    ok = got = 0
    async with websockets.connect("ws://localhost:8000/ws") as ws:
        for _, r in df.iterrows():
            arr = np.load(os.path.join(ROOT, r["path"])).astype(np.float32)
            frames = np.concatenate([np.repeat(arr[:1], 10, 0), arr, np.repeat(arr[:1], 12, 0)])
            for f in frames:
                await ws.send(f.tobytes())
            preds = []
            try:
                while True:
                    m = await asyncio.wait_for(ws.recv(), 1.0)
                    d = json.loads(m)
                    if d.get("type") == "prediction":
                        preds.append(d)
            except asyncio.TimeoutError:
                pass
            if not preds:
                print(f"{r['label']:<16} -> no segment")
                continue
            top = preds[0]["top"][0]
            got += 1
            ok += top["label"] == r["label"]
            print(f"{r['label']:<16} -> {top['label']:<16} p={top['p']:.2f} "
                  f"segments={len(preds)} latency={preds[0]['latency_ms']}ms")
    print(f"\n{ok}/{got} correct, {len(df) - got} clips produced no segment")


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 20))