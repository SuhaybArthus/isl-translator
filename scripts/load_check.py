"""Checks the connection cap and the idle timeout on a RUNNING server (not a stand-in).

Start the server with small limits, in one terminal:
  $env:ISL_MAX_CONNECTIONS="3"; $env:ISL_IDLE_SECONDS="5"
  .venv-onnx\\Scripts\\python -m uvicorn api.main:app --port 8000
Then, in another:
  .venv-onnx\\Scripts\\python scripts\\load_check.py 3 5
"""
import asyncio
import json
import sys

import websockets

URL = "ws://127.0.0.1:8000/ws"


async def main(cap, idle):
    held = [await websockets.connect(URL) for _ in range(cap)]
    extra = await websockets.connect(URL)
    msg = json.loads(await asyncio.wait_for(extra.recv(), 3))
    await extra.wait_closed()
    print("PASS" if msg.get("type") == "busy" and extra.close_code == 1013 else "FAIL",
          f"connection {cap + 1} is refused as busy (got '{msg.get('type')}', close code {extra.close_code})")

    await held[0].close()
    await asyncio.sleep(0.5)
    again = await websockets.connect(URL)
    await again.send(bytes(900))                          # one frame of 225 zeros
    try:
        msg = json.loads(await asyncio.wait_for(again.recv(), 1.0))
        print("FAIL", "a slot freed by a client leaving was not reusable, got", msg.get("type"))
    except asyncio.TimeoutError:
        print("PASS", "a slot freed by a client leaving is reusable")

    print(f"waiting up to {idle + 3} s for the idle timeout on a silent connection...")
    try:
        msg = json.loads(await asyncio.wait_for(held[1].recv(), idle + 3))
        print("PASS" if msg.get("type") == "idle" else "FAIL", f"a silent connection receives '{msg.get('type')}'")
    except asyncio.TimeoutError:
        print("FAIL", "a silent connection was not closed")
    for ws in held[1:] + [again]:
        await ws.close()


asyncio.run(main(int(sys.argv[1]) if len(sys.argv) > 1 else 3, float(sys.argv[2]) if len(sys.argv) > 2 else 5))