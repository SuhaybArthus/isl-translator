import os
os.environ["GLOG_minloglevel"] = "2"

import argparse
import glob
import shutil
import sys
import zipfile

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_keypoints import process_video, ROOT, WIDTH, VIDEO_EXT

RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zips", default=os.path.join(ROOT, "data", "zips"))
    ap.add_argument("--delete_zip", action="store_true")
    args = ap.parse_args()

    zips = sorted(glob.glob(os.path.join(args.zips, "*.zip")))
    print(f"Found {len(zips)} zips in {args.zips}")

    for zp in zips:
        stem = os.path.splitext(os.path.basename(zp))[0]      # e.g. Adjectives_1of8
        dest = os.path.join(RAW, stem)
        print(f"\n=== {stem} ===")
        with zipfile.ZipFile(zp) as z:
            members = [m.replace("\\", "/") for m in z.namelist()
                       if m.lower().endswith(VIDEO_EXT)]
            todo = [m for m in members
                    if not os.path.exists(os.path.join(OUT, stem, *os.path.splitext(m)[0].split("/")) + ".npy")]
            if todo:
                z.extractall(dest, todo)

        fails = 0
        for m in tqdm(todo, desc=stem):
            v = os.path.join(dest, *m.split("/"))
            dst = os.path.join(OUT, stem, *os.path.splitext(m)[0].split("/")) + ".npy"
            try:
                arr = process_video(v, WIDTH)
            except Exception as e:
                print(f"Failed: {m} ({e})")
                fails += 1
                continue
            if arr is None:
                print(f"No frames: {m}")
                fails += 1
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, arr)

        done = sum(os.path.exists(os.path.join(OUT, stem, *os.path.splitext(m)[0].split("/")) + ".npy")
                   for m in members)
        print(f"{stem}: {done}/{len(members)} videos have keypoints, {fails} failed")
        if done == len(members) and fails == 0:
            shutil.rmtree(dest, ignore_errors=True)            # delete extracted videos
            if args.delete_zip:
                os.remove(zp)
            print("Cleaned up videos.")
        else:
            print("Kept extracted videos so you can inspect the failures.")


if __name__ == "__main__":
    main()