import os
import glob
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT_DIR = os.path.join(ROOT, "data", "processed")
WIDTH = 640
LIMIT = None
VIDEO_EXT = (".mov", ".mp4", ".avi", ".mkv")


def lm_array(lms, n):
    if lms is None:
        return np.zeros(n * 3, dtype=np.float32)
    return np.array([[l.x, l.y, l.z] for l in lms.landmark], dtype=np.float32).flatten()


def process_video(path, width):
    cap = cv2.VideoCapture(path)
    feats = []
    with mp.solutions.holistic.Holistic(
        static_image_mode=False, model_complexity=1,
        min_detection_confidence=0.5, min_tracking_confidence=0.5,
    ) as holistic:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            if w > width:
                frame = cv2.resize(frame, (width, int(h * width / w)))
            res = holistic.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            feats.append(np.concatenate([
                lm_array(res.pose_landmarks, 33),
                lm_array(res.left_hand_landmarks, 21),
                lm_array(res.right_hand_landmarks, 21),
            ]))
    cap.release()
    return np.stack(feats) if feats else None


def get_label(video_path):
    d = os.path.dirname(video_path)
    while d and os.path.abspath(d) != os.path.abspath(RAW_DIR):
        head, sep, tail = os.path.basename(d).partition(".")
        if sep and head.strip().isdigit():
            return tail.strip()
        d = os.path.dirname(d)
    return os.path.basename(os.path.dirname(video_path))


def main():
    print("Looking in:", RAW_DIR, "| exists:", os.path.exists(RAW_DIR))
    files = glob.glob(os.path.join(RAW_DIR, "**", "*"), recursive=True)
    videos = sorted(f for f in files if f.lower().endswith(VIDEO_EXT))
    if LIMIT:
        videos = videos[:LIMIT]
    print(f"Found {len(videos)} videos")

    rows = []
    for v in tqdm(videos):
        rel_path = os.path.relpath(v, RAW_DIR)
        parts = rel_path.split(os.sep)
        category = parts[1] if len(parts) >= 4 else parts[0]
        label = get_label(v)

        dst = os.path.join(OUT_DIR, os.path.splitext(rel_path)[0] + ".npy")

        if os.path.exists(dst):
            arr = np.load(dst)  # resume: skip already processed
        else:
            try:
                arr = process_video(v, WIDTH)
            except Exception as e:
                print(f"Failed: {v} ({e})")
                continue
            if arr is None:
                print(f"Skipped (no frames): {v}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            np.save(dst, arr)

        rows.append({
            "path": os.path.relpath(dst, ROOT),
            "category": category,
            "label": label,
            "video": os.path.basename(v),
            "frames": len(arr),
            "left_hand_rate": float((arr[:, 99:162] != 0).any(axis=1).mean()),
            "right_hand_rate": float((arr[:, 162:225] != 0).any(axis=1).mean()),
        })

    os.makedirs(OUT_DIR, exist_ok=True)
    manifest_path = os.path.join(OUT_DIR, "manifest.csv")
    pd.DataFrame(rows).to_csv(manifest_path, index=False)
    print(f"Done. Manifest saved to {manifest_path}")


if __name__ == "__main__":
    main()