import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_keypoints import process_video, ROOT

DOG = os.path.join(ROOT, "data", "raw", "Animals_1of2", "Animals", "1. Dog")
CLIPS = ["MVI_8560.MP4", "MVI_8561.MP4", "MVI_3086.MOV", "MVI_3002.MOV"]  # last = good control

print(f"{'clip':<14}{'width':>6}{'left':>7}{'right':>7}{'sec':>7}")
for c in CLIPS:
    for w in [640, 960, 1280, 1920]:
        t = time.time()
        arr = process_video(os.path.join(DOG, c), w)
        l = (arr[:, 99:162] != 0).any(1).mean()
        r = (arr[:, 162:225] != 0).any(1).mean()
        print(f"{c:<14}{w:>6}{l:>7.2f}{r:>7.2f}{time.time() - t:>7.1f}")