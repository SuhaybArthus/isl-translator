import os
os.environ["GLOG_minloglevel"] = "2"   # hides the repeated TensorFlow warnings

import sys
import cv2
import mediapipe as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extract_keypoints import ROOT

CLIP = os.path.join(ROOT, "data", "raw", "Animals_1of2", "Animals", "1. Dog", "MVI_8560.MP4")
OUT = os.path.join(ROOT, "outputs", "inspect")
os.makedirs(OUT, exist_ok=True)
draw, hol = mp.solutions.drawing_utils, mp.solutions.holistic


def run(det, cx, tag, save=False):
    cap = cv2.VideoCapture(CLIP)
    n = l = r = 0
    with hol.Holistic(model_complexity=cx, min_detection_confidence=det,
                      min_tracking_confidence=det) as h:
        while True:
            ok, f = cap.read()
            if not ok:
                break
            f = cv2.resize(f, (640, int(f.shape[0] * 640 / f.shape[1])))
            res = h.process(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
            n += 1
            l += res.left_hand_landmarks is not None
            r += res.right_hand_landmarks is not None
            if save and n % 10 == 0:
                for lm in (res.pose_landmarks,):
                    draw.draw_landmarks(f, lm, hol.POSE_CONNECTIONS)
                for lm in (res.left_hand_landmarks, res.right_hand_landmarks):
                    draw.draw_landmarks(f, lm, hol.HAND_CONNECTIONS)
                cv2.imwrite(os.path.join(OUT, f"{tag}_f{n:03d}.jpg"), f)
    print(f"{tag:<18} left {l / n:.2f}  right {r / n:.2f}")


run(0.5, 1, "default", save=True)
run(0.3, 1, "conf0.3")
run(0.3, 2, "conf0.3_cx2")
run(0.1, 2, "conf0.1_cx2")