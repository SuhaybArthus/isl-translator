# ISL Translator

Real-time **Indian Sign Language (ISL) word recognition** from webcam video, with a planned GenAI stage that turns recognized words into grammatical sentences in **Tamil, Hindi, Telugu and Bengali**.

> **Status: work in progress.** The keypoint pipeline, dataset tooling and a BiLSTM baseline are done. Full-vocabulary training, the FastAPI/browser demo and the multilingual stage are planned (see [Roadmap](#roadmap)).

---

## Overview

```
Webcam -> Browser (MediaPipe Holistic JS) -> landmarks only -> FastAPI WebSocket
       -> sign segmentation -> preprocessing -> ONNX model -> word + confidence
       -> word buffer -> LLM (gloss -> English sentence)
       -> IndicTrans2 / LLM (English -> Tamil / Hindi / Telugu / Bengali)
```

Design choices:

- **Keypoints, not pixels.** Models train on MediaPipe pose + hand landmarks, which keeps training light and reduces sensitivity to background and clothing.
- **Split compute.** The target architecture runs landmark extraction in the browser and sends only coordinates to the server (much smaller payload than video frames).
- **One preprocessing module for training and serving**, with an automated parity test, to avoid train/serve mismatch.
- **Official benchmark split** from the INCLUDE authors, so results are comparable and signer sessions don't leak between train and test.

## Dataset

[INCLUDE](https://zenodo.org/record/4010759): about 4,300 videos, 263 ISL signs, 15 categories (check the Zenodo page for license and usage terms).

Ingestion progress (4 of 15 categories):

| Category | Status |
|---|---|
| Greetings, Animals, Days_and_Time, Adjectives | Ingested (1,115 clips, 77 classes) |
| Remaining 11 categories | Not yet ingested |

Train/val/test assignment comes from the official `include_train.txt`, `include_val.txt` and `include_test.txt` (AI4Bharat INCLUDE repo), placed in `data/raw/splits/`.

Note: some signs have only about 8 clips in the official split and none in val, so per-class validation is limited and test results are the more reliable signal.

## Tech stack

| Layer | Tools |
|---|---|
| Language | Python 3.11 |
| Landmarks | OpenCV, MediaPipe Holistic (pose 33 + left hand 21 + right hand 21) |
| Modeling | PyTorch, NumPy, pandas, scikit-learn |
| Testing | pytest |
| Planned | ONNX Runtime, FastAPI + WebSocket, browser MediaPipe JS, Docker, LLM API, IndicTrans2 |

## Project structure

```
isl-detector/
  data/
    zips/            downloaded INCLUDE zips (ingestion input)
    raw/             temporary unzipped videos + splits/ (official split files)
    processed/       keypoint .npy files + manifest.csv, manifest_split.csv
  src/
    data/dataset.py        Dataset class: normalization, gap filling, augmentation
    models/bilstm.py       BiLSTM classifier
    training/train.py      training loop, metrics (top-1, top-5, macro-F1)
    training/evaluate.py   confusion matrix, misclassified clips, per-class report
    inference/preprocess.py  preprocessing shared by training and serving
  scripts/
    extract_keypoints.py   MediaPipe Holistic keypoint extraction
    ingest.py              unzip -> extract keypoints -> verify -> delete videos
    build_manifest.py      rebuild manifest from .npy files
    make_splits.py         attach official train/val/test split
    check_duplicates.py    detect duplicate clips (leakage check)
    summarize.py           mean +/- std over seeds, writes README-ready table
    compare_width.py, inspect_clip.py   extraction diagnostics
  tests/test_preprocess.py   train/serve preprocessing parity test
  configs/labels.json
  api/  web/   (planned)
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

MediaPipe supports roughly Python 3.9 to 3.12. Use the CPU build of PyTorch if you have no NVIDIA GPU.

## Usage

**1. Ingest data.** Put INCLUDE zips in `data/zips/`, and the three official split files in `data/raw/splits/`.

```powershell
python scripts\ingest.py --delete_zip      # unzip, extract keypoints, delete videos and zip
python scripts\build_manifest.py           # rebuild manifest from keypoints
python scripts\make_splits.py              # attach official split
python scripts\check_duplicates.py         # expect 0 groups spanning splits
```

`ingest.py` is resumable (it skips clips that already have keypoints) and only deletes a zip after every clip in it has been processed. Keypoints are the only permanent artifact, so videos never accumulate on disk.

**2. Train and evaluate.**

```powershell
python -m src.training.train --name run1 --seed 1 --fill_gaps
python -m src.training.evaluate --name run1
python scripts\summarize.py                # aggregate runs named *_s<seed>
python -m pytest tests\test_preprocess.py
```

Useful flags: `--no_z` (x,y only), `--p_mirror` (mirror augmentation, swaps left/right landmarks), `--p_hand_drop` (hand-dropout augmentation), `--fill_gaps`.

## Method

**Features.** 75 landmarks x (x, y, z) = 225 values per frame.

**Preprocessing** (`src/inference/preprocess.py`):
- Center on the shoulder midpoint, scale by shoulder width.
- Undetected landmarks stay zero so "not visible" is distinguishable from a real position.
- Optional linear interpolation of short hand-tracking gaps (up to 5 frames, never at clip edges).
- Nearest-frame resampling to a fixed 64 frames.

**Augmentation** (training only): temporal crop, small rotation, scale, shift, jitter, optional mirroring (with left/right swap) and optional hand dropout.

**Model.** LayerNorm, 2-layer BiLSTM, mean + max pooling over time, linear head. Label smoothing 0.1, AdamW, cosine schedule, best checkpoint chosen on validation, test evaluated once.

## Results so far

### Early baseline: 18-class subset (Greetings, Animals, Days_and_Time)

247 train / 27 val / 48 test clips. Test top-1 was about 92% for the first run, with top-5 at 100%.

Ablation over 3 seeds (mean +/- std on test):

| Config | Test top-1 | Test top-5 | Test macro-F1 |
|---|---|---|---|
| base | 0.931 +/- 0.026 | 1.000 | 0.866 +/- 0.033 |
| + gap filling | 0.944 +/- 0.020 | 1.000 | 0.926 +/- 0.066 |
| + gap filling + hand dropout | 0.910 +/- 0.043 | 1.000 | 0.816 +/- 0.082 |

**How to read this honestly:** with 48 test clips, one clip is about 2%, and seed-to-seed variation is as large as the differences between configs. Gap filling was never worse, so it stays on, but this is not a proven gain. Hand dropout showed no benefit at this data size and will be re-tested on the full dataset. Validation accuracy was saturated at 100%, so it cannot separate configs.

### Findings from debugging

- **Hand detection drops on fast signs.** Per-clip hand-detection rates ranged from about 20% to 100%. The weakest clips are fast-motion signs, where hands blur and MediaPipe's tracker loses them.
- **Input resolution did not fix it.** Detection rates on the worst clips were flat from 640 px to 1920 px wide, so extraction stays at 640 px. `static_image_mode` was rejected because it would make training keypoints differ from the tracked keypoints seen in the live demo.
- **Gap filling recovers real frames.** On the 18-class set, frames with a missing left hand fell from 16.7% to 11.8%, and right hand from 25.5% to 16.4%.
- **Most test errors came from low-detection clips.** For example, two Dog clips from a single recording session (also test-only for that sign) were misclassified. Treat this as a data-coverage issue, not a model conclusion.
- **Dataset hygiene.** Duplicate-copy files (names like `MVI_0050_ (1).MOV`) were found in the official test list; they are filtered out, and `check_duplicates.py` verifies no identical clips span splits.

### Current data

1,115 clips, 77 classes (823 train / 94 val / 196 test), 0 duplicate groups. Training on this larger set has not been run yet.

## Limitations

- Recognizes **isolated signs**, not continuous signing or full sentences.
- Vocabulary is limited to the INCLUDE signs.
- Recognition errors will propagate to the sentence stage.
- ISL grammar differs from spoken languages, so generated sentences are approximations.
- Accuracy on the full 263-class set is expected to be lower than the early subset numbers; with some signs at 6 to 15 training clips, data scarcity is a real constraint.

## Roadmap

- [x] Project structure, keypoint extraction, resumable ingestion
- [x] Official train/val/test split, duplicate check
- [x] BiLSTM baseline, evaluation report, 3-seed ablation tooling
- [x] Shared preprocessing module with parity test
- [ ] Ingest remaining 11 categories (target: 263 classes)
- [ ] Full-vocabulary baseline; z-axis, mirror and hand-dropout ablations (3 seeds each)
- [ ] Transformer model; optional ST-GCN ablation
- [ ] ONNX export, INT8 quantization, latency benchmark (PyTorch vs ONNX Runtime)
- [ ] Sliding-window sign segmentation (motion variance / rest pose)
- [ ] FastAPI WebSocket service + browser client with in-browser MediaPipe
- [ ] Record a small personal test set to measure webcam generalization
- [ ] Sentence generation (strictly constrained LLM prompt) and word buffer
- [ ] Multilingual output (Tamil, Hindi, Telugu, Bengali), LLM vs IndicTrans2 comparison, native-speaker review
- [ ] Docker, CI, live demo

## Acknowledgements

- **INCLUDE dataset and benchmark split:** Sridhar, Ganesan, Kumar, Khapra. *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition.* ACM Multimedia 2020. doi:10.1145/3394171.3413528. Official code: [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (MIT).
- Inspired by [aju22/Real-Time-ISL-Translation](https://github.com/aju22/Real-Time-ISL-Translation); this project is an independent reimplementation using the full dataset and a different pipeline.
- Landmarks: [MediaPipe](https://developers.google.com/mediapipe).