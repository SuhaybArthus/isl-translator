# ISL Translator

Real-time **Indian Sign Language (ISL) word recognition** in the browser. A webcam feed is tracked locally with MediaPipe, only landmark numbers are sent to a small FastAPI service, and an ONNX Transformer returns the recognized word.

**Status:** Phase 1 (recognition) works end to end. Live webcam accuracy has **not** been measured yet, so the headline numbers below are on the INCLUDE benchmark, not on real use. Phase 2 (turning words into sentences in Tamil, Hindi, Telugu and Bengali) has not started. See [Limitations](#limitations-and-status) and [MODEL_CARD.md](MODEL_CARD.md).

## Results

All numbers are top-1 accuracy on the official INCLUDE test split (811 clips, 262 words) unless stated otherwise. "Seeds" are independent training runs; the spread is the standard deviation over 3 seeds, which understates the real uncertainty.

| Model | Test top-1 | Test top-5 | Macro-F1 |
|---|---|---|---|
| BiLSTM | 0.911 ± 0.005 | 0.982 | 0.906 |
| Transformer | 0.939 ± 0.005 | 0.990 | 0.939 |
| Transformer, x/y only (no z) | 0.947 ± 0.006 | 0.991 | 0.948 |
| **Transformer + hand-dropout augmentation** | **0.950 ± 0.007** | **0.993** | **0.952** |
| Transformer + mirror augmentation (1 seed) | 0.891 | 0.985 | 0.893 |
| **Released model `final_v1`** (hand-dropout model, fine-tuned, adds a "No sign" class) | 0.952 (served ONNX) | n/a | n/a |

**Where the accuracy comes from.** Most test clips (762 of 811) have a near-identical take by the same signer in training. On the 49 test clips from sessions with no such take, accuracy is much lower:

| Model (3 seeds, mean) | Seen sessions (762 clips) | Held-out sessions (49 clips) |
|---|---|---|
| BiLSTM | 0.929 | 0.633 |
| Transformer | 0.952 | 0.728 |
| Transformer + hand dropout | 0.956 | 0.850 |

The 49-clip subset is small (one clip is 2 points) and "session" is inferred from file numbers, so read it as a warning about new signers and cameras, not as a precise estimate. Hand-dropout augmentation beat the plain Transformer on every seed here.

**Left-handed signing.** A model trained on right-handed signers scores 0.242 on mirrored test clips. The service mirrors a sign when the left wrist clearly leads and, when unclear, keeps the more confident of the two readings. On the base model of the release this gives 0.935 on both the original and the mirrored test set (against 0.942 for the original alone). The mirrored set is a computer mirror of right-handed clips, not real left-handed signers.

**Speed** (model only, batch of 1, CPU of the development laptop):

| | Median | p95 | File size |
|---|---|---|---|
| PyTorch | 1.45 ms | 2.03 ms | n/a |
| ONNX Runtime, FP32 | 0.82 ms | 1.09 ms | 1.99 MB |
| ONNX Runtime, INT8 | 0.70 ms | 1.01 ms | 0.67 MB |

INT8 agrees with FP32 on 99.8% of test clips and scores 0.949 against 0.952, so the service uses FP32 by default. Server time for a full sign (preprocessing plus the model) was 3 to 6 ms in an earlier replay test. End-to-end latency in the browser has not been measured; most of it is tracking and the wait for the sign to end (about a third of a second).

## How it works

```
webcam -> MediaPipe Holistic in the browser -> 225 numbers per frame (75 landmarks x x,y,z)
       -> blended to a steady 25 fps -> WebSocket -> sign segmenter -> checks
       -> shoulder-normalized, gap-filled, 64-frame clip -> ONNX Transformer -> word
```

- **Landmarks, not video.** Pose (33), left hand (21) and right hand (21) points. No video leaves the browser.
- **Preprocessing** (`src/inference/preprocess.py`, shared by training and serving, with a test that checks they agree): centre on the shoulders, scale by shoulder width, fill hand-tracking gaps of up to 5 frames by interpolation, resample to 64 frames. Missing landmarks stay zero.
- **Segmenter** (`src/inference/segment.py`): a sign starts when a wrist rises above a rest line and ends after the hands stay down for 8 frames, with padding of 12 frames before and 8 after. The page calibrates the rest line from your resting hands when the camera starts.
- **Guards:** segments where the wrists barely move, or where a hand is tracked in under half of the frames, are ignored; so is a "No sign" prediction. The page only accepts a word above a confidence setting (50% by default) that also beats the runner-up by 10 points.
- **Left-handed signers:** see `src/inference/handedness.py`.

## Repository layout

```
api/main.py             FastAPI service: /ws, /health, recording endpoints, static files
web/                    translator page (index.html) and recording page (record.html), css/, js/
src/data/dataset.py     dataset, normalization, augmentation
src/models/             BiLSTM and Transformer
src/training/           training and evaluation
src/inference/          preprocessing, segmenter, handedness helpers (numpy only)
scripts/                ingest, splits, evaluation, ONNX export, fine-tuning, diagnostics
tests/                  unit tests (CI runs the ones that need no data)
Dockerfile, requirements-serve.txt, .github/workflows/ci.yml
```

## Setup

Two environments are needed, because MediaPipe and recent `onnx` require incompatible `protobuf` versions.

```powershell
python -m venv .venv                       # data extraction and training (MediaPipe, PyTorch)
.venv\Scripts\python -m pip install -r requirements.txt
python -m venv .venv-onnx                  # ONNX export and the service
.venv-onnx\Scripts\python -m pip install torch numpy pandas scikit-learn onnx onnxruntime fastapi "uvicorn[standard]" websockets
```

Use Python 3.9 to 3.12 (MediaPipe).

## Reproduce

1. **Data.** Download the INCLUDE zips from [Zenodo](https://zenodo.org/record/4010759) into `data/zips/`, and the official `include_train.txt`, `include_val.txt` and `include_test.txt` from [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE) (folder `train_test_paths`) into `data/raw/splits/`. Check the dataset's terms of use.
2. **Extract keypoints and build the split:**
   ```powershell
   python scripts\ingest.py --delete_zip      # unzip, extract landmarks, delete the videos
   python scripts\build_manifest.py
   python scripts\make_splits.py
   python scripts\check_duplicates.py         # expect 0 duplicate groups
   ```
   In this project 4,284 clips and 262 of the 263 words were ingested (8 clips of "Second (Number)" are missing).
3. **Train** (about 100 epochs on CPU; the three-seed ablation takes hours):
   ```powershell
   python -m src.training.train --name abl_drop_s1 --model transformer --seed 1 --fill_gaps --p_hand_drop 0.3
   python -m src.training.evaluate --name abl_drop_s1
   python scripts\leakage_check.py abl_drop_s1       # seen vs held-out sessions
   python scripts\summarize.py                       # mean and spread over seeds
   ```
4. **Export:** `.venv-onnx\Scripts\python scripts\export_onnx.py --name final_v1` writes FP32 and INT8 ONNX files and a `.meta.json` with the labels.

## Run the service

```powershell
$env:ISL_MODEL="final_v1"
.venv-onnx\Scripts\python -m uvicorn api.main:app --port 8000 --ws-max-size 16384
```

Open `http://127.0.0.1:8000`. Settings (environment variables):

| Variable | Default | Meaning |
|---|---|---|
| `ISL_MODEL` | `final_v1` | model name in `models/` |
| `ISL_VARIANT` | empty | `_int8` for the quantized file |
| `ISL_MIN_MOTION` | 1.0 | wrist travel, in shoulder widths, below which a segment is ignored |
| `ISL_MIN_HANDS` | 0.5 | share of frames with a tracked hand needed |
| `ISL_HAND_K` | 1.5 | how clearly one wrist must lead before mirroring |
| `ISL_RECORDING` | 1 | set to `0` on any shared server (the save endpoint has no login) |
| `ISL_MAX_CONNECTIONS` | 10 | simultaneous websocket clients; extra ones are told the server is busy |
| `ISL_IDLE_SECONDS` | 60 | a client that sends nothing for this long is disconnected |
| `ISL_MAX_FPS` | 100 | messages per second allowed from one client (a real page sends about 25) |

**Endpoints.** `GET /health`; `WebSocket /ws` takes one frame of 225 float32 values (binary) or `{"frame": [...]}`, and commands `{"cmd": "reset"}` and `{"cmd": "config", "rest_line": 1.2}`; it replies with `state`, `prediction` (top 5 labels with probabilities, timings) or `ignored` messages. `GET/POST /api/recordings` and `GET /api/labels` serve the recording page.

**Docker.** `docker build -t isl-translator .` then `docker run -p 8000:8000 isl-translator`. The image copies only the ONNX model, its metadata and the serving code; `models/` is git-ignored except the released model files, so keep the `COPY` line and the `.gitignore` exceptions in step with the model you release. I have not been able to test the Docker build here.

## Record your own signs and fine-tune

The recording page (`/record.html`) saves your takes as landmark files in `data/recordings/<word>/` with a countdown, automatic stop, trimming and a "No sign" word for idle behaviour. Raw takes are never modified by the fine-tuning script.

```powershell
python scripts\finetune_recordings.py --base abl_drop_s1 --name final_v1 --add_words "No sign"
```

It mixes your takes with INCLUDE training clips, holds out the latest session for testing, and saves a new model without touching the base. The released model was fine-tuned on **58 recorded takes (7 words including "No sign")** with 17 held out. Those held-out takes scored 17 of 17, which is too few, and probably too close to the training takes, to say anything about live accuracy.

## Limitations and status

- **Live accuracy is unmeasured.** INCLUDE was recorded with full-body, fixed cameras and a standing signer. A seated webcam user is a different setting, and the model is much weaker on unseen sessions (see the table above).
- **Isolated words only.** It recognizes one sign at a time from a fixed vocabulary of 262 INCLUDE words plus "No sign". It does not do continuous signing, fingerspelling or grammar. ISL has regional variation that INCLUDE does not cover.
- **"No sign" is thinly trained.** The idle class comes from a few recorded takes. False words when you are not signing may still appear; the guards reduce them but do not remove them.
- **Browser tracking differs from the training extraction.** Landmarks were extracted with MediaPipe Python at 25 fps, while the browser tracks at about 12 to 17 frames per second on the development laptop and the page fills in the gaps by blending. Parity was checked on one clip only.
- **Left-handed signing** is handled by a mirroring rule evaluated on computer-mirrored clips only.
- The API limits connections, idle sockets and message rates. I tested that logic against stand-ins for FastAPI, so run `scripts/load_check.py` against a real server. The Docker build and the CI run still need to be confirmed.

## Roadmap

- [ ] Measure live accuracy on held-out takes from another day and report before and after fine-tuning
- [ ] Measure end-to-end latency in the browser
- [x] Connection cap, idle timeout and message-rate limit in the API
- [ ] Verify the Docker build and confirm CI is green
- [ ] Phase 2: word buffer, a strict gloss-to-sentence prompt, and Tamil, Hindi, Telugu and Bengali output (an LLM API and IndicTrans2 compared, with native-speaker review)

## Acknowledgements and licence

- Dataset and benchmark split: Sridhar, Ganesan, Kumar and Khapra, *INCLUDE: A Large Scale Dataset for Indian Sign Language Recognition*, ACM Multimedia 2020, doi:10.1145/3394171.3413528. Code: [AI4Bharat/INCLUDE](https://github.com/AI4Bharat/INCLUDE).
- Inspired by [aju22/Real-Time-ISL-Translation](https://github.com/aju22/Real-Time-ISL-Translation). This project is an independent reimplementation with a different pipeline and the full dataset.
- Landmark tracking: [MediaPipe](https://developers.google.com/mediapipe).
- Licence for this code: not chosen yet. Add a `LICENSE` file before sharing it.