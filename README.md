# HeyGem12

**English | [简体中文](README.zh-CN.md)**

A pure-Python reimplementation of the HeyGem lip-sync engine for Python 3.12+.
Input a video plus a speech audio, output a lip-synced video.

**Why rewrite?** Large parts of the original HeyGem (face detection, landmark
extraction, audio features, ...) ship as **closed-source Cython `.so` binaries**
that only run on **Python 3.8** — upgrading Python breaks them. This project
reimplements the whole inference pipeline in **pure Python with zero `.so`
dependency**, so it runs natively on modern Python (3.12+).

Model weights (`dinet_v1_20240131.pth`, `wenetmodel.pt`) are the publicly
released downloads from
[HeyGem-Linux-Python-Hack](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack);
all inference code is a clean-room Python reimplementation.

**Output quality is on par with the original HeyGem.** Lip timing, teeth,
color, temporal stability and output length are all frame-aligned with the
original (verified on 23 paired clips rendered from the same source videos),
while several known issues of the original are fixed (hard failure on
multiple faces, stale-result caching in batch runs, open-mouth artifacts),
plus automatic largest-face selection, 60 fps input support and
faststart-enabled outputs.

## Demo

Six source videos, each driven by a Chinese and an English voice (inference on
an RTX 4090D, CUDA). Click a file name to download and watch (mp4):

| Source video | Chinese voice | English voice |
|---|---|---|
| [woman1_video.mp4](assets/woman1_video.mp4) | [woman1_cn.mp4](assets/woman1_cn.mp4) | [woman1_en.mp4](assets/woman1_en.mp4) |
| [woman2_video.mp4](assets/woman2_video.mp4) | [woman2_cn.mp4](assets/woman2_cn.mp4) | [woman2_en.mp4](assets/woman2_en.mp4) |
| [man1_video.mp4](assets/man1_video.mp4) | [man1_cn.mp4](assets/man1_cn.mp4) | [man1_en.mp4](assets/man1_en.mp4) |
| [man2_video.mp4](assets/man2_video.mp4) | [man2_cn.mp4](assets/man2_cn.mp4) | [man2_en.mp4](assets/man2_en.mp4) |
| [man3_video.mp4](assets/man3_video.mp4) | [man3_cn.mp4](assets/man3_cn.mp4) | [man3_en.mp4](assets/man3_en.mp4) |
| [man4_video.mp4](assets/man4_video.mp4) | [man4_cn.mp4](assets/man4_cn.mp4) | [man4_en.mp4](assets/man4_en.mp4) |

Driving audio: [man_cn.wav](assets/man_cn.wav) / [woman_cn.wav](assets/woman_cn.wav) (Chinese),
[man_en.wav](assets/man_en.wav) / [women_en.wav](assets/women_en.wav) (English).

## How it works

```
audio wav ──dB-mel──▶ WeNet conformer encoder (256-d, 25fps)
video mp4 ──per frame──▶ YuNet face detection ──▶ PFLD-98 landmarks
                              ──5-point semantic calibration──▶ 256×256 aligned crop
                        │
        GFPGAN restore of the source frame for the crop
        (eyes/neck keep the raw pixels)
                        │
        DINetV1: masked crop + same-frame unmasked reference + audio window (T=20)
                 ──▶ mouth inpainting + AdaAT appearance deformation
                        │
        paste back through a canonical face-oval mask
        (gain match + feathered border/neck) ──▶ output mp4 (+ driving audio)
```

Key points:

- **Audio frontend**: 16 kHz mono, 0.4 s of silence padded on both ends, 80-d
  dB-mel, conformer encoder producing 256-d features at 25 fps. Video frame `i`
  takes an audio window aligned to the video fps and audio duration — sliding
  per frame for short videos; for long videos the usable feature span is
  linearly resampled to the output length (25/30/60 fps all supported).
- **Keypoint chain**: YuNet detection → PFLD-98 landmarks → per-video semantic
  calibration (eye-distance ratio + nose/mouth vertical ratio, absorbing the
  landmark semantic drift across faces) → temporal stabilization (2-frame spike
  interpolation + median-3 + hard-switch EMA + scale anchoring): sub-pixel
  jitter when still, zero-lag tracking on large head motion.
- **Compositing**: a canonical face-oval mask (fixed in crop space) is pasted
  back into the source frame with a gain match to cancel the render's brightness
  offset, and feathered edges at the border and neck for a seam-free result;
  eyes and neck keep the raw image quality from the restoration stage.

## Installation

Python 3.12+ (3.8 not required). Separate dependency files for CPU and GPU:

```bash
cd heygem12
python -m venv .venv

# CPU build
.venv\Scripts\python -m pip install -r requirements-cpu.txt

# GPU build (CUDA 12.1; torch installs from the official PyTorch index)
.venv\Scripts\python -m pip install -r requirements-gpu.txt

python scripts\download_models.py        # download models into checkpoints\ (~890MB)
```

ONNX runtimes (GFPGAN/SCRFD/PFLD) prefer CUDA automatically and fall back to
CPU — no manual configuration needed.

## Model checkpoints

Three ways to get the models: run `python scripts\download_models.py` to fetch
them all; click the links in the table below to download one by one; or grab
the full set from **Baidu Netdisk** (extraction code `mgas`):
[https://pan.baidu.com/s/109Npnbya4_r97e0BrJbRCg?pwd=mgas](https://pan.baidu.com/s/109Npnbya4_r97e0BrJbRCg?pwd=mgas)

| Model | Size | Purpose |
|---|---|---|
| [dinet_v1_20240131.pth](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/dinet_v1_20240131.pth) | 374MB | ★ Core lip-sync network DINetV1: masked crop + same-frame reference + audio window → re-rendered mouth |
| [wenetmodel.pt](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/wenetmodel.pt) | 182MB | Speech features: conformer encoder, wav → 256-d acoustic features (25fps) |
| [GFPGANv1.4.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/GFPGANv1.4.onnx) | 325MB | Face restoration: super-resolves the source face as DINet's sharp input (eyes/neck kept raw) |
| [pfpld_robust_sim_bs1_8003.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/pfpld_robust_sim_bs1_8003.onnx) | 6.6MB | Facial landmarks: PFLD-98 points reduced to five, used for the 256×256 aligned crop |
| [scrfd_500m_bnkps_shape640x640.onnx](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack/releases/download/ckpts_and_onnx/scrfd_500m_bnkps_shape640x640.onnx) | 2.4MB | Semantic calibration reference: corrects per-video geometric offset of the PFLD five points (prevents face-shape distortion) |
| [face_detection_yunet_2023mar.onnx](https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | 0.2MB | Face detection: locates the face box and rough five points (entry of the alignment chain; from OpenCV Zoo) |

~890MB in total; drop them into `checkpoints/`. The first five come from the
`ckpts_and_onnx` release of
[HeyGem-Linux-Python-Hack](https://github.com/Holasyb918/HeyGem-Linux-Python-Hack);
the download script automatically uses a gh-proxy mirror when GitHub is slow.

## Usage

```bash
.venv\Scripts\python -m heygem12.cli --video input.mp4 --audio voice.wav --output result.mp4
```

Options:

| Option | Description |
|---|---|
| `--video` | Source video (near-frontal face works best; 25/30/60fps all supported) |
| `--audio` | Driving audio (wav, any length; a longer audio truncates the output to the audio span) |
| `--output` | Output mp4 path |
| `--device` | `cpu` (default) or `cuda` (GPU build of torch required) |
| `--max-frames` | Limit the number of frames (debugging) |
| `--no-kps-stabilize` | Disable keypoint temporal stabilization (on by default; keep it on) |
| `--no-pre-restore` | Disable the GFPGAN source-frame restoration (on by default) |
| `--restore` | Enable post-render GFPGAN (off by default) |
| `--fuse-mode` | `oval` (default, face-oval hard paste) or `fuse_mask` (feathered fusion) |

## Python API

```python
from heygem12.pipeline import LipSyncPipeline

pipe = LipSyncPipeline("checkpoints", device="cpu")
pipe.run("input.mp4", "voice.wav", "result.mp4")
```

## Speed reference

| Hardware | Synthesis (incl. GFPGAN restore) | Full 15s video (900 frames @60fps) |
|---|---|---|
| CPU (16 threads) | ~600 ms/frame | not recommended |
| RTX 4090D (CUDA) | ~76 ms/frame | ~100 s |

(Face detection ~25 ms/frame and video writing are extra; the fusion stage is
CPU image processing at ~38 ms/frame.)

## Known limitations

- Mouth amplitude deviates ±20~30% on some frames; fast head turns (motion
  blur) show a 4-8px geometric offset for a few frames — the detector noise
  floor; the under-chin area carries ~1-2px content-level differences.
- Model weights are for personal / non-commercial use only (HeyGem license);
  the code itself is free to use, but commercial use requires replacing or
  retraining the DINet weights.

## Repository layout

```
heygem12/
├── heygem12/
│   ├── models/        DINetV1 network (reverse-engineered reimplementation)
│   ├── audio/         WeNet feature frontend (vendored wenet transformer)
│   ├── face/          YuNet detection / PFLD-98 landmarks / SCRFD / alignment & paste masks
│   ├── third_party/   wenet transformer code (Apache-2.0)
│   ├── pipeline.py    end-to-end pipeline
│   └── cli.py         command-line entry
├── scripts/           model download / comparison-video tools
├── checkpoints/       model weights (~890MB after download)
└── assets/            demo videos and driving audio
```

## Star History

[![Star History Chart](https://api.star-history.com/chart?repos=oaifaye/heygem12&type=date&legend=top-left)](https://www.star-history.com/?repos=oaifaye%2Fheygem12&type=date&legend=top-left)