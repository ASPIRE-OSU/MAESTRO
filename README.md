# MAESTRO Benchmark

Official benchmark code for the **MAESTRO** dataset - the Multimodal Auditory-attention Egocentric Speech-TRacking Open corpus. This repository contains the preprocessing pipeline, baseline models, training scripts, and pre-computed results for all four benchmark tasks defined in the accompanying paper.

> **Dataset:** [HuggingFace — aspire-osu/maestro-eeg-dataset](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset)

---

![MAESTRO experimental setup](media/setup.jpg)

## Overview

MAESTRO is a 16-subject, 100-trial multimodal auditory attention decoding (AAD) dataset. Each trial records five synchronised data streams while participants listen to four simultaneously presented speakers and attend to one:

| Modality | Device | Rate |
|---|---|---|
| EEG (32 ch) | ANTNeuro eego mylab | 500 Hz |
| Gaze + pupillometry | Tobii Pro Glasses 3 | ~50 Hz |
| IMU (accel + gyro) | Tobii Pro Glasses 3 | ~120 Hz |
| Egocentric video | Tobii Pro Glasses 3 | 25 fps |
| Audio (4 speakers + 2 noise) | 3 loudspeaker devices | 16 kHz |

---

## Benchmark Tasks

| Task | Description | Chance |
|---|---|---|
| **T1** | Four-class attended source decoding | 25% |
| **T2** | Attended hemisphere decoding (left vs right) | 50% |
| **T3** | Attended eccentricity decoding (inner vs outer) | 50% |
| **T4** | Attended envelope reconstruction (Pearson r) | — |

---

## Results

All results use a 30-second decision window and are averaged across five stratified folds.

### T1 — Four-class attended source decoding

| Condition | EEG only | EEG + Video + Gaze + IMU |
|---|---|---|
| Pooled | 57.4% | 58.7% |
| LOSO | 59.7% | 60.6% |
| Chance | 25.0% | 25.0% |

### T2 — Attended hemisphere decoding

| Mode | Accuracy |
|---|---|
| EEG only | 77.5% |
| EEG + Video + Gaze + IMU | 78.3% |
| Chance | 50.0% |

### T3 — Attended eccentricity decoding

| Mode | Accuracy |
|---|---|
| EEG only | 68.7% |
| EEG + Video + Gaze + IMU | 70.5% |
| Chance | 50.0% |

### T4 — Envelope reconstruction

| Mode | Pearson r |
|---|---|
| EEG only | 0.003 |
| EEG + Video + Gaze + IMU | 0.019 |

---

## Repository Structure

```
MAESTRO/
├── scripts/
│   ├── dataloader.py               # Preprocessing, sync, windowing, PyTorch Dataset
│   ├── model_classification.py     # Multi-encoder dilated conv network (T1)
│   ├── model_spatial.py            # Binary variant for T2 and T3
│   ├── model_reconstruction.py     # Linear backward model + Pearson loss (T4)
│   ├── train_pooled.py             # T1 — pooled 5-fold CV
│   ├── train_loso.py               # T1 — leave-one-subject-out
│   ├── train_hemisphere.py         # T2 — hemisphere decoding
│   ├── train_eccentricity.py       # T3 — eccentricity decoding
│   ├── train_reconstruction.py     # T4 — envelope reconstruction
│   └── dl_maestro.py               # Dataset download script with rate limit handling
└── results/
    ├── results_T1_pooled/          # Pre-computed T1 pooled results
    ├── results_T1_loso/            # Pre-computed T1 LOSO results
    ├── results_T2/                 # Pre-computed T2 results
    ├── results_T3/                 # Pre-computed T3 results
    └── results_T4/                 # Pre-computed T4 results
```

---

## Installation

```bash
git clone https://github.com/NaimulHassan/MAESTRO
cd MAESTRO
pip install -r requirements.txt
```
> **Note:** PyTorch must be installed separately to match your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command. The scripts were tested with Python 3.7, PyTorch 1.13, and CUDA 11.8.

## Downloading the Dataset

The dataset is publicly available on HuggingFace. Use the provided download script which handles rate limiting automatically by downloading one subject at a time with retries:

```bash
python scripts/dl_maestro.py --local_dir maestro-data
```

To download specific subjects only:

```bash
python scripts/dl_maestro.py --local_dir maestro-data --subjects 1 2 3
```

The script downloads in three sequential phases — metadata, modality data (EEG, gaze, IMU), and media (audio, video, timing) — with a short pause between each batch to stay within HuggingFace's free-tier rate limits.

---

## Dataset Format

The dataset follows a partitioned Parquet layout:

```
/data/maestro/
├── metadata/
│   ├── trials.csv              # Trial metadata (trial_id, attended_speaker, kind)
│   ├── bad_channels.csv        # Per-subject bad EEG channel list
│   ├── audio_layout.json       # Speaker-to-file mapping and azimuth angles
│   └── eeg_channels.json       # Channel names and montage
├── data/
│   ├── eeg/subject=S01/trial=eval_001.parquet    # Columns: t, ch_Fp1, ..., ch_O2
│   ├── gaze/subject=S01/trial=eval_001.parquet   # Columns: t, gaze2d_x/y, gaze3d_x/y/z, L/R_pupil, ...
│   └── imu/subject=S01/trial=eval_001.parquet    # Columns: t, ax, ay, az, gx, gy, gz
└── media/
    ├── audio/<trial_id>/speaker{N}_dev{D}_{L|R}_spkid{ID}.flac
    ├── video/subject=S01/eval_001.mp4
    └── timing/subject=S01/trial=eval_001.json    # Unified sync timestamps
```

---

## Usage

All scripts take `--local_path` as the dataset root. An optional `--cache_dir` caches preprocessed video features so subsequent runs load instantly.

### T1 — Four-class attended source decoding (pooled)

```bash
# EEG only
python scripts/train_pooled.py --local_path maestro-data --mode eeg

# Full multimodal
python scripts/train_pooled.py --local_path maestro-data --mode eeg_vgi \
                       --cache_dir /cache
```

### T1 — Leave-one-subject-out

```bash
python scripts/train_loso.py --local_path maestro-data --mode eeg
python scripts/train_loso.py --local_path maestro-data --mode eeg_vgi --cache_dir /cache
```

### T2 — Attended hemisphere decoding

```bash
python scripts/train_hemisphere.py --local_path maestro-data --mode eeg
python scripts/train_hemisphere.py --local_path maestro-data --mode eeg_vgi --cache_dir /cache
```

### T3 — Attended eccentricity decoding

```bash
python scripts/train_eccentricity.py --local_path maestro-data --mode eeg
python scripts/train_eccentricity.py --local_path maestro-data --mode eeg_vgi --cache_dir /cache
```

### T4 — Envelope reconstruction

```bash
python scripts/train_reconstruction.py --local_path maestro-data --mode eeg
python scripts/train_reconstruction.py --local_path maestro-data --mode eeg_vgi --cache_dir /cache
```

### Supported modes

| Mode | Input |
|---|---|
| `eeg` | EEG only |
| `gaze` | Gaze only |
| `imu` | IMU only |
| `video` | Optical flow only |
| `gi` | Gaze + IMU |
| `eeg_gaze` | EEG + Gaze |
| `eeg_video` | EEG + Video |
| `eeg_vg` | EEG + Video + Gaze |
| `eeg_vgi` | EEG + Video + Gaze + IMU |

---

## Preprocessing

All modalities are resampled to 64 Hz and processed as non-overlapping 30-second windows (1,920 samples). Missing samples in gaze and IMU are handled per channel by dropping invalid samples before interpolation.

| Modality | Pipeline |
|---|---|
| EEG | Bandpass 1–40 Hz (4th-order Butterworth) → common average reference → downsample 500→64 Hz |
| Audio | Hilbert envelope → low-pass 20 Hz → downsample 16000→64 Hz → z-score |
| Gaze | Per-channel NaN drop → linear interpolation to 64 Hz grid → low-pass 10 Hz → z-score |
| IMU | Per-channel NaN drop → linear interpolation → resample to 64 Hz → low-pass 20 Hz → z-score |
| Video | Farneback dense optical flow on 160×90 frames → 4 statistics → resample 25→64 Hz → z-score |

Synchronisation uses the unified timing JSON (`media/timing/`), with only speech devices (IDs 3 and 5) used for sync alignment.

---

## Baseline Models

### Classification (T1, T2, T3)

A multi-encoder causal dilated convolutional network. Each modality is processed by a dedicated encoder, fused via concatenation and linear projection, and compared against speaker audio envelope embeddings via cosine similarity.

| Encoder | Layers | Receptive field |
|---|---|---|
| EEG | 7 + 1×1 spatial conv | ~34s |
| Audio | 7 (shared weights) | ~34s |
| Gaze | 6 | ~11.4s |
| IMU | 6 | ~11.4s |
| Video | 4 | ~1.3s |

Training: Adam lr=1e⁻⁴, label smoothing 0.1, gradient clipping 1.0, early stopping patience 10.

### Reconstruction (T4)

A linear backward model — a single causal Conv1d (kernel=32 samples, 0.5s) applied across all input channels jointly. Trained by minimising negative Pearson correlation.

---

## Citation

If you use MAESTRO in your research, please cite:

```bibtex
@article{hassan2025maestro,
  title   = {{MAESTRO}: A Multimodal Auditory-attention Egocentric Speech-TRacking Open Corpus},
  author  = {Hassan, K M Naimul and Alavi, Seyed Ali and Williamson, Donald},
  journal = {TBD},
  year    = {2026}
}
```

---

## License

[![License: CC BY-SA 4.0](https://img.shields.io/badge/License-CC%20BY--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/4.0/)

`SPDX-License-Identifier: CC-BY-SA-4.0`

This code and dataset are released under the [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/). You are free to share and adapt the material for any purpose, provided you give appropriate credit and distribute any derivative works under the same license.
