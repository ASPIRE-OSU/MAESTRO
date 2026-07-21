# MAESTRO Benchmark

Official benchmark code for the **MAESTRO** dataset — the Multimodal Auditory-attention Egocentric Speech-TRacking Open corpus. This repository contains the preprocessing pipeline, baseline models, training scripts, and pre-computed results for all four benchmark tasks defined in the accompanying paper.

> **Dataset:** [HuggingFace — aspire-osu/maestro-eeg-dataset](https://huggingface.co/datasets/aspire-osu/maestro-eeg-dataset)

---

![MAESTRO experimental setup](media/setup.png)

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
| **T1** | Four-class attended source decoding (pooled and leave-one-subject-out) | 25% |
| **T2** | Attended hemisphere decoding (left vs right) | 50% |
| **T3** | Attended eccentricity decoding (inner vs outer) | 50% |
| **T4** | Attended envelope reconstruction (Pearson r) | — |

---

## Results

All results use a 30-second decision window, are averaged across five stratified folds (T1 pooled, T2, T3, T4) or across 16 held-out subjects (T1 LOSO), and are reported as mean ± standard deviation. Multimodal combinations are produced via **late fusion**: independently-trained single-modality models combined by a lightweight learned softmax-weighted combiner, evaluated fold-matched against the single-modality checkpoints with no leakage. "Best fusion" denotes the best-performing multi-modality combination among all 11 possible (6 pairs, 4 triples, 1 full combination); it is not always the full four-modality combination.

### T1 — Four-class attended source decoding, pooled

| Mode | Accuracy |
|---|---|
| EEG only | 58.6% ± 5.7% |
| Gaze only | 62.0% ± 5.1% |
| IMU only | 56.8% ± 6.8% |
| Video only | 59.1% ± 7.1% |
| Best fusion (Gaze+IMU+Video) | 62.7% ± 6.5% |
| Full combination (EEG+Gaze+IMU+Video) | 61.9% ± 6.1% |
| Chance | 25.0% |

### T1 — Four-class attended source decoding, leave-one-subject-out (LOSO)

| Mode | Accuracy |
|---|---|
| EEG only | 56.9% ± 5.6% |
| Gaze only | 60.2% ± 5.6% |
| IMU only | 60.2% ± 5.5% |
| Video only | 58.4% ± 4.2% |
| Best fusion (Gaze+IMU) | 59.9% ± 5.2% |
| Chance | 25.0% |

### T2 — Attended hemisphere decoding

| Mode | Accuracy |
|---|---|
| EEG only | 78.1% ± 3.2% |
| Gaze only | 78.5% ± 1.5% |
| IMU only | 78.7% ± 1.6% |
| Video only | 76.4% ± 0.8% |
| Best fusion (EEG+Gaze) | 79.0% ± 4.2% |
| Chance | 50.0% |

### T3 — Attended eccentricity decoding

| Mode | Accuracy |
|---|---|
| EEG only | 70.5% ± 1.1% |
| Gaze only | 68.2% ± 1.7% |
| IMU only | 68.3% ± 3.1% |
| Video only | 70.5% ± 1.8% |
| Best fusion (EEG+IMU+Video) | 70.5% ± 2.7% |
| Chance | 50.0% |

### T4 — Envelope reconstruction

| Mode | Pearson r |
|---|---|
| EEG only | 0.0030 ± 0.0025 |
| Gaze only | 0.0000 ± 0.0000 |
| IMU only | 0.0131 ± 0.0070 |
| Video only | 0.0015 ± 0.0016 |
| Best combination (Gaze+IMU+Video) | 0.0196 ± 0.0028 |
| Full combination (EEG+Gaze+IMU+Video) | 0.0192 ± 0.0022 |

Full per-mode results for all 15 modality combinations, all four single-modality checkpoints (5 folds each, or 16 subjects for LOSO), and the pairwise error-complementarity and SNR-stratified analyses are provided as JSON files under `results/`.

---

## Repository Structure

```
MAESTRO/
├── scripts/
│   ├── dataloader.py               # Preprocessing, sync, windowing, PyTorch Dataset, mode registry
│   ├── model_classification.py     # Multi-encoder dilated conv network, fixed-concatenation fusion (T1)
│   ├── model_spatial.py            # Binary variant for T2 and T3
│   ├── model_reconstruction.py     # Linear backward model + Pearson loss (T4)
│   ├── late_fusion.py              # Late-fusion combiner (independent single-modality models + learned softmax weights)
│   ├── train_pooled.py             # T1 — pooled 5-fold CV
│   ├── train_loso_hot.py           # T1 — leave-one-subject-out, with combined subject + trial-content hold-out
│   ├── train_hemisphere.py         # T2 — hemisphere decoding
│   ├── train_eccentricity.py       # T3 — eccentricity decoding
│   ├── train_reconstruction.py     # T4 — envelope reconstruction, single mode
│   ├── train_rec_all.py            # T4 — sweeps all 15 modality combinations in one run
│   ├── analyze_snr.py              # SNR-stratified accuracy analysis (T1 pooled), quantile-binned
│   ├── analyze_error.py            # Pairwise error-complementarity analysis between single modalities (T1 pooled)
│   └── dl_maestro.py               # Dataset download script with rate limit handling (added in the camera-ready release)
└── results/
    ├── results_pooled/             # T1 pooled — checkpoints (5 folds × 4 single modalities) + result JSONs
    ├── results_loso/               # T1 LOSO — checkpoints (16 subjects × 4 single modalities) + result JSONs
    ├── results_hemisphere/         # T2 — checkpoints + result JSONs
    ├── results_eccentricity/       # T3 — checkpoints + result JSONs
    ├── results_reconstruction/     # T4 — checkpoints (5 folds × 15 modality combinations) + result JSONs
    ├── results_late_fusion/        # Late-fusion results for all 11 multi-modality combinations, per task
    ├── results_snr/                # Output of analyze_snr.py — SNR-stratified accuracy (T1 pooled)
    └── results_complementary/      # Output of analyze_error.py — pairwise error-complementarity (T1 pooled)
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
│   ├── trials.csv              # Trial metadata (trial_id, attended_speaker, kind, snr_db, ...)
│   ├── bad_channels.csv        # Per-subject bad EEG channel list
│   ├── audio_layout.json       # Speaker-to-file mapping and azimuth angles
│   └── eeg_channels.json       # Channel names and montage
├── data/
│   ├── eeg/subject=S01/trial=eval_001.parquet    # Columns: t_sec, sample_idx, ch_Fp1, ..., ch_O2
│   ├── gaze/subject=S01/trial=eval_001.parquet   # Columns: t, gaze2d_x/y, gaze3d_x/y/z, L/R_pupil, ...
│   └── imu/subject=S01/trial=eval_001.parquet    # Columns: t, ax, ay, az, gx, gy, gz
└── media/
    ├── audio/<trial_id>/speaker{N}_dev{D}_{L|R}_spkid{ID}.flac
    ├── video/subject=S01/eval_001.mp4
    └── timing/subject=S01/trial=eval_001.json    # Unified sync timestamps (anchor_unix, end_unix, per-stream offsets)
```

---

## Usage

All scripts take `--local_path` as the dataset root and `--mode` to select the input modality combination. An optional `--cache_dir` caches preprocessed EEG/video/gaze/IMU features so subsequent runs load instantly.

### T1 — Four-class attended source decoding (pooled)

```bash
# Single modality
python scripts/train_pooled.py --local_path maestro-data --mode eeg

# Multimodal combination
python scripts/train_pooled.py --local_path maestro-data --mode eeg_gaze_imu_video \
                       --cache_dir /cache
```

### T1 — Leave-one-subject-out (LOSO)

```bash
python scripts/train_loso_hot.py --local_path maestro-data --mode eeg
python scripts/train_loso_hot.py --local_path maestro-data --mode eeg_gaze_imu_video --cache_dir /cache
```

### T2 — Attended hemisphere decoding

```bash
python scripts/train_hemisphere.py --local_path maestro-data --mode eeg
python scripts/train_hemisphere.py --local_path maestro-data --mode eeg_gaze_imu_video --cache_dir /cache
```

### T3 — Attended eccentricity decoding

```bash
python scripts/train_eccentricity.py --local_path maestro-data --mode eeg
python scripts/train_eccentricity.py --local_path maestro-data --mode eeg_gaze_imu_video --cache_dir /cache
```

### T4 — Envelope reconstruction

```bash
# Single mode
python scripts/train_reconstruction.py --local_path maestro-data --mode eeg

# All 15 modality combinations in one run
python scripts/train_rec_all.py --local_path maestro-data --cache_dir /cache
```

### Late fusion (multi-modality, all tasks)

Multimodal results are produced by combining independently-trained single-modality checkpoints via a learned softmax-weighted combiner, rather than training a single model end-to-end on concatenated inputs. This requires the four single-modality checkpoints for a task to already exist (from the commands above):

```bash
python scripts/late_fusion.py --task pooled --mode eeg_gaze_imu_video \
    --ckpt_dir results/results_pooled --local_path maestro-data --cache_dir /cache --combine learned

# Sweep all 11 multi-modality combinations, plus fold in the 4 existing single-modality results
python scripts/late_fusion.py --task pooled --mode all \
    --ckpt_dir results/results_pooled --local_path maestro-data --cache_dir /cache --combine learned
```

`--task` accepts `pooled`, `loso`, `hemisphere`, or `eccentricity`.

### SNR-stratified and error-complementarity analysis (T1 pooled)

Both scripts reuse the already-trained single-modality (and, for SNR, late-fusion) checkpoints — no retraining required — evaluating fold-matched with no leakage.

```bash
# Accuracy vs. SNR, quantile-binned to keep bin sizes balanced across the SNR distribution
python scripts/analyze_snr.py --local_path maestro-data --cache_dir /cache \
    --ckpt_dir results/results_pooled --results results/results_snr

# Pairwise agreement/disagreement between single-modality models, with McNemar's test
python scripts/analyze_error.py --local_path maestro-data --cache_dir /cache \
    --ckpt_dir results/results_pooled --results results/results_complementary
```

### Supported modes

All 15 non-empty combinations of the four modalities are supported, in a canonical `eeg_gaze_imu_video`-style naming scheme. Three short legacy aliases are also accepted for backward compatibility.

| Mode | Input |
|---|---|
| `eeg`, `gaze`, `imu`, `video` | Single modality (4) |
| `eeg_gaze`, `eeg_imu`, `eeg_video`, `gaze_imu`, `gaze_video`, `imu_video` | Pairs (6) |
| `eeg_gaze_imu`, `eeg_gaze_video`, `eeg_imu_video`, `gaze_imu_video` | Triples (4) |
| `eeg_gaze_imu_video` | Full combination (1) |
| `gi` → `gaze_imu`, `eeg_vg` → `eeg_gaze_video`, `eeg_vgi` → `eeg_gaze_imu_video` | Legacy aliases |

---

## Preprocessing

All modalities are resampled to 64 Hz and processed as non-overlapping 30-second windows (1,920 samples). Missing samples in gaze and IMU are handled per channel by dropping invalid samples before interpolation.

| Modality | Pipeline |
|---|---|
| EEG | 60 Hz notch → bandpass 1–40 Hz (4th-order Butterworth) → bad-channel detection → mastoid-preferred reference (average of M1/M2 if both are good, otherwise full 32-channel average) → spherical-spline interpolation of bad channels → per-channel z-score → downsample 500→64 Hz |
| Audio | Hilbert envelope → low-pass 20 Hz (4th-order Butterworth) → downsample 16000→64 Hz → z-score |
| Gaze | Per-channel NaN drop → linear interpolation to 64 Hz grid → low-pass 10 Hz (4th-order Butterworth) → z-score |
| IMU | Per-channel NaN drop → linear interpolation to native sampling rate → resample to 64 Hz → low-pass 20 Hz (4th-order Butterworth) → z-score |
| Video | Grayscale conversion → Farneback dense optical flow on 160×90 frames → 4 statistics (mean/std flow magnitude, mean horizontal/vertical flow) → resample to 64 Hz → z-score |

Synchronisation uses the unified timing JSON (`media/timing/`): `align.anchor_unix`/`align.end_unix` define the shared alignment window across all modalities, with EEG masked per-sample from its own recorded timestamps, gaze/IMU each using their own per-stream clock offset, and all four speakers' audio sharing one reference time regardless of recording device.

---

## Baseline Models

### Classification (T1, T2, T3)

A multi-encoder causal dilated convolutional network. Each modality is processed by a dedicated encoder, projected to a shared embedding width, fused via concatenation and a linear projection, and compared against speaker audio envelope embeddings via cosine similarity.

| Encoder | Layers | Receptive field |
|---|---|---|
| EEG | 7 + 1×1 spatial conv | ~34s |
| Audio | 7 (shared weights) | ~34s |
| Gaze | 6 | ~11.4s |
| IMU | 6 | ~11.4s |
| Video | 4 | ~1.3s |

All encoders use a uniform 16-dimensional embedding width. Training: Adam lr=1e⁻⁴, label smoothing 0.1, gradient clipping 1.0, early stopping patience 10. Each cross-validation fold is seeded independently (including a mode-dependent offset) so that different modality combinations sharing a fold never receive identical model initialization.

### Reconstruction (T4)

A linear backward model — a single causal Conv1d (kernel=32 samples, 0.5s) applied across all input channels jointly. Trained by minimising negative Pearson correlation, with L2 weight decay approximating ridge regression.

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

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

`SPDX-License-Identifier: CC-BY-NC-SA-4.0`

This code and dataset are released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)](https://creativecommons.org/licenses/by-nc-sa/4.0/). You are free to share and adapt the material for non-commercial purposes, provided you give appropriate credit and distribute any derivative works under the same license.
