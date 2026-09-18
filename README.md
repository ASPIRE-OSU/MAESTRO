# MAESTRO Benchmark

Official benchmark code for the **MAESTRO** dataset — the Multimodal Auditory-attention Egocentric Speech-TRacking Open corpus. This repository contains the preprocessing pipeline, the baseline model, training scripts, and pre-computed results for the attended-speaker decoding benchmark defined in the accompanying paper.

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
| Audio (4 speakers + 2 noise) | 6 loudspeakers | 16 kHz |

---

## Benchmark Task

The benchmark is **four-class attended-speaker decoding**: given a decision window, identify which of the four simultaneously presented speakers the listener is attending to (chance 25%). It is evaluated at **five decision window sizes** (5, 10, 15, 20, 30 s) under **both** official split protocols:

- **within-subject** — 5-fold cross-validation, folds defined by trial content and pooled across all 16 subjects. Every subject appears in both train and test, on disjoint trial content (content generalization).
- **leave-one-subject-out (LOSO)** — 16 folds, one held-out subject per fold, trained on the other 15 (subject generalization).

Both protocols read the dataset's own official split definitions from `splits/{within,loso}/fold_*.json` rather than reconstructing splits internally — see [Dataset Format](#dataset-format).

> The repository also contains code and results for two auxiliary spatial tasks — hemisphere (left/right) and eccentricity (inner/outer) decoding — under `scripts/train_hemisphere.py`, `scripts/train_eccentricity.py`, and the `spat_*` result folders. These are not part of the paper's benchmark and are provided for completeness only.

---

## Results

Accuracy (mean ± std across folds — 5 folds for within-subject, 16 held-out subjects for LOSO) for the four single modalities and the best-performing multimodal combination at each window size. **"Best multimodal"** is the highest-accuracy combination among all 11 multi-modality combinations (6 pairs, 4 triples, 1 full four-way) at that split and window, named in parentheses; it is not always the full four-modality combination and it varies across window sizes.

All results are produced by a single model trained end-to-end (`train_aad.py`); there is no separate late-fusion stage. Full per-mode results for all 15 modality combinations, paired significance testing against EEG-only, the permutation null and contribution, and the SNR-stratified analysis are provided as JSON under `results/`.

### Within-subject (chance 25%)

| Window | EEG | Gaze | IMU | Video | Best multimodal |
|---|---|---|---|---|---|
| 5s | 43.63% ± 1.76% | 37.13% ± 1.16% | 38.21% ± 2.80% | 43.71% ± 4.20% | 56.64% ± 3.28% (EEG+Gaze+IMU+Video) |
| 10s | 47.57% ± 4.08% | 37.58% ± 1.75% | 38.75% ± 2.13% | 42.59% ± 4.90% | 58.45% ± 3.86% (EEG+Gaze+Video) |
| 15s | 53.17% ± 1.66% | 39.98% ± 2.10% | 38.84% ± 2.44% | 44.33% ± 3.86% | 61.20% ± 1.83% (EEG+Gaze+IMU+Video) |
| 20s | 53.62% ± 2.67% | 37.95% ± 2.38% | 34.85% ± 4.16% | 43.16% ± 2.44% | 62.80% ± 2.72% (EEG+Gaze+IMU+Video) |
| 30s | 59.06% ± 2.04% | 36.04% ± 2.79% | 35.41% ± 4.14% | 43.37% ± 6.63% | 69.83% ± 2.33% (EEG+Gaze+Video) |

### Leave-one-subject-out (chance 25%)

| Window | EEG | Gaze | IMU | Video | Best multimodal |
|---|---|---|---|---|---|
| 5s | 43.64% ± 3.18% | 33.99% ± 8.05% | 37.07% ± 10.86% | 39.57% ± 12.71% | 54.41% ± 11.83% (EEG+Gaze+IMU+Video) |
| 10s | 50.19% ± 6.98% | 34.95% ± 9.38% | 34.77% ± 9.46% | 43.56% ± 14.66% | 57.33% ± 10.37% (EEG+Gaze+Video) |
| 15s | 52.40% ± 7.02% | 36.08% ± 9.09% | 38.48% ± 13.53% | 41.35% ± 13.65% | 59.59% ± 11.15% (EEG+IMU+Video) |
| 20s | 55.31% ± 9.92% | 36.87% ± 8.98% | 32.16% ± 11.29% | 41.56% ± 17.61% | 61.29% ± 9.32% (EEG+IMU+Video) |
| 30s | 61.88% ± 11.71% | 35.46% ± 9.44% | 35.46% ± 12.69% | 44.69% ± 14.52% | 67.45% ± 18.58% (EEG+Gaze+IMU+Video) |

---

## Repository Structure

```
MAESTRO/
├── scripts/
│   ├── dataloader.py               # Preprocessing, sync, windowing, candidate construction, official-split loading, mode registry
│   ├── model_classification.py     # Multi-encoder dilated conv network (4-class)
│   ├── model_spatial.py            # Binary variant for the auxiliary spatial tasks
│   ├── losses.py                   # Training objective (cross-entropy + auxiliary, contrastive, hinge, anti-collapse, adversarial terms)
│   ├── evaluation.py               # Permutation battery: null, contribution, stratified nulls, diagnostics
│   ├── train_aad.py                # Four-class attended-speaker training (within or loso)
│   ├── train_hemisphere.py         # Auxiliary — hemisphere (LOSO)
│   ├── train_eccentricity.py       # Auxiliary — eccentricity (LOSO)
│   ├── analyze_snr.py              # SNR-stratified accuracy from saved checkpoints (no retraining)
│   ├── collect_results.py          # Aggregate result JSONs
│   ├── make_paper_tables.py        # Regenerate the paper's tables from results/
│   ├── make_paper_figures.py       # Regenerate the paper's figures from results/
│   └── dl_maestro.py               # Dataset download with rate-limit handling and HF token auth
└── results/
    ├── res_within_w{5,10,15,20,30}_h{2.5,5,7.5,10,15}_qmatch/   # Four-class within-subject — result JSONs, one folder per window
    ├── res_loso_w{...}_qmatch/                                  # Four-class LOSO
    ├── spat_{within,loso}_w{...}_qmatch/                        # Auxiliary hemisphere / eccentricity
    ├── significance/                                           # Paired t-test outputs vs EEG-only
    ├── snr/                                                     # SNR-stratified analysis
    ├── paper_tables/  paper_figures/                           # Regenerated tables and figures
    └── ablate_nohinge_*/                                       # Ablation: objective without the hinge terms
```


---

## Installation

```bash
git clone https://github.com/ASPIRE-OSU/MAESTRO
cd MAESTRO
pip install -r requirements.txt
```
> **Note:** PyTorch must be installed separately to match your CUDA version. See [pytorch.org](https://pytorch.org/get-started/locally/) for the correct install command.

## Downloading the Dataset

The dataset is publicly available on HuggingFace. Use the provided download script, which handles rate limiting automatically:

```bash
export HF_TOKEN=hf_your_token_here
python scripts/dl_maestro.py --local_dir maestro-data
```

To download specific subjects only:

```bash
python scripts/dl_maestro.py --local_dir maestro-data --subjects 1 2 3
```

If no token is provided (neither `--token` nor `HF_TOKEN`), the script prints a warning and proceeds anyway, which is fine for the public dataset but required for any private/gated access.

---

## Dataset Format

The dataset follows a partitioned Parquet layout:

```
/data/maestro/
├── splits/
│   ├── within/fold_0.json ... fold_4.json    # 5-fold content-based split, pooled across all 16 subjects
│   └── loso/fold_00.json ... fold_15.json    # 16-fold subject-based split, one held-out subject each
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
    └── timing/subject=S01/trial=eval_001.json    # Unified sync timestamps
```

All scripts read the split files directly via `dataloader.load_official_splits()` as the authoritative source of train/test partitioning — splits are never reconstructed internally.

---

## Usage

All scripts take `--local_path` (dataset root), `--split_setting` (`within` or `loso`), and `--window_sec`/`--hop_sec` to select the decision window. `--cache_dir` caches preprocessed features; `--dataset_cache` additionally memoises the assembled dataset, worth setting when sweeping windows since the per-trial cache does not store audio envelopes.

### Training

A single model is trained end-to-end per configuration. `--mode` accepts any of the 15 non-empty modality combinations directly:

```bash
# EEG alone, within-subject, 30 s window
python scripts/train_aad.py --local_path maestro-data --mode eeg --split_setting within --window_sec 30 --hop_sec 15

# Full four-modality model, LOSO, 10 s window
python scripts/train_aad.py --local_path maestro-data --mode eeg_gaze_imu_video --split_setting loso --window_sec 10 --hop_sec 5
```

Each run writes a result JSON (accuracy, permutation null, contribution, and the diagnostic battery) to `--results`. Multimodal configurations are trained as one jointly-fused model with modality dropout — there is no separate combiner over frozen single-modality checkpoints.

### SNR-stratified analysis

Reuses the saved per-fold checkpoints — no retraining:

```bash
python scripts/analyze_snr.py --mode eeg --split_setting loso --local_path maestro-data \
    --window_sec 30 --hop_sec 15 --model_root results/res --results results/snr
```

### Supported modes

All 15 non-empty combinations of the four modalities, in canonical `eeg_gaze_imu_video`-style naming. Three short aliases are also accepted.

| Mode | Input |
|---|---|
| `eeg`, `gaze`, `imu`, `video` | Single modality (4) |
| `eeg_gaze`, `eeg_imu`, `eeg_video`, `gaze_imu` (alias `gi`), `gaze_video`, `imu_video` | Pairs (6) |
| `eeg_gaze_imu`, `eeg_gaze_video` (alias `eeg_vg`), `eeg_imu_video`, `gaze_imu_video` | Triples (4) |
| `eeg_gaze_imu_video` (alias `eeg_vgi`) | Full combination (1) |

---

## Preprocessing

All modalities are resampled to 64 Hz and z-scored per channel, per trial.

| Modality | Pipeline |
|---|---|
| EEG | 60 Hz notch → bandpass 1–40 Hz (4th-order Butterworth, filtfilt) → bad-channel detection (flat: std < 1e-9; saturated: ≥10% of samples at the ADC clip; or outlier first-difference variance via a MAD threshold) → mastoid-preferred reference (falls back to full-channel average) → spherical-spline interpolation of bad channels via MNE if installed → per-channel z-score → downsample 500→64 Hz |
| Audio | Per-device playback-timestamp alignment → Hilbert envelope → low-pass 20 Hz (4th-order Butterworth) → downsample 16000→64 Hz → per-trial z-score. Envelope extraction is linear and the z-score is affine-invariant, so a level difference between talkers is removed exactly; the residual **shape** difference between attended and competing envelopes is removed by candidate construction (below). |
| Gaze | Per-channel NaN drop → linear interpolation to 64 Hz grid → low-pass 10 Hz → z-score |
| IMU | Per-channel NaN drop → linear interpolation to native rate → resample to 64 Hz → low-pass 20 Hz → z-score |
| Video | Downsample frames to 160×90 → grayscale → Farneback dense optical flow → 4 statistics per frame pair (mean/std flow magnitude, mean horizontal/vertical flow) → resample native fps→64 Hz → z-score |

**Candidate construction.** Within each decision window the four speaker envelopes are distribution-matched by histogram equalization (`quantile_match_candidates`): each envelope is sorted, the sorted values are averaged across the four, and each sample is replaced by the shared value at its own rank. All four then hold an identical multiset of values, so any statistic computed from amplitudes alone is equal by construction and only the temporal ordering distinguishes them. The four envelopes are also assigned to slots in a random order per window, so a slot's position carries no information about the label.

Synchronisation uses the unified timing JSON (`media/timing/`): EEG filtering runs on the full unmasked trial recording before windowing, so filter edge transients fall outside the analysis window; all streams are then aligned to a shared anchor/end timestamp per trial, and each speaker's audio is additionally aligned by its own per-device playback-start timestamp.

---

## Baseline Model

A multi-encoder dilated convolutional network (`model_classification.py`), based on Accou et al. Each active modality is processed by a dedicated 5-layer encoder (kernel 3, dilations 2⁰…2⁴, receptive field 63 samples ≈ 0.98 s) into a 16-dimensional embedding.

| Encoder | Detail |
|---|---|
| EEG | 5 layers + a 1×1 spatial convolution (8 filters) mixing the 32 channels |
| Audio | 5 layers, weights shared across the four envelope streams |
| Gaze / IMU / Video | 5 layers each |

Convolutions are **centered** (not causal), GroupNorm follows every convolution, and the final layer is linear so embeddings may be negative.

**Two scores are combined.** The EEG embedding is compared with each speaker envelope by **time-centered Pearson correlation**, giving one score per slot; centering ensures a time-constant embedding scores zero against every candidate, so the collapsed solution is pinned at chance. Separately, every modality embedding is pooled over time and passed to a classifier over the four speaker positions with no audio input. The two scores are summed and the highest-scoring slot is the prediction.

**Training.** AdamW (lr 1e⁻³, weight decay 1e⁻⁴), gradient clipping 1.0, label smoothing 0.1, batches of 32 windows drawn from one participant at a time. Modalities are fused in a single end-to-end model with modality dropout (p=0.3, never all at once, disabled at evaluation). The objective adds five terms to the cross-entropy — per-modality auxiliary, within-participant contrastive, permutation/zeros hinges, anti-collapse, and an audio-only adversary behind a gradient-reversal layer (see `losses.py`). Checkpoints are selected on the validation contribution. The learning rate is halved after five epochs without improvement (floor 1e⁻⁶); training stops after twelve without improvement, up to 50 epochs.

---

## Citation

If you use MAESTRO in your research, please cite:

```bibtex
@article{hassan2026maestro,
  title   = {{MAESTRO}: A Multimodal Auditory-attention Egocentric Speech-TRacking Open Corpus},
  author  = {Hassan, K M Naimul and Alavi, Ali and Williamson, Donald S.},
  journal = {IEEE Transactions on Audio, Speech, and Language Processing},
  year    = {2026}
}
```

---

## License

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

`SPDX-License-Identifier: CC-BY-NC-SA-4.0`

Released under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

