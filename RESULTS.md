# T1 — four-class attended-talker decoding

Produced by `scripts/train_aad.py` on branch `fixes`, 16 subjects, 50 epochs,
one-time test evaluation per fold, candidates distribution-matched (`qmatch`).

**Read every cell as three numbers, not one.** *Accuracy* is what the decoder
scores. *Permuted* is what it scores when the physiological recordings are
shuffled across test windows, each window keeping its own audio candidates and
its own label — i.e. the part obtainable from the audio alone. *Contribution* is
the difference, and is the only part attributable to the recording.

On the previous revision of this benchmark the permuted accuracy equalled the
real accuracy at every one of these cells, so the contribution was zero and the
reported number was entirely audio. Here the permuted accuracy sits at chance
(0.25–0.33), which is what the fix was for.

Regenerate with:

```bash
python scripts/collect_results.py <results_root> --markdown RESULTS.md
```

40 / 40 runs present. Chance = 0.2500. Candidates are distribution-matched (`qmatch`), so the audio-only floor is at chance; see the probe value in each run's JSON.


## Within-subject (intra, 5 folds)

### Accuracy

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | 0.4363 ± 0.018 | 0.4757 ± 0.041 | 0.5317 ± 0.017 | 0.4994 ± 0.047 | 0.5906 ± 0.020 |
| EEG+IMU | 0.4914 ± 0.025 | 0.5150 ± 0.040 | 0.5413 ± 0.041 | 0.4782 ± 0.083 | 0.6408 ± 0.027 |
| EEG+Video | 0.4990 ± 0.048 | 0.5270 ± 0.027 | 0.5469 ± 0.020 | 0.5481 ± 0.031 | 0.6256 ± 0.022 |
| EEG+Gaze | 0.5124 ± 0.039 | 0.5527 ± 0.018 | 0.5797 ± 0.016 | 0.4436 ± 0.098 | 0.6515 ± 0.038 |

### Accuracy under permuted recordings (audio-only floor)

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | 0.3251 ± 0.025 | 0.3163 ± 0.055 | 0.2751 ± 0.033 | 0.2852 ± 0.065 | 0.2710 ± 0.007 |
| EEG+IMU | 0.3281 ± 0.021 | 0.2777 ± 0.035 | 0.2604 ± 0.018 | 0.2597 ± 0.019 | 0.2643 ± 0.008 |
| EEG+Video | 0.3187 ± 0.038 | 0.2886 ± 0.019 | 0.2616 ± 0.011 | 0.2586 ± 0.013 | 0.2607 ± 0.011 |
| EEG+Gaze | 0.3079 ± 0.040 | 0.2817 ± 0.022 | 0.2623 ± 0.012 | 0.2532 ± 0.011 | 0.2647 ± 0.013 |

### Contribution of the recording (accuracy − permuted)

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | +0.1112 | +0.1595 | +0.2566 | +0.2142 | +0.3196 |
| EEG+IMU | +0.1633 | +0.2373 | +0.2809 | +0.2185 | +0.3766 |
| EEG+Video | +0.1803 | +0.2384 | +0.2852 | +0.2895 | +0.3649 |
| EEG+Gaze | +0.2044 | +0.2710 | +0.3174 | +0.1904 | +0.3867 |

## Leave-one-subject-out (16 folds)

### Accuracy

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | 0.4364 ± 0.032 | 0.5019 ± 0.070 | 0.5240 ± 0.070 | 0.4875 ± 0.110 | 0.6188 ± 0.117 |
| EEG+IMU | 0.4854 ± 0.063 | 0.5093 ± 0.078 | 0.5580 ± 0.087 | 0.4613 ± 0.157 | 0.6304 ± 0.134 |
| EEG+Video | 0.5088 ± 0.074 | 0.5419 ± 0.067 | 0.5625 ± 0.084 | 0.5531 ± 0.087 | 0.6344 ± 0.125 |
| EEG+Gaze | 0.5054 ± 0.075 | 0.5214 ± 0.102 | 0.5590 ± 0.086 | 0.5169 ± 0.131 | 0.6245 ± 0.162 |

### Accuracy under permuted recordings (audio-only floor)

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | 0.3252 ± 0.032 | 0.2927 ± 0.029 | 0.2618 ± 0.016 | 0.2630 ± 0.030 | 0.2559 ± 0.038 |
| EEG+IMU | 0.3129 ± 0.017 | 0.2813 ± 0.027 | 0.2544 ± 0.012 | 0.2536 ± 0.029 | 0.2659 ± 0.029 |
| EEG+Video | 0.3199 ± 0.020 | 0.2763 ± 0.024 | 0.2630 ± 0.024 | 0.2694 ± 0.039 | 0.2613 ± 0.033 |
| EEG+Gaze | 0.3118 ± 0.019 | 0.2699 ± 0.025 | 0.2593 ± 0.019 | 0.2605 ± 0.030 | 0.2694 ± 0.023 |

### Contribution of the recording (accuracy − permuted)

| Modalities | 5 s | 10 s | 15 s | 20 s | 30 s |
|---|---|---|---|---|---|
| EEG | +0.1112 | +0.2092 | +0.2621 | +0.2245 | +0.3628 |
| EEG+IMU | +0.1725 | +0.2280 | +0.3036 | +0.2077 | +0.3645 |
| EEG+Video | +0.1889 | +0.2656 | +0.2995 | +0.2837 | +0.3731 |
| EEG+Gaze | +0.1936 | +0.2515 | +0.2997 | +0.2564 | +0.3551 |

## Supporting controls, headline window (10 s)

| Modalities | Split | Zeros-input acc | Flip rate | Collapse | Within-trial contribution | p |
|---|---|---|---|---|---|---|
| EEG | within | 0.3485 | 0.571 | 0.439 | +0.1262 | 0.133 |
| EEG+IMU | within | 0.2797 | 0.700 | 0.310 | +0.1490 | 0.048 |
| EEG+Video | within | 0.3098 | 0.691 | 0.356 | +0.1455 | 0.048 |
| EEG+Gaze | within | 0.3518 | 0.701 | 0.327 | +0.1515 | 0.048 |
| EEG | loso | 0.3219 | 0.693 | 0.332 | +0.1717 | 0.054 |
| EEG+IMU | loso | 0.3059 | 0.701 | 0.290 | +0.1498 | 0.062 |
| EEG+Video | loso | 0.2756 | 0.696 | 0.324 | +0.1581 | 0.048 |
| EEG+Gaze | loso | 0.2887 | 0.705 | 0.308 | +0.1377 | 0.048 |

## Checkpoints

Please visit : [Checkpoints](https://github.com/ASPIRE-OSU/MAESTRO/releases/tag/weights-v1)

