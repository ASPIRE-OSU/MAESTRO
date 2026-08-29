"""
make_paper_figures.py
---------------------
Regenerates the two benchmark figures from the fixed results.

  fig_folds.png   per-fold (= per-held-out-listener) contribution under LOSO,
                  for EEG alone and for the best EEG-containing multimodal
                  configuration, at every decision window.  This is the
                  fold-level form of the permutation test: each bar is one
                  listener's accuracy minus that listener's own permuted
                  accuracy, so a bar above zero means that listener's decision
                  depended on their recording.

  fig_snr.png     accuracy AND the permutation null per SNR bin.  The previous
                  version of this figure plotted accuracy alone, which cannot
                  distinguish better neural decoding at high SNR from an easier
                  acoustic shortcut at high SNR.

Neither figure had a generating script in the repository before; both are now
reproducible from the result JSONs alone.
"""

import argparse
import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WINDOWS = [5.0, 10.0, 15.0, 20.0, 30.0]
FULL = "eeg_gaze_imu_video"
EEG_MULTI = ["eeg_gaze", "eeg_imu", "eeg_video", "eeg_gaze_imu",
             "eeg_gaze_video", "eeg_imu_video", "eeg_gaze_imu_video"]
LABEL = {"eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU", "eeg_video": "EEG+Video",
         "eeg_gaze_imu": "EEG+Gaze+IMU", "eeg_gaze_video": "EEG+Gaze+Video",
         "eeg_imu_video": "EEG+IMU+Video",
         "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video"}


def load_t1(root, split="loso"):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "res_*", "results_*.json"))):
        m = re.search(r"results_(.+)_(within|loso)\.json$", os.path.basename(p))
        if not m or m.group(2) != split:
            continue
        d = json.load(open(p))
        runs[(m.group(1), float(d["window_sec"]))] = d
    return runs


def folds_vec(d, key="contribution", n=16):
    v = np.full(n, np.nan)
    for fid, fd in d["folds"].items():
        i = int(fid)
        if 0 <= i < n and key in fd:
            v[i] = fd[key]
    return v


# ── figure 1: per-listener contribution ──────────────────────────────────────
# Form: a paired-range (dumbbell) chart, one row per held-out listener.  The job
# of these data is a PAIRED COMPARISON across listeners, not a time series and
# not a magnitude ranking, so the mark is a connector between two dots rather
# than a bar: the bar length would encode the contribution twice (position and
# area) while hiding the quantity that matters, which is the gap.  The per-cell
# means are already in the tables; what a table cannot show is the spread across
# listeners and the sign of each listener's change, which is what this figure is
# for.

C_EEG, C_ALL = "#2a78d6", "#eb6834"      # categorical slots 1 and 2; validated:
                                          # CVD dE 24.7 (protan), normal 33.6
INK, INK2, GRID = "#232321", "#63625c", "#dcdbd4"


def fig_folds(root, out):
    runs = load_t1(root, "loso")
    wins = [w for w in WINDOWS if ("eeg", w) in runs and (FULL, w) in runs]
    E = np.vstack([folds_vec(runs[("eeg", w)]) for w in wins])      # (W, 16)
    B = np.vstack([folds_vec(runs[(FULL, w)]) for w in wins])
    e_mu, b_mu = np.nanmean(E, 0), np.nanmean(B, 0)
    order = np.argsort(e_mu)                                        # weakest first

    fig = plt.figure(figsize=(7.16, 3.45), dpi=400)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.55, 1.0], wspace=0.28)
    axL, axR = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # ---- left: one row per listener, EEG dot -> four-modality dot ------------
    y = np.arange(16)
    axL.axvline(0, color=INK, lw=0.9, zorder=2)
    for r, s in enumerate(order):
        lo, hi = min(e_mu[s], b_mu[s]), max(e_mu[s], b_mu[s])
        axL.plot([lo, hi], [r, r], color=GRID, lw=2.0, solid_capstyle="round",
                 zorder=3)
    axL.scatter(e_mu[order], y, s=34, color=C_EEG, edgecolor="white", lw=0.9,
                zorder=5, label="EEG")
    axL.scatter(b_mu[order], y, s=34, color=C_ALL, edgecolor="white", lw=0.9,
                zorder=5, label="EEG+Gaze+IMU+Video")
    axL.set_yticks(y)
    axL.set_yticklabels([f"P{s+1}" for s in order], fontsize=6.8)
    axL.set_ylim(-0.8, 15.8)
    axL.set_xlabel("contribution (accuracy $-$ permuted accuracy),\n"
                   "mean over the five decision windows", fontsize=7.6)
    axL.tick_params(axis="x", labelsize=7)
    axL.tick_params(axis="y", length=0)
    axL.grid(axis="x", color=GRID, lw=0.5, zorder=0)
    axL.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        axL.spines[sp].set_visible(False)
    axL.spines["bottom"].set_color(INK2)
    handles, labels = axL.get_legend_handles_labels()
    axL.set_title("a   every participant, averaged over windows", fontsize=8,
                  loc="left", color=INK, pad=6)

    # ---- right: the same 16 listeners at each window -------------------------
    rng = np.random.default_rng(0)
    for k, w in enumerate(wins):
        for arr, c, dx in ((E[k], C_EEG, -0.16), (B[k], C_ALL, 0.16)):
            jit = (rng.random(len(arr)) - 0.5) * 0.13
            axR.scatter(np.full(len(arr), k + dx) + jit, arr, s=11, color=c,
                        alpha=0.75, edgecolor="none", zorder=3)
            axR.plot([k + dx - 0.10, k + dx + 0.10],
                     [np.nanmedian(arr)] * 2, color=INK, lw=1.6, zorder=4)
    axR.axhline(0, color=INK, lw=0.9, zorder=2)
    axR.set_xticks(range(len(wins)))
    axR.set_xticklabels([f"{w:g}" for w in wins], fontsize=7)
    axR.set_xlabel("decision window (s)", fontsize=7.6)
    axR.set_ylabel("contribution", fontsize=7.6)
    axR.tick_params(axis="y", labelsize=7)
    axR.grid(axis="y", color=GRID, lw=0.5, zorder=0)
    axR.set_axisbelow(True)
    for sp in ("top", "right"):
        axR.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        axR.spines[sp].set_color(INK2)
    axR.set_title("b   each participant at each window", fontsize=8, loc="left",
                  color=INK, pad=6)
    axR.legend(handles, labels, fontsize=6.9, frameon=False, loc="upper left",
               handletextpad=0.35, borderpad=0.1, labelspacing=0.3)
    axR.text(0.985, 0.065, "every point above zero", transform=axR.transAxes,
             ha="right", va="bottom", fontsize=6.5, color=INK2, style="italic")

    npos = int((np.vstack([E, B]) > 0).sum())
    ntot = int(np.isfinite(np.vstack([E, B])).sum())
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print(f"wrote {out}  ({npos}/{ntot} listener-window points above zero; "
          f"series: EEG vs {FULL}, both fixed in advance)")


# ── figure 2: SNR bins, accuracy against its own null ────────────────────────

def fig_snr(root, out, mode="eeg", split="loso"):
    files = sorted(glob.glob(os.path.join(root, "snr", f"snr_{mode}_{split}_w*.json")))
    if not files:
        print("no SNR results yet; skipping fig_snr")
        return
    per_bin_acc, per_bin_null, labels = {}, {}, {}
    for p in files:
        d = json.load(open(p))
        for b, v in d["bins"].items():
            per_bin_acc.setdefault(int(b), []).append(v["accuracy"])
            per_bin_null.setdefault(int(b), []).append(v["null_mean"])
            labels[int(b)] = v["label"]
    bins = sorted(per_bin_acc)
    acc = np.array([np.mean(per_bin_acc[b]) for b in bins])
    accsd = np.array([np.std(per_bin_acc[b]) for b in bins])
    nul = np.array([np.mean(per_bin_null[b]) for b in bins])
    nulsd = np.array([np.std(per_bin_null[b]) for b in bins])

    fig, ax = plt.subplots(figsize=(4.0, 3.0), dpi=400)
    xs = np.arange(len(bins))
    ax.fill_between(xs, acc - accsd, acc + accsd, color="#4a7fb5", alpha=0.16,
                    lw=0)
    ax.fill_between(xs, nul - nulsd, nul + nulsd, color="#b0616a", alpha=0.14,
                    lw=0)
    ax.plot(xs, acc, "o-", color="#2f6096", lw=1.5, ms=4.5, label="accuracy")
    ax.plot(xs, nul, "s--", color="#a2454f", lw=1.3, ms=4,
            label="permuted (audio-only floor)")
    ax.axhline(0.25, color="#546e7a", lw=0.9, ls=":", label="chance")
    for i, (a, n) in enumerate(zip(acc, nul)):
        ax.annotate(f"{a-n:+.2f}", (xs[i], (a + n) / 2), fontsize=6.2,
                    ha="center", va="center", color="#37474f",
                    bbox=dict(fc="white", ec="none", pad=0.8, alpha=0.85))
    ax.set_xticks(xs)
    ax.set_xticklabels([labels[b] for b in bins], fontsize=6.4, rotation=12)
    ax.set_xlabel("SNR bin (equal-count quantiles)", fontsize=8)
    ax.set_ylabel("four-class accuracy", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_ylim(0.15, max(0.75, acc.max() + 0.12))
    ax.legend(fontsize=6.6, frameon=False, loc="upper left")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight", pad_inches=0.03, facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--out", default="paper_figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fig_folds(args.root, os.path.join(args.out, "fig_folds.png"))
    fig_snr(args.root, os.path.join(args.out, "fig_snr.png"))
