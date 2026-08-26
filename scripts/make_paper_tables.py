"""
make_paper_tables.py
--------------------
Builds every table, per-fold export and figure input the dataset paper's
benchmark section needs, from the result JSONs written by `train_aad.py`,
`train_hemisphere.py`, `train_eccentricity.py` and `analyze_snr.py`.

Every cell carries three numbers, never one:

    accuracy      what the decoder scores
    permuted      what it scores when the physiological recordings are permuted
                  across test windows, each window keeping its own candidates
                  and its own label
    contribution  accuracy - permuted

and two fold-level statistics: how many folds have a positive contribution, and
a signed-rank test over folds.  An accuracy on its own does not distinguish a
decoder that uses the recording from one that reads the candidates; on the
previous revision of this benchmark the two were indistinguishable.

Outputs (into --out):
    table2_accuracy.tex        T1, 15 modes x (5 within + 5 loso), acc +/- sd
    table2_contribution.tex    T1, same geometry, contribution
    table2_null.tex            T1, same geometry, permutation null
    table3_accuracy.tex        T2/T3 LOSO, 15 modes x (5 hemi + 5 ecc)
    table3_contribution.tex
    table3_null.tex
    folds_t1.csv               per-fold accuracy/null/contribution, every cell
    folds_t2t3.csv             ditto
    snr_bins.csv               per-SNR-bin accuracy/null/contribution
    summary.md                 human-readable version of everything
    stats.json                 machine-readable, for cross-checking prose
"""

import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict

import math

import numpy as np

MODES = ["eeg", "gaze", "imu", "video",
         "eeg_gaze", "eeg_imu", "eeg_video",
         "gaze_imu", "gaze_video", "imu_video",
         "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video", "gaze_imu_video",
         "eeg_gaze_imu_video"]
LABEL = {"eeg": "EEG", "gaze": "Gaze", "imu": "IMU", "video": "Video",
         "eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU", "eeg_video": "EEG+Video",
         "gaze_imu": "Gaze+IMU", "gaze_video": "Gaze+Video",
         "imu_video": "IMU+Video", "eeg_gaze_imu": "EEG+Gaze+IMU",
         "eeg_gaze_video": "EEG+Gaze+Video", "eeg_imu_video": "EEG+IMU+Video",
         "gaze_imu_video": "Gaze+IMU+Video",
         "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video"}
# row groups printed with \midrule between them, as the paper does
GROUPS = [MODES[0:4], MODES[4:10], MODES[10:14], MODES[14:]]
WINDOWS = [5.0, 10.0, 15.0, 20.0, 30.0]
# Modes with no EEG have no coupling branch, so the candidate envelopes never
# enter the forward pass: those rows are a plain classification of the class
# index, not a match-mismatch decision.  Flagged in every output.
NO_AUDIO = [m for m in MODES if "eeg" not in m]


# ── signed-rank test over folds, no SciPy dependency ──────────────────────────

def wilcoxon_p(x):
    """Two-sided Wilcoxon signed-rank p for H0: median(x) = 0, normal
    approximation with tie and continuity correction.  Falls back to an exact
    sign test when n is too small for the approximation to mean anything."""
    x = np.asarray([v for v in x if v != 0.0], dtype=float)
    n = len(x)
    if n == 0:
        return 1.0
    if n < 6:                                   # exact two-sided sign test
        k = int(np.sum(x > 0))
        from math import comb
        tail = sum(comb(n, i) for i in range(min(k, n - k) + 1))
        return min(1.0, 2.0 * tail / (2 ** n))
    order = np.argsort(np.abs(x))
    ranks = np.empty(n, dtype=float)
    a = np.abs(x)[order]
    i = 0
    while i < n:                                # average ranks within ties
        j = i
        while j + 1 < n and a[j + 1] == a[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    r = np.empty(n, dtype=float)
    r[order] = ranks
    w = float(np.sum(r[x > 0]))
    mu = n * (n + 1) / 4.0
    _, counts = np.unique(a, return_counts=True)
    tie = float(np.sum(counts ** 3 - counts))
    sd = np.sqrt(n * (n + 1) * (2 * n + 1) / 24.0 - tie / 48.0)
    if sd == 0:
        return 1.0
    z = (abs(w - mu) - 0.5) / sd
    return float(min(1.0, 2.0 * 0.5 * math.erfc(z / np.sqrt(2))))


def fold_stats(d):
    """Per-fold contributions -> (n_folds, n_positive, p)."""
    folds = d.get("folds", {})
    c = [f["contribution"] for f in folds.values() if "contribution" in f]
    if not c:
        return 0, 0, None
    return len(c), int(sum(v > 0 for v in c)), wilcoxon_p(c)


# ── loading ───────────────────────────────────────────────────────────────────

def load_t1(root):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "res_*", "results_*.json"))):
        m = re.search(r"results_(.+)_(within|loso)\.json$", os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        runs[(m.group(1), float(d["window_sec"]), m.group(2))] = d
    return runs


def load_spatial(root, split="loso"):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "spat_*", "results_*.json"))):
        m = re.search(r"results_(.+)_(hemisphere|eccentricity)_(within|loso)\.json$",
                      os.path.basename(p))
        if not m:
            continue
        d = json.load(open(p))
        if d.get("split_setting") != split:
            continue
        runs[(m.group(1), m.group(2), float(d["window_sec"]))] = d
    return runs


def load_snr(root):
    runs = {}
    for p in sorted(glob.glob(os.path.join(root, "snr", "snr_*.json"))):
        d = json.load(open(p))
        runs[(d["mode_key"], d["split_setting"], float(d["window_sec"]))] = d
    return runs


# ── LaTeX emission ────────────────────────────────────────────────────────────

def mean_sd(d, key):
    v = d.get("mean", {}).get(key) if d else None
    return (None, None) if not v else (v[0], v[1])


def latex_table(runs, keyer, metric, pct=True, signed=False, star_no_audio=True):
    """One \\midrule-grouped block of rows; `keyer(mode, col)` -> runs key."""
    out = []
    for gi, grp in enumerate(GROUPS):
        for mode in grp:
            cells = []
            for col in COLS:
                d = runs.get(keyer(mode, col))
                v, s = mean_sd(d, metric)
                if v is None:
                    cells.append("---")
                elif signed:
                    cells.append(f"${100*v:+.2f}$" if pct else f"${v:+.4f}$")
                else:
                    cells.append(f"{100*v:.2f}$\\pm${100*s:.2f}" if pct
                                 else f"{v:.4f}")
            name = LABEL[mode]
            if star_no_audio and mode in NO_AUDIO:
                name += r"$^{\dagger}$"
            out.append(f"{name} & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            out.append(r"\midrule")
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root", help="fixbranch_results root")
    ap.add_argument("--out", default="paper_tables")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t1 = load_t1(args.root)
    sp = load_spatial(args.root, "loso")
    snr = load_snr(args.root)

    stats = {"t1": {}, "t2t3": {}, "snr": {}, "coverage": {}}

    # ---- Table II -----------------------------------------------------------
    global COLS
    COLS = [("within", w) for w in WINDOWS] + [("loso", w) for w in WINDOWS]
    k2 = lambda mode, col: (mode, col[1], col[0])
    for metric, fn, signed in (("accuracy", "table2_accuracy.tex", False),
                               ("null_mean", "table2_null.tex", False),
                               ("contribution", "table2_contribution.tex", True)):
        open(os.path.join(args.out, fn), "w").write(
            latex_table(t1, k2, metric, signed=signed))

    # ---- Table III ----------------------------------------------------------
    COLS = [("hemisphere", w) for w in WINDOWS] + [("eccentricity", w) for w in WINDOWS]
    k3 = lambda mode, col: (mode, col[0], col[1])
    for metric, fn, signed in (("accuracy", "table3_accuracy.tex", False),
                               ("null_mean", "table3_null.tex", False),
                               ("contribution", "table3_contribution.tex", True)):
        open(os.path.join(args.out, fn), "w").write(
            latex_table(sp, k3, metric, signed=signed))

    # ---- per-fold CSVs ------------------------------------------------------
    with open(os.path.join(args.out, "folds_t1.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "split", "window_s", "fold", "accuracy",
                    "null_mean", "null_std", "contribution", "p_permutation",
                    "zeros_accuracy", "flip_rate", "collapse",
                    "null_position", "contribution_position",
                    "null_trial", "contribution_trial", "n_windows",
                    "uses_audio"])
        for (mode, win, split), d in sorted(t1.items()):
            for fid, fd in sorted(d["folds"].items(), key=lambda kv: int(kv[0])):
                w.writerow([mode, split, f"{win:g}", fid] +
                           [fd.get(k) for k in
                            ("accuracy", "null_mean", "null_std", "contribution",
                             "p_permutation", "zeros_accuracy", "flip_rate",
                             "collapse", "null_position", "contribution_position",
                             "null_trial", "contribution_trial", "n_windows")] +
                           [int(mode not in NO_AUDIO)])

    with open(os.path.join(args.out, "folds_t2t3.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "task", "window_s", "fold", "accuracy", "null_mean",
                    "null_std", "contribution", "p_permutation",
                    "zeros_accuracy", "flip_rate", "collapse", "n_windows",
                    "uses_audio"])
        for (mode, task, win), d in sorted(sp.items()):
            for fid, fd in sorted(d["folds"].items(), key=lambda kv: int(kv[0])):
                w.writerow([mode, task, f"{win:g}", fid] +
                           [fd.get(k) for k in
                            ("accuracy", "null_mean", "null_std", "contribution",
                             "p_permutation", "zeros_accuracy", "flip_rate",
                             "collapse", "n_windows")] +
                           [int(mode not in NO_AUDIO)])

    with open(os.path.join(args.out, "snr_bins.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["mode", "split", "window_s", "bin", "range_db", "n_windows",
                    "accuracy", "accuracy_sd", "null_mean", "contribution",
                    "contribution_sd", "folds_positive", "n_folds"])
        for (mode, split, win), d in sorted(snr.items()):
            for b, v in sorted(d["bins"].items(), key=lambda kv: int(kv[0])):
                w.writerow([mode, split, f"{win:g}", b, v["label"],
                            v["n_windows"], v["accuracy"], v["accuracy_sd"],
                            v["null_mean"], v["contribution"],
                            v["contribution_sd"], v["folds_positive"],
                            v["n_folds"]])

    # ---- fold statistics + coverage ----------------------------------------
    for (mode, win, split), d in t1.items():
        n, pos, p = fold_stats(d)
        a, _ = mean_sd(d, "accuracy")
        stats["t1"][f"{mode}|{split}|{win:g}"] = {
            "accuracy": a, "null": mean_sd(d, "null_mean")[0],
            "contribution": mean_sd(d, "contribution")[0],
            "n_folds": n, "folds_positive": pos, "p_wilcoxon": p,
            "probe": d.get("audio_only_probe"), "uses_audio": mode not in NO_AUDIO}
    for (mode, task, win), d in sp.items():
        n, pos, p = fold_stats(d)
        stats["t2t3"][f"{mode}|{task}|{win:g}"] = {
            "accuracy": mean_sd(d, "accuracy")[0],
            "null": mean_sd(d, "null_mean")[0],
            "contribution": mean_sd(d, "contribution")[0],
            "n_folds": n, "folds_positive": pos, "p_wilcoxon": p,
            "probe": d.get("audio_only_probe"), "uses_audio": mode not in NO_AUDIO}
    for k, d in snr.items():
        stats["snr"]["|".join(str(x) for x in k)] = d["bins"]

    stats["coverage"] = {
        "t1": {"have": len(t1), "want": len(MODES) * len(WINDOWS) * 2,
               "missing": [f"{m}/w{w:g}/{s}" for s in ("within", "loso")
                           for m in MODES for w in WINDOWS
                           if (m, w, s) not in t1]},
        "t2": {"have": sum(1 for k in sp if k[1] == "hemisphere"),
               "want": len(MODES) * len(WINDOWS),
               "missing": [f"{m}/w{w:g}" for m in MODES for w in WINDOWS
                           if (m, "hemisphere", w) not in sp]},
        "t3": {"have": sum(1 for k in sp if k[1] == "eccentricity"),
               "want": len(MODES) * len(WINDOWS),
               "missing": [f"{m}/w{w:g}" for m in MODES for w in WINDOWS
                           if (m, "eccentricity", w) not in sp]},
        "snr": {"have": len(snr), "want": len(MODES) * len(WINDOWS)},
    }
    json.dump(stats, open(os.path.join(args.out, "stats.json"), "w"), indent=1)

    # ---- readable summary ---------------------------------------------------
    L = []
    cov = stats["coverage"]
    L.append("# Benchmark tables\n")
    L.append(f"T1 {cov['t1']['have']}/{cov['t1']['want']} | "
             f"T2 {cov['t2']['have']}/{cov['t2']['want']} | "
             f"T3 {cov['t3']['have']}/{cov['t3']['want']} | "
             f"SNR {cov['snr']['have']}/{cov['snr']['want']}\n")
    for tag in ("t1", "t2", "t3"):
        if cov[tag]["missing"]:
            L.append(f"\nMISSING {tag}: {', '.join(cov[tag]['missing'])}\n")
    L.append("\n`†` = no EEG, so no coupling branch: the candidate envelopes "
             "never enter the forward pass and the row is a classification of "
             "the class index, not a match-mismatch decision.\n")

    def block(title, runs, keyer, cols, chance):
        L.append(f"\n## {title}  (chance {chance})\n")
        L.append("| Mode | " + " | ".join(f"{c[0][:4]} {c[1]:g}s" for c in cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for mode in MODES:
            row = [LABEL[mode] + ("†" if mode in NO_AUDIO else "")]
            for c in cols:
                d = runs.get(keyer(mode, c))
                if d is None:
                    row.append("—")
                    continue
                a = mean_sd(d, "accuracy")[0]
                nl = mean_sd(d, "null_mean")[0]
                ct = mean_sd(d, "contribution")[0]
                n, pos, p = fold_stats(d)
                row.append(f"{100*a:.1f}/{100*nl:.1f}/**{100*ct:+.1f}** {pos}/{n}")
            L.append("| " + " | ".join(row) + " |")
        L.append("\nCells read accuracy / permuted / **contribution**, then folds "
                 "with positive contribution.\n")

    block("T1 within-subject", t1, k2, [("within", w) for w in WINDOWS], 0.25)
    block("T1 LOSO", t1, k2, [("loso", w) for w in WINDOWS], 0.25)
    block("T2 hemisphere (LOSO)", sp, k3,
          [("hemisphere", w) for w in WINDOWS], 0.50)
    block("T3 eccentricity (LOSO)", sp, k3,
          [("eccentricity", w) for w in WINDOWS], 0.50)

    if snr:
        L.append("\n## SNR-stratified (LOSO), accuracy / permuted / contribution\n")
        bins = sorted({int(b) for d in snr.values() for b in d["bins"]})
        L.append("| Mode | window | " + " | ".join(f"bin {b}" for b in bins) + " |")
        L.append("|---" * (len(bins) + 2) + "|")
        for (mode, split, win), d in sorted(snr.items()):
            row = [LABEL[mode] + ("†" if mode in NO_AUDIO else ""), f"{win:g}s"]
            for b in bins:
                v = d["bins"].get(str(b)) or d["bins"].get(b)
                row.append("—" if v is None else
                           f"{100*v['accuracy']:.1f}/{100*v['null_mean']:.1f}/"
                           f"**{100*v['contribution']:+.1f}**")
            L.append("| " + " | ".join(row) + " |")

    open(os.path.join(args.out, "summary.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote -> {args.out}/")


if __name__ == "__main__":
    main()
