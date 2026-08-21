"""
collect_results.py
------------------
Assembles the per-run JSONs written by `train_aad.py` into the benchmark tables.

Every cell reports three numbers, not one:

    accuracy      what the decoder scores
    permuted      what it scores when the physiological recordings are permuted
                  across test windows, each window keeping its own candidates and
                  its own label -- i.e. the fraction obtainable from the audio
                  alone
    contribution  accuracy - permuted, the fraction attributable to the recording

An accuracy on its own does not distinguish a decoder that uses the recording
from one that reads the candidates; on the previous revision of this benchmark
those two were indistinguishable and the contribution was +0.0009.

Usage:
    python collect_results.py <results_root> [--markdown out.md]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

MODE_ORDER = ["eeg", "eeg_imu", "eeg_video", "eeg_gaze"]
MODE_LABEL = {"eeg": "EEG", "eeg_imu": "EEG+IMU",
              "eeg_video": "EEG+Video", "eeg_gaze": "EEG+Gaze"}
WINDOW_ORDER = [5.0, 10.0, 15.0, 20.0, 30.0]


def load(root):
    """root/<results>_<split>_w<W>_h<H>_<candidates>/results_<mode>_<split>.json"""
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, "*", "results_*.json"))):
        d = json.load(open(path))
        m = re.search(r"results_(.+)_(within|loso)\.json$", os.path.basename(path))
        if not m:
            continue
        mode, split = m.group(1), m.group(2)
        w = d.get("window_sec")
        if w is None:
            wm = re.search(r"_w([\d.]+)_h", os.path.dirname(path))
            w = float(wm.group(1)) if wm else None
        runs[(mode, float(w), split)] = d
    return runs


def cell(d, key):
    if d is None:
        return None
    v = d.get("mean", {}).get(key)
    return v[0] if v else None


def fmt(v, nd=4, signed=False):
    if v is None:
        return "—"
    return f"{v:+.{nd}f}" if signed else f"{v:.{nd}f}"


def table(runs, split, metric, signed=False):
    lines = ["| Modalities | " + " | ".join(f"{w:g} s" for w in WINDOW_ORDER) + " |",
             "|---" * (len(WINDOW_ORDER) + 1) + "|"]
    for mode in MODE_ORDER:
        row = [MODE_LABEL[mode]]
        for w in WINDOW_ORDER:
            d = runs.get((mode, w, split))
            v = cell(d, metric)
            sd = None
            if d and metric in d.get("mean", {}):
                sd = d["mean"][metric][1]
            row.append(fmt(v, signed=signed) +
                       (f" ± {sd:.3f}" if sd is not None and not signed else ""))
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--markdown", default=None)
    args = ap.parse_args()

    runs = load(args.root)
    have = len(runs)
    out = []
    out.append("# T1 — four-class attended-talker decoding\n")
    out.append(f"{have} / 40 runs present. Chance = 0.2500. Candidates are "
               "distribution-matched (`qmatch`), so the audio-only floor is at "
               "chance; see the probe value in each run's JSON.\n")

    for split, name in (("within", "Within-subject (intra, 5 folds)"),
                        ("loso", "Leave-one-subject-out (16 folds)")):
        out.append(f"\n## {name}\n")
        out.append("### Accuracy\n")
        out.append(table(runs, split, "accuracy"))
        out.append("\n### Accuracy under permuted recordings (audio-only floor)\n")
        out.append(table(runs, split, "null_mean"))
        out.append("\n### Contribution of the recording (accuracy − permuted)\n")
        out.append(table(runs, split, "contribution", signed=True))

    out.append("\n## Supporting controls, headline window (10 s)\n")
    out.append("| Modalities | Split | Zeros-input acc | Flip rate | Collapse | "
               "Within-trial contribution | p |")
    out.append("|---|---|---|---|---|---|---|")
    for split in ("within", "loso"):
        for mode in MODE_ORDER:
            d = runs.get((mode, 10.0, split))
            if d is None:
                continue
            out.append(f"| {MODE_LABEL[mode]} | {split} | "
                       f"{fmt(cell(d, 'zeros_accuracy'))} | "
                       f"{fmt(cell(d, 'flip_rate'), 3)} | "
                       f"{fmt(cell(d, 'collapse'), 3)} | "
                       f"{fmt(cell(d, 'contribution_trial'), signed=True)} | "
                       f"{fmt(cell(d, 'p_permutation'), 3)} |")

    missing = [(m, w, s) for s in ("within", "loso")
               for m in MODE_ORDER for w in WINDOW_ORDER
               if (m, w, s) not in runs]
    if missing:
        out.append(f"\n## Missing runs ({len(missing)})\n")
        out.append(", ".join(f"{m}/w{w:g}/{s}" for m, w, s in missing))

    text = "\n".join(out)
    print(text)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)), exist_ok=True)
        with open(args.markdown, "w") as f:
            f.write(text + "\n")
        print(f"\nsaved -> {args.markdown}")


if __name__ == "__main__":
    main()
