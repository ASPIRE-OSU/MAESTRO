"""
collect_spatial.py
------------------
Assembles the per-run JSONs written by `train_hemisphere.py` and
`train_eccentricity.py` into the T2/T3 benchmark table, and emits it in the
LaTeX row format the paper uses.

Every cell reports accuracy AND the contribution over the permutation null.
On the previous revision the two grouped references were acoustically
distinguishable -- an audio-only probe reached 0.700 against 0.500 chance -- so
an accuracy alone did not establish that anything had been decoded from the
recordings.

Usage:
    python collect_spatial.py <results_root> [--markdown out.md] [--latex out.tex]
"""

import argparse
import glob
import json
import os
import re

import numpy as np

MODE_ORDER = ["eeg", "gaze", "imu", "video",
              "eeg_gaze", "eeg_imu", "eeg_video",
              "gaze_imu", "gaze_video", "imu_video",
              "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video",
              "gaze_imu_video", "eeg_gaze_imu_video"]
MODE_LABEL = {
    "eeg": "EEG", "gaze": "Gaze", "imu": "IMU", "video": "Video",
    "eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU", "eeg_video": "EEG+Video",
    "gaze_imu": "Gaze+IMU", "gaze_video": "Gaze+Video",
    "imu_video": "IMU+Video", "eeg_gaze_imu": "EEG+Gaze+IMU",
    "eeg_gaze_video": "EEG+Gaze+Video", "eeg_imu_video": "EEG+IMU+Video",
    "gaze_imu_video": "Gaze+IMU+Video",
    "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video",
}
GROUPS = [("eeg", "gaze", "imu", "video"),
          ("eeg_gaze", "eeg_imu", "eeg_video",
           "gaze_imu", "gaze_video", "imu_video"),
          ("eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video",
           "gaze_imu_video"),
          ("eeg_gaze_imu_video",)]
WINDOWS = [5.0, 10.0, 15.0, 20.0, 30.0]
TASKS = ["hemisphere", "eccentricity"]


def load(root):
    """root/<prefix>_<split>_w<W>_h<H>_<cand>/results_<mode>_<task>_<split>.json"""
    runs, probes = {}, {}
    pat = re.compile(r"results_(.+)_(hemisphere|eccentricity)_(within|loso)\.json$")
    for path in sorted(glob.glob(os.path.join(root, "*", "results_*.json"))):
        m = pat.search(os.path.basename(path))
        if not m:
            continue
        d = json.load(open(path))
        mode, task, split = m.group(1), m.group(2), m.group(3)
        w = d.get("window_sec")
        if w is None:
            wm = re.search(r"_w([\d.]+)_h", os.path.dirname(path))
            w = float(wm.group(1)) if wm else None
        runs[(mode, task, float(w), split)] = d
        probes[(task, float(w))] = d.get("audio_only_probe")
    return runs, probes


def val(d, key):
    if d is None:
        return None
    v = d.get("mean", {}).get(key)
    return v[0] if v else None


def sd(d, key):
    if d is None:
        return None
    v = d.get("mean", {}).get(key)
    return v[1] if v else None


def table(runs, split, metric, signed=False, pct=False):
    head = ["| Modalities | " + " | ".join(
        f"{t[:4].upper()} {w:g}s" for t in TASKS for w in WINDOWS) + " |",
        "|---" * (1 + 2 * len(WINDOWS)) + "|"]
    for mode in MODE_ORDER:
        row = [MODE_LABEL[mode]]
        for t in TASKS:
            for w in WINDOWS:
                v = val(runs.get((mode, t, w, split)), metric)
                if v is None:
                    row.append("—")
                elif signed:
                    row.append(f"{v:+.4f}")
                else:
                    row.append(f"{100*v:.2f}" if pct else f"{v:.4f}")
        head.append("| " + " | ".join(row) + " |")
    return "\n".join(head)


def latex_rows(runs, task, split, metric, signed=False, pct=True):
    """Rows in the paper's own format: Mode & w5 & w10 & ... with \\midrule
    between singles / pairs / triples / all."""
    out = []
    for gi, grp in enumerate(GROUPS):
        for mode in grp:
            cells = []
            for w in WINDOWS:
                d = runs.get((mode, task, w, split))
                v, s = val(d, metric), sd(d, metric)
                if v is None:
                    cells.append("---")
                elif signed:
                    cells.append(f"${100*v:+.2f}$" if pct else f"${v:+.4f}$")
                else:
                    cells.append(f"{100*v:.2f}$\\pm${100*s:.2f}" if pct
                                 else f"{v:.4f}$\\pm${s:.4f}")
            out.append(f"{MODE_LABEL[mode]} & " + " & ".join(cells) + r" \\")
        if gi < len(GROUPS) - 1:
            out.append(r"\midrule")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--split", default="loso", choices=["loso", "within"])
    ap.add_argument("--markdown", default=None)
    ap.add_argument("--latex", default=None)
    args = ap.parse_args()

    runs, probes = load(args.root)
    want = len(MODE_ORDER) * len(WINDOWS) * len(TASKS)
    have = sum(1 for k in runs if k[3] == args.split)

    out = [f"# T2 / T3 — binary spatial decoding ({args.split})\n",
           f"{have} / {want} runs present. Chance = 0.5000. The two grouped "
           "references are distribution-matched, so the audio-only probe is "
           "near chance; the previous, un-matched construction leaked +0.200.\n",
           "\n## Audio-only acceptance probe\n",
           "| Task | " + " | ".join(f"{w:g} s" for w in WINDOWS) + " |",
           "|---" * (len(WINDOWS) + 1) + "|"]
    for t in TASKS:
        out.append("| " + t + " | " + " | ".join(
            f"{probes.get((t, w)):.4f}" if probes.get((t, w)) is not None
            else "—" for w in WINDOWS) + " |")

    out.append("\n## Accuracy (%)\n")
    out.append(table(runs, args.split, "accuracy", pct=True))
    out.append("\n## Accuracy under permuted recordings\n")
    out.append(table(runs, args.split, "null_mean"))
    out.append("\n## Contribution of the recording\n")
    out.append(table(runs, args.split, "contribution", signed=True))

    missing = [(m, t, w) for t in TASKS for m in MODE_ORDER for w in WINDOWS
               if (m, t, w, args.split) not in runs]
    if missing:
        out.append(f"\n## Missing runs ({len(missing)})\n")
        out.append(", ".join(f"{m}/{t}/w{w:g}" for m, t, w in missing))

    text = "\n".join(out)
    print(text)
    if args.markdown:
        os.makedirs(os.path.dirname(os.path.abspath(args.markdown)) or ".",
                    exist_ok=True)
        open(args.markdown, "w").write(text + "\n")
        print(f"\nsaved -> {args.markdown}")
    if args.latex:
        blocks = []
        for metric, signed in (("accuracy", False), ("contribution", True)):
            for t in TASKS:
                blocks.append(f"% ---- {t} / {metric} ----")
                blocks.append(latex_rows(runs, t, args.split, metric,
                                         signed=signed))
        open(args.latex, "w").write("\n".join(blocks) + "\n")
        print(f"saved -> {args.latex}")


if __name__ == "__main__":
    main()
