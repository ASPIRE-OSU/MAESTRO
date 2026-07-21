"""
train_rec_all.py
----------------------------
Sweep script for T4 envelope reconstruction — runs train_reconstruction.py's
run_kfold() across ALL 15 canonical modality combinations (4 singles, 6
pairs, 4 triples, 1 full combination) in one invocation, rather than
calling train_reconstruction.py once per mode by hand.
"""

import os
import json
import argparse

import numpy as np

from dataloader import build_dataset, VALID_MODES
from train_reconstruction import run_kfold

# The 15 canonical modes (excludes the 3 legacy short aliases gi/eeg_vg/
# eeg_vgi, since those are redundant duplicates of gaze_imu/
# eeg_gaze_video/eeg_gaze_imu_video respectively — no need to train and
# report the same combination twice under two names).
CANONICAL_MODES = [
    "eeg", "gaze", "imu", "video",
    "eeg_gaze", "eeg_imu", "eeg_video", "gaze_imu", "gaze_video", "imu_video",
    "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video", "gaze_imu_video",
    "eeg_gaze_imu_video",
]


def parse_args():
    p = argparse.ArgumentParser(
        description="Sweep T4 linear reconstruction across all modality combinations"
    )
    p.add_argument("--local_path",   default='maestro',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",    default='cache',
                   help="Cache directory for video/gaze/IMU/EEG features")
    p.add_argument("--results",      default="results_reconstruction",
                   help="Directory for checkpoints and result JSONs "
                        "(same as train_reconstruction.py's --results)")
    p.add_argument("--n_splits",     type=int,   default=5)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--include_aliases", action="store_true",
                   help="Also run the 3 legacy short aliases (gi, eeg_vg, "
                        "eeg_vgi) even though they duplicate canonical "
                        "modes already covered — off by default")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Optional: run only these specific modes instead "
                        "of the full sweep (e.g. --modes eeg gaze video)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    if args.modes:
        modes_to_run = args.modes
        for m in modes_to_run:
            assert m in VALID_MODES, f"Unknown mode: {m}. Valid: {VALID_MODES}"
    else:
        modes_to_run = list(CANONICAL_MODES)
        if args.include_aliases:
            modes_to_run += ["gi", "eeg_vg", "eeg_vgi"]

    print(f"Running T4 reconstruction sweep across {len(modes_to_run)} modes:")
    print(f"  {modes_to_run}\n")

    all_summaries = {}

    for i, mode in enumerate(modes_to_run, 1):
        print(f"\n{'#'*60}")
        print(f"# [{i}/{len(modes_to_run)}] Mode: {mode}")
        print(f"{'#'*60}")

        data = build_dataset(local_path=args.local_path, mode=mode,
                             cache_dir=args.cache_dir)

        summary = run_kfold(
            data=data, results_dir=args.results, mode=mode,
            n_splits=args.n_splits, epochs=args.epochs,
            batch_size=args.batch_size, lr=args.lr,
            weight_decay=args.weight_decay,
        )
        all_summaries[mode] = summary

    # ── Combined summary table + JSON ────────────────────────────────────────
    print(f"\n{'#'*60}")
    print(f"# T4 RECONSTRUCTION SWEEP SUMMARY  ({len(all_summaries)} modes)")
    print(f"{'#'*60}")
    print(f"{'Mode':<22} {'Mean r':>10} {'Std':>10}")
    print("-" * 46)
    # Sort by mean_pearson_r descending, so the best mode is easy to spot
    for mode, s in sorted(all_summaries.items(),
                          key=lambda kv: -kv[1]["mean_pearson_r"]):
        print(f"{mode:<22} {s['mean_pearson_r']:>10.4f} {s['std_pearson_r']:>10.4f}")

    combined_path = os.path.join(args.results, "reconstruction_results_ALL.json")
    with open(combined_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nCombined sweep results saved to {combined_path}")


if __name__ == "__main__":
    main()