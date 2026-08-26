"""
analyze_snr.py — SNR-stratified benchmark analysis, T1.

WHAT CHANGED, AND WHY
---------------------
The previous revision reported T1 accuracy per SNR bin and nothing else. Read
on its own, a rising accuracy-versus-SNR curve looks like "decoding gets easier
as the mixture gets easier". It cannot mean that unless the *floor* is flat, and
on this corpus there is no reason to assume it is: the cue that identifies the
attended talker acoustically -- an affine-invariant difference in the shape of
its amplitude envelope, which survives per-candidate standardisation -- lives in
the same mixture whose SNR is being varied. A curve without its floor cannot
distinguish "the brain tracks better at high SNR" from "the shortcut is easier
at high SNR".

This revision therefore reports THREE numbers per SNR bin:

    accuracy      what the decoder scores on that bin's windows
    permuted      what it scores on the same windows when each is given
                  ANOTHER window's recording, keeping its own candidates and
                  its own label -- the bin's audio-only floor
    contribution  accuracy - permuted, the part attributable to the recording

It also evaluates the models trained by the fixed `train_aad.py` (loading the
per-fold checkpoints those runs saved) rather than the late-fusion combiner of
frozen single-modality models, which no longer exists: under the fixed
architecture, fusion happens inside the model through modality dropout and a
fusion head, so there is no separate combiner to train.

Usage
-----
  python analyze_snr.py --mode eeg --split_setting loso \\
      --window_sec 10 --hop_sec 5 \\
      --local_path <dataset> --cache_dir <cache> --dataset_cache <dscache> \\
      --model_root /fs/scratch/.../fixbranch_results/res \\
      --results results_snr

  # every window size, one mode
  for w in 5 10 15 20 30; do
    h=$(python3 -c "print($w/2)")
    python analyze_snr.py --mode eeg --split_setting loso \\
        --window_sec $w --hop_sec $h ... ;
  done
"""

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import (build_dataset_cached, AADDataset, collate_fn,
                        make_candidate_bank, load_official_splits,
                        get_official_split_windows,
                        compute_global_content_holdout,
                        content_per_window, position_in_trial,
                        N_SPEAKERS, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)
from model_classification import AADModel
from evaluation import Evaluator
from train_aad import SEED, DEVICE, MODE_LABELS


# ── quantile-based SNR binning (unchanged from the previous revision) ──────────
# SNR is approximately normally distributed across trials, so equal-width dB
# bins leave the tails with very few windows; equal-COUNT (quantile) bins keep
# every bin's estimate comparably reliable, at the cost of each bin spanning a
# different, data-driven dB range.

def compute_snr_bins(snr_values: np.ndarray, n_bins: int) -> np.ndarray:
    """n_bins+1 edges with roughly equal counts. Computed ONCE from the full
    dataset, so "bin 0" means the same SNR range everywhere results are
    compared."""
    edges = np.percentile(snr_values, np.linspace(0, 100, n_bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    return edges


def assign_snr_bins(snr_array: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    n_bins = len(bin_edges) - 1
    return np.clip(np.digitize(snr_array, bin_edges[1:-1], right=False),
                   0, n_bins - 1)


def bin_label(i: int, edges: np.ndarray) -> str:
    return f"{edges[i]:.1f} to {edges[i+1]:.1f} dB"


def snr_per_window(data: dict, local_path: str) -> np.ndarray:
    """Per-window SNR, joined from metadata/trials.csv on stimulus-content id.

    The cached dataset already carries the content id for every window, so this
    needs no second pass over the recordings.
    """
    df = pd.read_csv(Path(local_path) / "metadata" / "trials.csv")
    df = df[df["kind"] == "main"].copy()
    assert "snr_db" in df.columns, (
        "trials.csv has no 'snr_db' column -- cannot run the SNR analysis. "
        f"Columns: {list(df.columns)}")
    # Content ids in the cached dataset are the trials.csv trial_id strings
    # ('eval_001' ...); older caches may carry the bare 1-based number, so key
    # the lookup on both forms.
    snr = df["snr_db"].to_numpy(dtype=float)
    lut = {}
    for t, v in zip(df["trial_id"].astype(str), snr):
        lut[t] = v
        m = re.search(r"(\d+)$", t)
        if m:
            lut[int(m.group(1))] = v
            lut[m.group(1)] = v
    content = content_per_window(data)

    def key(c):
        c = c.item() if hasattr(c, "item") else c
        return int(c) if isinstance(c, (int, np.integer)) else str(c)

    missing = sorted({str(key(c)) for c in np.unique(content)}
                     - {str(k) for k in lut})
    assert not missing, f"no snr_db for content ids {missing[:10]}"
    return np.array([lut[key(c)] for c in content], dtype=float)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--dataset_cache", default=None)
    p.add_argument("--mode", choices=VALID_MODES, default="eeg")
    p.add_argument("--split_setting", choices=["loso", "within"], default="loso")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--candidates", default="qmatch")
    p.add_argument("--n_candidates", type=int, default=N_SPEAKERS)
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)
    p.add_argument("--n_bins", type=int, default=4,
                   help="Number of quantile (equal-count) SNR bins")
    p.add_argument("--test_shuffles", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--held_out_content_frac", type=float, default=0.2)
    p.add_argument("--model_root", required=True,
                   help="The --results prefix the train_aad.py run used; the "
                        "per-fold checkpoints are read from "
                        "<root>_<split>_w<W>_h<H>_<candidates>/")
    p.add_argument("--results", default="results_snr")
    args = p.parse_args()

    splits_dir = args.splits_dir or os.path.join(args.local_path, "splits")
    window_sec = (args.window_sec if args.window_sec is not None
                  else dl_WINDOW_SEC)
    hop_sec = args.hop_sec if args.hop_sec is not None else window_sec
    args.window_sec, args.hop_sec = window_sec, hop_sec
    # ':g' so the name matches the one train_aad.py wrote (w10, not w10.0)
    model_dir = (f"{args.model_root}_{args.split_setting}"
                 f"_w{window_sec:g}_h{hop_sec:g}_{args.candidates}")
    os.makedirs(args.results, exist_ok=True)
    label = MODE_LABELS[args.mode]
    print(f"Device: {DEVICE} | Mode: {label} | Split: {args.split_setting} | "
          f"window {args.window_sec}s\nCheckpoints: {model_dir}")

    data = build_dataset_cached(local_path=args.local_path, mode=args.mode,
                                cache_dir=args.cache_dir,
                                window_sec=args.window_sec,
                                hop_sec=args.hop_sec,
                                dataset_cache=args.dataset_cache)
    bank = make_candidate_bank(data, args.candidates,
                               args.window_sec or dl_WINDOW_SEC, args.hop_sec,
                               n_cand=args.n_candidates, seed=SEED)

    snr = snr_per_window(data, args.local_path)
    edges = compute_snr_bins(snr, args.n_bins)
    print(f"\nSNR quantile bins ({args.n_bins}, from the full dataset):")
    for b in range(args.n_bins):
        print(f"  bin {b}: {bin_label(b, edges)}")

    folds = load_official_splits(splits_dir, args.split_setting)
    if args.split_setting == "loso":
        train_content, heldout_content = compute_global_content_holdout(
            data, held_out_content_frac=args.held_out_content_frac, seed=SEED)
    win_content = content_per_window(data)
    win_position = position_in_trial(data)

    per_fold = {}
    for fold_info in folds:
        fold_num = fold_info["fold"]
        _, te_idx = get_official_split_windows(data, fold_info)
        if args.split_setting == "loso":
            te_idx = te_idx[np.isin(win_content[te_idx], list(heldout_content))]
        if len(te_idx) < 5:
            print(f"Fold {fold_num}: too few test windows, skipping")
            continue

        ckpt = os.path.join(
            model_dir, f"fold_{fold_num}_{args.mode}_{args.split_setting}.pt")
        if not os.path.exists(ckpt):
            print(f"Fold {fold_num}: no checkpoint at {ckpt}, skipping")
            continue

        loader = DataLoader(
            AADDataset(data, te_idx, bank, train=False,
                       n_cand=args.n_candidates),
            batch_size=args.batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0)
        model = AADModel(mode=args.mode).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))

        strata = {"position": win_position[te_idx],
                  "trial": data["trial_ids"][te_idx]}
        ev = Evaluator(model, loader, DEVICE, strata=strata)
        overall = ev.battery(n_shuffle=args.test_shuffles)
        bins = ev.by_group(assign_snr_bins(snr[te_idx], edges),
                           n_shuffle=args.test_shuffles)
        per_fold[fold_num] = {"overall": overall, "bins": bins}
        print(f"  fold {fold_num}: acc={overall['accuracy']:.4f} "
              f"null={overall['null_mean']:.4f} "
              f"contribution={overall['contribution']:+.4f} | "
              + " ".join(f"b{b}:{v['accuracy']:.3f}/{v['null_mean']:.3f}"
                         for b, v in sorted(bins.items())))

    # ── aggregate: mean over folds, so every fold weighs the same ─────────────
    agg = {}
    for b in range(args.n_bins):
        rows = [f["bins"][b] for f in per_fold.values() if b in f["bins"]]
        if not rows:
            continue
        agg[b] = {
            "label": bin_label(b, edges),
            "n_folds": len(rows),
            "n_windows": int(sum(r["n"] for r in rows)),
            "accuracy": float(np.mean([r["accuracy"] for r in rows])),
            "accuracy_sd": float(np.std([r["accuracy"] for r in rows])),
            "null_mean": float(np.mean([r["null_mean"] for r in rows])),
            "contribution": float(np.mean([r["contribution"] for r in rows])),
            "contribution_sd": float(np.std([r["contribution"] for r in rows])),
            "folds_positive": int(sum(r["contribution"] > 0 for r in rows)),
        }

    out = {
        "mode": label, "mode_key": args.mode,
        "split_setting": args.split_setting,
        "candidates": args.candidates, "n_candidates": args.n_candidates,
        "window_sec": args.window_sec, "hop_sec": args.hop_sec,
        "chance_level": 1.0 / args.n_candidates,
        "snr_bin_edges": [float(e) for e in edges],
        "bins": agg,
        "overall": {
            k: float(np.mean([f["overall"][k] for f in per_fold.values()]))
            for k in ("accuracy", "null_mean", "contribution")
        } if per_fold else {},
        "per_fold": per_fold,
        "note": "Each bin reports accuracy WITH its own permutation null. A "
                "rising accuracy curve is only evidence of better neural "
                "decoding if the null does not rise with it.",
    }
    print("\nSNR bins (mean over folds):")
    print(f"{'bin':>4} {'range':>18} {'acc':>7} {'null':>7} {'contribution':>13} "
          f"{'folds+':>7}")
    for b, v in sorted(agg.items()):
        print(f"{b:>4} {v['label']:>18} {v['accuracy']:>7.4f} "
              f"{v['null_mean']:>7.4f} {v['contribution']:>+13.4f} "
              f"{v['folds_positive']}/{v['n_folds']:>3}")

    path = os.path.join(
        args.results,
        f"snr_{args.mode}_{args.split_setting}_w{window_sec:g}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
