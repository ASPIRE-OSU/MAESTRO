"""
recompute_significance.py
-------------------------
Re-derives the permutation test for every trained cell of the benchmark at high
permutation count, so that a significance level can be reported per model.

WHY THIS EXISTS.  The benchmark runs evaluate each cell against 20 permutations
of the physiological recordings.  Twenty is enough to establish that a decoder
uses the recordings, which is what `train_aad.py` needs in order to select a
checkpoint, but it is not enough to report a significance level: the permutation
p-value is (#{null >= real} + 1)/(n + 1), so with n = 20 no fold can score below
1/21 = 0.0476, and 1299 of the 1575 T1 fold models sit exactly on that floor.
A p-value quoted from those runs would report the permutation count, not the
effect.

WHAT THIS COMPUTES.  For every cell it reloads the per-fold checkpoints the
training runs saved, rebuilds the identical test partition, and recomputes the
null with `--n_perm` permutations (default 10000).  Permutations are cheap here
because `Evaluator` caches the encoder outputs once: a permutation re-indexes
those cached embeddings and runs the scoring head, with no encoder forward pass.

Two levels are reported.

  per fold   real accuracy, the null distribution's mean and standard deviation,
             the count of permutations reaching the real accuracy, and the exact
             permutation p-value with its floor at 1/(n_perm + 1).

  per cell   the statistic the tables report.  For each permutation index r the
             accuracy is averaged across folds, giving a null distribution for
             the cell-level mean under the hypothesis that the decision does not
             depend on the recordings; the cell's p-value is the fraction of
             that distribution reaching the observed cell mean.  This is a test
             of the quantity the tables actually print, against the same null
             those tables are referred to, and it does not assume normality or
             independence across windows within a fold.

Usage:
    python recompute_significance.py --task aad --mode eeg --split loso \
        --window_sec 10 --hop_sec 5 --local_path ... --cache_dir ... \
        --dataset_cache ... --model_root .../res --out .../significance
"""

import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import (build_dataset_cached, AADDataset, collate_fn,
                        make_candidate_bank, make_grouped_candidate_bank,
                        group_labels, load_official_splits,
                        get_official_split_windows,
                        compute_global_content_holdout, content_per_window,
                        N_SPEAKERS, VALID_MODES, WINDOW_SEC as dl_WINDOW_SEC)
from model_classification import AADModel
from evaluation import Evaluator
from train_aad import SEED, DEVICE, MODE_LABELS

TASKS = ("aad", "hemisphere", "eccentricity")


def perm_pvalue(null, real):
    """(#{null >= real} + 1) / (n + 1), the standard permutation p-value with
    the observed statistic included in the reference set."""
    n = len(null)
    return float((np.sum(np.asarray(null) >= real) + 1) / (n + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=TASKS, default="aad")
    ap.add_argument("--mode", choices=VALID_MODES, default="eeg")
    ap.add_argument("--split", choices=["within", "loso"], default="loso")
    ap.add_argument("--window_sec", type=float, required=True)
    ap.add_argument("--hop_sec", type=float, required=True)
    ap.add_argument("--local_path", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--dataset_cache", default=None)
    ap.add_argument("--splits_dir", default=None)
    ap.add_argument("--candidates", default="qmatch")
    ap.add_argument("--model_root", required=True,
                    help="the --results prefix the training run used")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_perm", type=int, default=10000)
    ap.add_argument("--skip_existing", action="store_true",
                    help="leave cells that already have an output file alone, so "
                         "an interrupted sweep resumes instead of restarting")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--held_out_content_frac", type=float, default=0.2)
    args = ap.parse_args()

    splits_dir = args.splits_dir or os.path.join(args.local_path, "splits")
    W, H = args.window_sec, args.hop_sec
    model_dir = (f"{args.model_root}_{args.split}_w{W:g}_h{H:g}_{args.candidates}")
    os.makedirs(args.out, exist_ok=True)
    n_cand = N_SPEAKERS if args.task == "aad" else 2

    name = f"sig_{args.task}_{args.mode}_{args.split}_w{W:g}.json"
    if args.skip_existing and os.path.exists(os.path.join(args.out, name)):
        print(f"{args.task} | {args.mode} | {args.split} | {W:g}s: already done")
        return
    print(f"{args.task} | {MODE_LABELS[args.mode]} | {args.split} | {W:g}s | "
          f"{args.n_perm} permutations\ncheckpoints: {model_dir}", flush=True)

    data = build_dataset_cached(local_path=args.local_path, mode=args.mode,
                                cache_dir=args.cache_dir, window_sec=W,
                                hop_sec=H, dataset_cache=args.dataset_cache)

    if args.task == "aad":
        bank = make_candidate_bank(data, args.candidates, W, H,
                                   n_cand=n_cand, seed=SEED)
        data_t = data
    else:
        bank = make_grouped_candidate_bank(data, args.task, args.candidates)
        data_t = dict(data)
        data_t["att_idxs"] = group_labels(data["att_idxs"], args.task)

    folds = load_official_splits(splits_dir, args.split)
    if args.split == "loso":
        _, heldout = compute_global_content_holdout(
            data, held_out_content_frac=args.held_out_content_frac, seed=SEED)
    win_content = content_per_window(data)

    per_fold, null_means = {}, []
    for fi in folds:
        f = fi["fold"]
        _, te = get_official_split_windows(data, fi)
        if args.split == "loso":
            te = te[np.isin(win_content[te], list(heldout))]
        if len(te) < 5:
            continue
        suffix = (f"{args.mode}_{args.split}" if args.task == "aad"
                  else f"{args.mode}_{args.task}_{args.split}")
        ckpt = os.path.join(model_dir, f"fold_{f}_{suffix}.pt")
        if not os.path.exists(ckpt):
            print(f"  fold {f}: no checkpoint, skipping", flush=True)
            continue

        loader = DataLoader(
            AADDataset(data_t, te, bank, train=False, n_cand=n_cand),
            batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn,
            num_workers=0)
        model = AADModel(mode=args.mode, n_classes=n_cand).to(DEVICE)
        model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
        ev = Evaluator(model, loader, DEVICE)

        real = ev.accuracy(ev.logits())
        nulls = np.empty(args.n_perm, dtype=np.float64)
        for k in range(args.n_perm):
            rng = np.random.default_rng(1_000_000 + 7919 * f + k)
            nulls[k] = ev.accuracy(ev.logits(perm=ev._permutation(rng)))
        null_means.append(nulls)

        per_fold[f] = {
            "accuracy": real,
            "null_mean": float(nulls.mean()),
            "null_std": float(nulls.std(ddof=1)),
            "contribution": float(real - nulls.mean()),
            "n_perm_at_or_above": int(np.sum(nulls >= real)),
            "p_permutation": perm_pvalue(nulls, real),
            "n_windows": int(ev.N),
        }
        print(f"  fold {f}: acc={real:.4f} null={nulls.mean():.4f} "
              f"D={real-nulls.mean():+.4f} p={per_fold[f]['p_permutation']:.2e}",
              flush=True)

    if not per_fold:
        print("no folds evaluated"); return

    # ── cell-level test on the quantity the tables report ────────────────────
    M = np.vstack(null_means)                       # (folds, n_perm)
    cell_null = M.mean(axis=0)                      # null of the fold-mean
    cell_real = float(np.mean([v["accuracy"] for v in per_fold.values()]))
    cell_p = perm_pvalue(cell_null, cell_real)

    out = {
        "task": args.task, "mode": args.mode, "mode_label": MODE_LABELS[args.mode],
        "split_setting": args.split, "window_sec": W, "hop_sec": H,
        "candidates": args.candidates, "n_candidates": n_cand,
        "chance_level": 1.0 / n_cand, "n_perm": args.n_perm,
        "p_floor": 1.0 / (args.n_perm + 1),
        "n_folds": len(per_fold),
        "accuracy": cell_real,
        "null_mean": float(cell_null.mean()),
        "null_std": float(cell_null.std(ddof=1)),
        "contribution": float(cell_real - cell_null.mean()),
        "p_permutation_cell": cell_p,
        "n_perm_at_or_above_cell": int(np.sum(cell_null >= cell_real)),
        "z_cell": float((cell_real - cell_null.mean()) /
                        (cell_null.std(ddof=1) + 1e-12)),
        "folds": per_fold,
        "note": "p_permutation_cell tests the fold-averaged accuracy against the "
                "distribution of fold-averaged accuracies under permutation of "
                "the physiological recordings. Its floor is 1/(n_perm+1).",
    }
    with open(os.path.join(args.out, name), "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\ncell: acc={cell_real:.4f} null={cell_null.mean():.4f} "
          f"D={out['contribution']:+.4f} z={out['z_cell']:.1f} "
          f"p={cell_p:.3e} (floor {out['p_floor']:.1e})\nsaved -> {name}")


if __name__ == "__main__":
    main()
