"""
spatial_tasks.py — shared implementation of the two binary spatial benchmarks,
T2 (attended hemisphere) and T3 (attended eccentricity).

`train_hemisphere.py` and `train_eccentricity.py` are thin entry points over
this module; the only thing that differs between them is which loudspeakers are
grouped together.

WHAT CHANGED, AND WHY IT MATTERS FOR ANY NUMBER THESE SCRIPTS PRINT
-------------------------------------------------------------------
The previous revision of T2/T3 reused the T1 architecture unchanged: dilated
encoders for the recording and for each of the two grouped envelope references,
compared by a time-averaged cosine similarity and a linear read-out.  That
carries the same two defects the T1 audit found, and the binary form makes both
worse.

1. THE REFERENCES ARE ACOUSTICALLY CONFOUNDED.  Whichever group contains the
   attended talker inherits that talker's envelope signature, and the signature
   is affine-invariant, so per-candidate standardisation does not remove it.
   With only two references to choose between, a decoder with a learned audio
   encoder needs to answer an easier question than in T1.  We therefore
   distribution-match the two references (`--candidates qmatch`, the default):
   both then carry the identical multiset of values and differ only in temporal
   ordering.  The audio-only probe printed before training certifies this.

2. THE SCORING FUNCTION HAD A DEGENERATE OPTIMUM.  A time-constant recording
   embedding reduced the old read-out to a linear classifier on the audio alone.
   The coupling head here subtracts the embedding's own temporal mean first, so
   a time-constant embedding scores exactly zero on every reference, all scores
   tie, and a decoder that ignores the recording is pinned at 0.5 by arithmetic.

3. SELECTION AND REPORTING.  Checkpoints are selected on validation
   contribution, not validation accuracy, and every fold reports accuracy, the
   permutation null and their difference.  Report all three: an accuracy on its
   own does not distinguish a decoder that reads the recording from one that
   reads the references.

The orientation branch is unchanged in spirit but now predicts the binary group
rather than the loudspeaker index, and still takes no audio input, so its
permutation null is exactly 0.5 by construction.
"""

import json
import os
import zlib
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import (build_dataset, build_dataset_cached, AADDataset,
                        collate_fn, SubjectBatchSampler,
                        make_grouped_candidate_bank, group_labels,
                        SPATIAL_GROUP_NAMES,
                        audio_only_probe, load_official_splits,
                        get_official_split_windows, carve_inner_val,
                        carve_inner_val_content, compute_global_content_holdout,
                        subject_per_window, content_per_window,
                        position_in_trial, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)
from model_classification import AADModel
from losses import total_loss, DEFAULTS as LOSS_DEFAULTS
from evaluation import Evaluator
from train_aad import (SEED, DEVICE, MODE_LABELS, _mode_seed, train_model)

N_GROUPS = 2


def run_task(data, task, results_dir, mode="eeg", split_setting="within",
             splits_dir="splits", epochs=50, batch_size=32, lr=1e-3,
             inner_val_frac=0.2, held_out_content_frac=0.2,
             candidates="qmatch", window_sec=None, hop_sec=None,
             loss_cfg=None, test_shuffles=20, val_shuffles=3):
    os.makedirs(results_dir, exist_ok=True)
    label = MODE_LABELS[mode]
    g0, g1 = SPATIAL_GROUP_NAMES[task]

    # ── certify the reference pair BEFORE training ───────────────────────────
    bank = make_grouped_candidate_bank(data, task, candidates)
    probe = audio_only_probe(bank, content_per_window(data))
    chance = 1.0 / N_GROUPS
    print(f"\n[audio-only probe] task={task} construction='{candidates}' "
          f"K={N_GROUPS}: {probe:.4f} (chance {chance:.4f}, "
          f"excess {probe - chance:+.4f})")
    if probe - chance > 0.05:
        print("  WARNING: the reference pair is acoustically confounded. Any "
              "accuracy below is partly obtainable without the recording. Use "
              "--candidates qmatch.")

    # The orientation branch's auxiliary label, the test-set class distribution
    # and the candidate `pos` must all be the GROUP, not the loudspeaker.
    data_t = dict(data)
    data_t["att_idxs"] = group_labels(data["att_idxs"], task)

    folds = load_official_splits(splits_dir, split_setting)
    print(f"\nTask: {task} ({g0}/{g1}) | Mode: {label} | "
          f"Protocol: {split_setting} | {len(folds)} folds")

    if split_setting == "loso":
        train_content, heldout_content = compute_global_content_holdout(
            data, held_out_content_frac=held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): {len(train_content)} train / "
              f"{len(heldout_content)} held-out content trials")

    win_content = content_per_window(data)
    win_position = position_in_trial(data)
    fold_results = {}

    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if split_setting == "loso":
            tr_idx = tr_idx[np.isin(win_content[tr_idx], list(train_content))]
            te_idx = te_idx[np.isin(win_content[te_idx], list(heldout_content))]
            inner_tr, inner_vl = carve_inner_val(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)
        else:
            inner_tr, inner_vl = carve_inner_val_content(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)

        if len(te_idx) < 5 or len(inner_tr) < 20:
            print(f"Fold {fold_num}: too few windows, skipping")
            continue

        te_dist = dict(sorted(Counter(
            data_t["att_idxs"][te_idx].tolist()).items()))
        print(f"\n{'='*62}\nFold {fold_num}  [{task}, {split_setting}, {label}]")
        print(f"  Inner train : {len(inner_tr)} windows")
        print(f"  Inner val   : {len(inner_vl)} windows")
        print(f"  TEST        : {len(te_idx)} windows, "
              f"dist {{{g0}: {te_dist.get(0,0)}, {g1}: {te_dist.get(1,0)}}}")
        print(f"{'='*62}")

        mk = lambda idx, tr: AADDataset(data_t, idx, bank, train=tr,
                                        n_cand=N_GROUPS)
        tr_loader = DataLoader(
            mk(inner_tr, True),
            batch_sampler=SubjectBatchSampler(subject_per_window(data)[inner_tr],
                                              batch_size, seed=SEED + fold_num),
            collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(mk(inner_vl, False), batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        te_loader = DataLoader(mk(te_idx, False), batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        torch.manual_seed(_mode_seed(SEED + fold_num, mode + task))
        np.random.seed(_mode_seed(SEED + fold_num, mode + task))
        model = AADModel(mode=mode, n_classes=N_GROUPS).to(DEVICE)

        ckpt = os.path.join(results_dir,
                            f"fold_{fold_num}_{mode}_{task}_{split_setting}.pt")
        best_val = train_model(model, tr_loader, vl_loader, epochs, ckpt, lr=lr,
                               loss_cfg=loss_cfg, val_shuffles=val_shuffles)

        strata = {"position": win_position[te_idx],
                  "trial": data["trial_ids"][te_idx]}
        res = Evaluator(model, te_loader, DEVICE,
                        strata=strata).battery(n_shuffle=test_shuffles)
        res["best_val_contribution"] = best_val
        res["n_inner_train_windows"] = int(len(inner_tr))
        res["n_inner_val_windows"] = int(len(inner_vl))
        res["test_group_dist"] = te_dist

        print(f"  -> Fold {fold_num}: acc={res['accuracy']:.4f} "
              f"null={res['null_mean']:.4f} "
              f"contribution={res['contribution']:+.4f} "
              f"p={res['p_permutation']:.3f} collapse={res['collapse']:.3f}")
        fold_results[fold_num] = res

    keys = ["accuracy", "null_mean", "contribution", "zeros_accuracy",
            "flip_rate", "collapse", "p_permutation",
            "null_position", "contribution_position",
            "null_trial", "contribution_trial"]
    vals = list(fold_results.values())
    summary = {
        "task": task, "groups": [g0, g1],
        "mode": label, "split_setting": split_setting,
        "candidates": candidates, "n_candidates": N_GROUPS,
        "window_sec": window_sec, "hop_sec": hop_sec,
        "audio_only_probe": probe, "chance_level": chance,
        "folds": fold_results,
        "mean": {k: [float(np.mean([v[k] for v in vals])),
                     float(np.std([v[k] for v in vals]))]
                 for k in keys if vals and k in vals[0]},
        "note": "Report accuracy WITH null_mean and contribution. Accuracy "
                "alone does not distinguish a decoder that uses the recording "
                "from one that reads the two grouped references.",
    }
    m = summary["mean"]
    print(f"\n{split_setting.upper()} summary [{task}, {label}] "
          f"window={window_sec}s candidates={candidates}:")
    print(f"  accuracy     = {m['accuracy'][0]:.4f} +- {m['accuracy'][1]:.4f} "
          f"(chance {chance:.4f})")
    print(f"  permuted     = {m['null_mean'][0]:.4f}")
    print(f"  contribution = {m['contribution'][0]:+.4f}")

    out = os.path.join(results_dir,
                       f"results_{mode}_{task}_{split_setting}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out}")
    return summary


def build_parser(task: str, description: str):
    import argparse
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--dataset_cache", default=None)
    p.add_argument("--mode", choices=VALID_MODES, default="eeg")
    p.add_argument("--results", default=f"results_{task}")
    p.add_argument("--split_setting", choices=["loso", "within"],
                   default="within")
    p.add_argument("--splits_dir", default=None)
    p.add_argument("--candidates", choices=["qmatch", "raw"], default="qmatch",
                   help="qmatch (default) distribution-matches the two grouped "
                        "references; raw reproduces the previous, confounded "
                        "construction.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--inner_val_frac", type=float, default=0.2)
    p.add_argument("--held_out_content_frac", type=float, default=0.2)
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)
    p.add_argument("--test_shuffles", type=int, default=20)
    p.add_argument("--val_shuffles", type=int, default=3)
    for k, v in LOSS_DEFAULTS.items():
        p.add_argument(f"--{k}", type=float, default=v)
    return p


def main(task: str, description: str):
    args = build_parser(task, description).parse_args()
    splits_dir = args.splits_dir or os.path.join(args.local_path, "splits")
    window_sec = (args.window_sec if args.window_sec is not None
                  else dl_WINDOW_SEC)
    hop_sec = args.hop_sec if args.hop_sec is not None else window_sec
    args.window_sec, args.hop_sec = window_sec, hop_sec
    results_dir = (f"{args.results}_{args.split_setting}"
                   f"_w{window_sec:g}_h{hop_sec:g}_{args.candidates}")

    print(f"Device: {DEVICE} | Task: {task} | Mode: {args.mode} | "
          f"Split: {args.split_setting}")

    data = build_dataset_cached(local_path=args.local_path, mode=args.mode,
                                cache_dir=args.cache_dir,
                                window_sec=args.window_sec,
                                hop_sec=args.hop_sec,
                                dataset_cache=args.dataset_cache)
    loss_cfg = {k: getattr(args, k) for k in LOSS_DEFAULTS}
    run_task(data, task, results_dir=results_dir, mode=args.mode,
             split_setting=args.split_setting, splits_dir=splits_dir,
             epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
             inner_val_frac=args.inner_val_frac,
             held_out_content_frac=args.held_out_content_frac,
             candidates=args.candidates,
             window_sec=args.window_sec, hop_sec=args.hop_sec,
             loss_cfg=loss_cfg, test_shuffles=args.test_shuffles,
             val_shuffles=args.val_shuffles)
