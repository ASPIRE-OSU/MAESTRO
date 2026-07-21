"""
analyze_error_complementarity.py
------------------------------------
Tests whether pairs of single modalities make SHARED or COMPLEMENTARY
errors on T1 pooled, using already-trained checkpoints (no
retraining).
"""

import os
import json
import argparse
from collections import defaultdict

import numpy as np
from scipy.stats import chi2, binomtest
import torch
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        get_trial_level_splits)
from model_classification import AADModel

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SINGLE_MODES = ["eeg", "gaze", "imu", "video"]
MODE_DISPLAY = {"eeg": "EEG", "gaze": "Gaze", "imu": "IMU", "video": "Video"}


def _to(x):
    return x.to(DEVICE) if x is not None else None


def _single_modality_forward(model, mode: str, eeg, video, gaze, imu, audio):
    e = eeg   if mode == "eeg"   else None
    v = video if mode == "video" else None
    g = gaze  if mode == "gaze"  else None
    i = imu   if mode == "imu"   else None
    return model(e, v, g, i, audio)


def get_correctness_per_window(models: dict, loader: DataLoader) -> dict:
    """
    Runs every model in `models` over the SAME batches from `loader`
    (shuffle=False, so every model sees windows in identical order),
    returning {mode: np.ndarray of bool, one per window} — True where
    that modality's prediction was correct.
    """
    for m in models.values():
        m.eval()

    correct = {mode: [] for mode in models}

    with torch.no_grad():
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)
            trues  = labels.argmax(dim=1)

            for mode, model in models.items():
                probs = _single_modality_forward(model, mode, eeg, video, gaze, imu, audio)
                preds = probs.argmax(dim=1)
                correct[mode].append((preds == trues).cpu().numpy())

    return {mode: np.concatenate(arrs) for mode, arrs in correct.items()}


def mcnemar_test(only_a: int, only_b: int) -> dict:
    """
    McNemar's test on the two off-diagonal cells of a paired 2x2 table.
    Uses the exact binomial version when the off-diagonal total is small
    (< 25, standard threshold), otherwise the chi-square approximation
    with continuity correction.
    """
    n_disagree = only_a + only_b
    if n_disagree == 0:
        return {"method": "n/a (no disagreements)", "statistic": None, "p_value": 1.0}

    if n_disagree < 25:
        # Exact binomial test: under H0, only_a ~ Binomial(n_disagree, 0.5)
        result = binomtest(only_a, n_disagree, p=0.5)
        return {"method": "exact_binomial", "statistic": None, "p_value": float(result.pvalue)}
    else:
        stat = (abs(only_a - only_b) - 1) ** 2 / n_disagree
        p    = float(1 - chi2.cdf(stat, df=1))
        return {"method": "chi2_continuity_corrected", "statistic": float(stat), "p_value": p}


def analyze_pair(mode_a: str, mode_b: str, correctness: dict) -> dict:
    a = correctness[mode_a]
    b = correctness[mode_b]
    assert len(a) == len(b), "Mismatched window counts between modalities"

    both_right   = int(np.sum(a & b))
    only_a_right = int(np.sum(a & ~b))
    only_b_right = int(np.sum(~a & b))
    both_wrong   = int(np.sum(~a & ~b))
    total = len(a)

    complementary_rate = (only_a_right + only_b_right) / total
    mcnemar = mcnemar_test(only_a_right, only_b_right)

    return {
        "mode_a": mode_a, "mode_b": mode_b, "n_windows": total,
        "both_right":   both_right,   "both_right_pct":   100 * both_right   / total,
        "only_a_right": only_a_right, "only_a_right_pct": 100 * only_a_right / total,
        "only_b_right": only_b_right, "only_b_right_pct": 100 * only_b_right / total,
        "both_wrong":   both_wrong,   "both_wrong_pct":   100 * both_wrong   / total,
        "complementary_rate_pct": 100 * complementary_rate,
        "mcnemar": mcnemar,
    }


def run(ckpt_dir: str, local_path: str, cache_dir: str,
       n_splits: int = 5, batch_size: int = 32):
    print("Loading dataset (all 4 modalities)...")
    data = build_dataset(local_path=local_path, mode="eeg_gaze_imu_video",
                         cache_dir=cache_dir)

    # Pool per-window correctness across ALL folds for each modality —
    # concatenating fold-by-fold arrays is the correct pooling here since
    # each fold contributes a disjoint set of windows (its own held-out
    # split), so concatenation naturally weights each fold by its actual
    # window count, same principle as the SNR/complementarity pooling
    # used elsewhere in this analysis.
    all_correctness = {mode: [] for mode in SINGLE_MODES}

    for fold, tr_idx, vl_idx in get_trial_level_splits(data, n_splits=n_splits, seed=SEED):
        fold_num = fold + 1
        print(f"\nFold {fold_num}/{n_splits}")

        models = {}
        missing = False
        for mode in SINGLE_MODES:
            ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_num}_{mode}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Checkpoint not found: {ckpt_path} — skipping this fold")
                missing = True
                break
            m = AADModel(mode=mode).to(DEVICE)
            m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[mode] = m
        if missing:
            continue

        vl_ds = AADDataset(data, vl_idx, train=False)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        fold_correctness = get_correctness_per_window(models, vl_loader)
        for mode in SINGLE_MODES:
            all_correctness[mode].append(fold_correctness[mode])
            acc = fold_correctness[mode].mean()
            print(f"  {MODE_DISPLAY[mode]}: {acc:.4f} accuracy on {len(fold_correctness[mode])} windows")

    pooled_correctness = {mode: np.concatenate(all_correctness[mode]) for mode in SINGLE_MODES}

    # ── all 6 pairs ──────────────────────────────────────────────────────────
    pairs = [(a, b) for i, a in enumerate(SINGLE_MODES) for b in SINGLE_MODES[i+1:]]
    results = {}
    print(f"\n{'='*70}\nPairwise error complementarity (pooled across {n_splits} folds)\n{'='*70}")
    for mode_a, mode_b in pairs:
        r = analyze_pair(mode_a, mode_b, pooled_correctness)
        key = f"{mode_a}_vs_{mode_b}"
        results[key] = r
        print(f"\n{MODE_DISPLAY[mode_a]} vs {MODE_DISPLAY[mode_b]}  (n={r['n_windows']} windows)")
        print(f"  Both right     : {r['both_right_pct']:.1f}%")
        print(f"  Only {MODE_DISPLAY[mode_a]:<5} right: {r['only_a_right_pct']:.1f}%")
        print(f"  Only {MODE_DISPLAY[mode_b]:<5} right: {r['only_b_right_pct']:.1f}%")
        print(f"  Both wrong     : {r['both_wrong_pct']:.1f}%")
        print(f"  Complementary rate: {r['complementary_rate_pct']:.1f}%")
        print(f"  McNemar's test: {r['mcnemar']['method']}, p={r['mcnemar']['p_value']:.4f}")

    return results


def parse_args():
    p = argparse.ArgumentParser(description="T1 pooled error-complementarity analysis")
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--ckpt_dir",   default="results_pooled")
    p.add_argument("--results",    default="results_complementarity")
    p.add_argument("--n_splits",   type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    results = run(args.ckpt_dir, args.local_path, args.cache_dir,
                  args.n_splits, args.batch_size)

    out_path = os.path.join(args.results, "error_complementarity_pooled.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # ── summary table, sorted by complementary rate ─────────────────────────
    print(f"\n{'='*70}\nSummary (sorted by complementary rate, highest first)\n{'='*70}")
    print(f"{'Pair':<20} {'Comp. rate':>12} {'McNemar p':>12}")
    for key, r in sorted(results.items(), key=lambda kv: -kv[1]["complementary_rate_pct"]):
        p_str = f"{r['mcnemar']['p_value']:.4f}" if r['mcnemar']['p_value'] is not None else "n/a"
        print(f"{key:<20} {r['complementary_rate_pct']:>10.1f}%  {p_str:>12}")


if __name__ == "__main__":
    main()