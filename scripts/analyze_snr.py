"""
analyze_snr_pooled.py
------------------------
SNR-stratified accuracy analysis for T1 Pooled
"""

import os
import json
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataloader import (load_trial, mode_uses, VALID_MODES, N_SPEAKERS,
                        AADDataset, collate_fn, get_trial_level_splits)
from model_classification import AADModel
from late_fusion import (LateFusionCombiner, train_combiner,
                         _active_modalities, _single_modality_forward, _to)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANONICAL_MODES = [
    "eeg", "gaze", "imu", "video",
    "eeg_gaze", "eeg_imu", "eeg_video", "gaze_imu", "gaze_video", "imu_video",
    "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video", "gaze_imu_video",
    "eeg_gaze_imu_video",
]
SINGLE_MODES = ["eeg", "gaze", "imu", "video"]

MODE_DISPLAY = {
    "eeg": "EEG", "gaze": "Gaze", "imu": "IMU", "video": "Video",
    "eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU", "eeg_video": "EEG+Video",
    "gaze_imu": "Gaze+IMU", "gaze_video": "Gaze+Video", "imu_video": "IMU+Video",
    "eeg_gaze_imu": "EEG+Gaze+IMU", "eeg_gaze_video": "EEG+Gaze+Video",
    "eeg_imu_video": "EEG+IMU+Video", "gaze_imu_video": "Gaze+IMU+Video",
    "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video",
}


# ── quantile-based SNR binning ──────────────────────────────────────────────────
# SNR is normally distributed across trials, so equal-WIDTH dB bins leave the
# tails with very few windows (as low as n=16), while the center bin can have
# n=270+ — any accuracy computed in a sparse tail bin is unreliable (can hit
# exact 0%/100% with zero variance just from having too few windows), and a
# shared-scale plot would visually treat those noisy tail points as equally
# trustworthy as the well-sampled center. Equal-COUNT (quantile) bins fix this
# by choosing bin edges so every bin gets roughly the same number of windows,
# at the cost of each bin spanning a different, data-driven dB range.

def compute_snr_bins(snr_values: np.ndarray, n_bins: int) -> np.ndarray:
    """
    Returns n_bins+1 bin edges such that each bin contains roughly equal
    counts of `snr_values`. Computed ONCE from the full dataset (not
    per-fold, not per-mode), so "bin 0" means the same SNR range
    everywhere results are compared.
    """
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(snr_values, quantiles)
    edges[0]  -= 1e-6   # ensure the minimum value falls inside bin 0, not
    edges[-1] += 1e-6   # excluded by a half-open interval at the boundary
    return edges


def assign_snr_bins(snr_array: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Maps raw SNR values to bin index (0 .. n_bins-1) using bin_edges
    from compute_snr_bins(). np.digitize with right=False + the edge
    padding above ensures every value lands in [0, n_bins-1], none fall
    outside the range."""
    n_bins = len(bin_edges) - 1
    bin_idx = np.digitize(snr_array, bin_edges[1:-1], right=False)
    return np.clip(bin_idx, 0, n_bins - 1)


def bin_label(bin_idx: int, bin_edges: np.ndarray) -> str:
    lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
    return f"{lo:.1f} to {hi:.1f} dB"


# ── self-contained dataset builder with SNR tracking (dataloader.py untouched) ──

def build_dataset_with_snr(local_path: str,
                           mode: str = "eeg_gaze_imu_video",
                           subjects="all",
                           trials: str = "main",
                           cache_dir: str = None) -> dict:
    """
    Parallel version of dataloader.build_dataset() that additionally
    reads trials.csv's "snr_db" column and returns a per-window "snr" array
    alongside everything build_dataset() already returns. Defaults to
    loading ALL 4 modalities (mode="eeg_gaze_imu_video") since this is
    built ONCE and reused for every mode's evaluation below.
    """
    assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
    root = Path(local_path)

    trials_df   = pd.read_csv(root / "metadata" / "trials.csv")
    audio_meta  = json.loads((root / "metadata" / "audio_layout.json").read_text())
    audio_layout = audio_meta["speakers"]

    if trials == "main":
        trials_df = trials_df[trials_df["kind"] == "main"].copy()

    assert "snr_db" in trials_df.columns, (
        "trials.csv has no 'snr_db' column — cannot run SNR-stratified analysis. "
        f"Columns found: {list(trials_df.columns)}")

    subj_list = list(range(1, 17)) if subjects == "all" else list(subjects)
    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)

    all_eeg      = [] if use_eeg   else None
    all_video    = [] if use_video else None
    all_gaze     = [] if use_gaze  else None
    all_imu      = [] if use_imu   else None
    all_audio    = [[] for _ in range(N_SPEAKERS)]
    all_att_idxs = []
    all_trial_ids= []
    all_snr      = []
    trial_meta   = []
    tid_ctr      = 0

    for s in subj_list:
        sid = f"S{s:02d}"
        for _, row in trials_df.iterrows():
            tid     = row["trial_id"]
            att_spk = int(row["attended_speaker"])
            snr     = int(row["snr_db"])

            result = load_trial(
                local_path      = str(root),
                sid             = sid,
                tid             = tid,
                audio_layout    = audio_layout,
                attended_speaker= att_spk,
                mode            = mode,
                cache_dir       = cache_dir,
            )
            if result is None:
                continue

            n_win = result["audio"][0].shape[0]
            if use_eeg:   all_eeg.append(result["eeg"])
            if use_video: all_video.append(result["video"])
            if use_gaze:  all_gaze.append(result["gaze"])
            if use_imu:   all_imu.append(result["imu"])
            for i in range(N_SPEAKERS):
                all_audio[i].append(result["audio"][i])
            all_att_idxs.append(result["att_idxs"])
            all_trial_ids.append(np.full(n_win, tid_ctr, dtype=np.int64))
            all_snr.append(np.full(n_win, snr, dtype=np.int64))
            trial_meta.append({"trial_id": tid_ctr, "att_idx": att_spk - 1, "snr": snr})
            tid_ctr += 1

        print(f"Subject {s}: loaded")

    def _cat(lst):
        if lst is None or not lst:
            return None
        lst = [x for x in lst if x is not None]
        return np.concatenate(lst, axis=0) if lst else None

    dataset = {
        "eeg":   _cat(all_eeg),
        "video": _cat(all_video),
        "gaze":  _cat(all_gaze),
        "imu":   _cat(all_imu),
        "audio": [np.concatenate(all_audio[i], axis=0) for i in range(N_SPEAKERS)],
        "att_idxs":           np.concatenate(all_att_idxs,  axis=0),
        "trial_ids":          np.concatenate(all_trial_ids, axis=0),
        "snr":                np.concatenate(all_snr,        axis=0),
        "trial_meta_ids":     np.array([t["trial_id"] for t in trial_meta], dtype=np.int64),
        "trial_meta_att_idx": np.array([t["att_idx"]  for t in trial_meta], dtype=np.int64),
        "trial_meta_snr":     np.array([t["snr"]       for t in trial_meta], dtype=np.int64),
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal (all modalities): {n_trials} trials, {n_windows} windows")
    from collections import Counter
    snr_dist = Counter(dataset["trial_meta_snr"].tolist())
    print("SNR dist (trials per level):", dict(sorted(snr_dist.items())))
    return dataset


# ── SNR-aware evaluation (single modality OR combiner) ─────────────────────────

def evaluate_by_snr(models: dict, active_modalities: list, loader: DataLoader,
                    bin_by_window: np.ndarray, combiner=None):
    """
    Runs `models` (one model per active modality) over `loader`
    (shuffle=False, so batch order matches bin_by_window's order
    exactly), combining their outputs via `combiner` if given (multi-
    modality case) or using the single model directly (single-modality
    case, active_modalities has length 1). `bin_by_window` holds
    PRE-COMPUTED quantile bin indices (see compute_snr_bins /
    assign_snr_bins), not raw SNR values — bucketing happens on the bin
    index here so each fold's correct/total counts are pooled within the
    bin BEFORE any ratio is taken, giving statistically correct per-bin
    accuracy rather than an after-the-fact average of raw-SNR ratios.
    """
    for m in models.values():
        m.eval()
    if combiner is not None:
        combiner.eval()

    correct_by_bin = defaultdict(int)
    total_by_bin   = defaultdict(int)

    idx = 0
    with torch.no_grad():
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)

            probs_list = [
                _single_modality_forward(models[m], m, eeg, video, gaze, imu, audio)
                for m in active_modalities
            ]
            combined = combiner(probs_list) if combiner is not None else probs_list[0]

            preds = combined.argmax(dim=1)
            trues = labels.argmax(dim=1)
            correct_mask = (preds == trues).cpu().numpy()

            batch_size = labels.size(0)
            batch_bins = bin_by_window[idx: idx + batch_size]
            idx += batch_size

            for b, is_correct in zip(batch_bins, correct_mask):
                total_by_bin[int(b)]   += 1
                correct_by_bin[int(b)] += int(is_correct)

    return correct_by_bin, total_by_bin


# ── per-mode runner ──────────────────────────────────────────────────────────────

def run_mode(mode: str, data: dict, bin_edges: np.ndarray, ckpt_dir: str,
            n_splits: int = 5, batch_size: int = 32,
            combine: str = "learned", epochs: int = 30):
    print(f"\n{'='*60}\nMode: {mode}\n{'='*60}")

    is_single = mode in SINGLE_MODES
    active_modalities = [mode] if is_single else _active_modalities(mode)
    n_bins = len(bin_edges) - 1

    per_fold_acc = defaultdict(list)
    total_all    = defaultdict(int)

    for fold, tr_idx, vl_idx in get_trial_level_splits(data, n_splits=n_splits, seed=SEED):
        fold_num = fold + 1

        # Load the single-modality checkpoint(s) this mode needs — same
        # files whether the mode is a single modality or part of a
        # multi-modality combination.
        models = {}
        missing = False
        for m in active_modalities:
            ckpt_path = os.path.join(ckpt_dir, f"fold_{fold_num}_{m}.pt")
            if not os.path.exists(ckpt_path):
                print(f"  Fold {fold_num}: checkpoint not found ({ckpt_path}), skipping fold")
                missing = True
                break
            model_m = AADModel(mode=m).to(DEVICE)
            model_m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[m] = model_m
        if missing:
            continue

        vl_ds = AADDataset(data, vl_idx, train=False)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        # Bin THIS fold's val windows using the GLOBAL bin edges (computed
        # once in main() from the full dataset), so "bin 0" means the same
        # SNR range in every fold and every mode.
        bin_for_val = assign_snr_bins(data["snr"][vl_idx], bin_edges)

        combiner = None
        if not is_single:
            if combine == "learned":
                # Re-train the combiner for this fold, reusing late_fusion.py's
                # own training logic exactly — identical to how the
                # results_late_fusion JSONs were produced, just re-run here
                # so we can additionally break the evaluation down by SNR bin.
                tr_ds = AADDataset(data, tr_idx, train=True)
                tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                                       collate_fn=collate_fn, num_workers=0)
                combiner, _ = train_combiner(
                    models, active_modalities, tr_loader, vl_loader,
                    label_map=None, epochs=epochs)
            else:   # combine == "mean" — no training needed
                combiner = _MeanCombiner()

        correct_by_bin, total_by_bin = evaluate_by_snr(
            models, active_modalities, vl_loader, bin_for_val, combiner=combiner)

        for b in total_by_bin:
            total_all[b] += total_by_bin[b]
            per_fold_acc[b].append(correct_by_bin[b] / total_by_bin[b])

        print(f"  Fold {fold_num}: evaluated "
              f"{sum(total_by_bin.values())} windows across {len(total_by_bin)} SNR bins")

    snr_results = {}
    for b in sorted(per_fold_acc.keys()):
        accs = per_fold_acc[b]
        snr_results[b] = {
            "bin_range":       bin_label(b, bin_edges),
            "mean_accuracy":   float(np.mean(accs)),
            "std_accuracy":    float(np.std(accs)),
            "n_folds":         len(accs),
            "n_windows_total": total_all[b],
        }
        print(f"  Bin {b} ({bin_label(b, bin_edges)}): "
              f"acc={np.mean(accs):.4f} ± {np.std(accs):.4f} "
              f"(n={total_all[b]} windows, {len(accs)} folds)")

    return snr_results


class _MeanCombiner:
    """Simple unweighted mean across active modalities' probability
    vectors — used when --combine mean, matching late_fusion.py's own
    'mean' option (no training needed)."""
    def eval(self):
        pass
    def __call__(self, probs_list):
        return torch.stack(probs_list, dim=0).mean(dim=0)


def parse_args():
    p = argparse.ArgumentParser(description="SNR-stratified analysis for T1 Pooled (singles + late fusion)")
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--ckpt_dir",   default="results_pooled",
                   help="Directory containing the SINGLE-modality checkpoints "
                        "(fold_k_{mode}.pt) — used for both single-modality "
                        "modes and as the frozen base models for late fusion")
    p.add_argument("--results",    default="results_snr")
    p.add_argument("--n_splits",   type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--combine",    choices=["mean", "learned"], default="learned",
                   help="How to combine multi-modality predictions — "
                        "should match how results_late_fusion was generated")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Epochs for re-training the late-fusion combiner "
                        "per fold (only used when --combine learned)")
    p.add_argument("--n_bins",     type=int, default=4,
                   help="Number of quantile (equal-count) SNR bins to use "
                        "instead of raw per-dB grouping — fixes unreliable "
                        "tail bins when SNR is normally distributed across "
                        "trials (default 4)")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Optional: only run these modes instead of all 15")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    modes_to_run = args.modes if args.modes else list(CANONICAL_MODES)
    for m in modes_to_run:
        assert m in VALID_MODES, f"Unknown mode: {m}"

    print("Loading dataset (all 4 modalities, built once, reused for every mode)...")
    data = build_dataset_with_snr(local_path=args.local_path, cache_dir=args.cache_dir)

    # Compute quantile bin edges ONCE, from the FULL dataset's SNR
    # distribution (all windows, all subjects) — so "bin 0" refers to the
    # same SNR range across every fold and every mode's evaluation.
    bin_edges = compute_snr_bins(data["snr"], args.n_bins)
    print(f"\nSNR quantile bins ({args.n_bins} bins, computed from full dataset):")
    for b in range(args.n_bins):
        print(f"  Bin {b}: {bin_label(b, bin_edges)}")

    all_results = {}
    for mode in modes_to_run:
        snr_results = run_mode(mode, data, bin_edges, args.ckpt_dir, args.n_splits,
                               args.batch_size, args.combine, args.epochs)
        all_results[mode] = snr_results

    out_path = os.path.join(args.results, "snr_analysis_pooled.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_bins": args.n_bins,
            "bin_edges": bin_edges.tolist(),
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")

    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 7))
        cmap = plt.cm.get_cmap("tab20", len(all_results))
        bin_midpoints = [(bin_edges[b] + bin_edges[b+1]) / 2 for b in range(args.n_bins)]
        for i, (mode, snr_results) in enumerate(all_results.items()):
            bins  = sorted(snr_results.keys())
            xs    = [bin_midpoints[b] for b in bins]
            means = [snr_results[b]["mean_accuracy"] * 100 for b in bins]
            stds  = [snr_results[b]["std_accuracy"]  * 100 for b in bins]
            ax.errorbar(xs, means, yerr=stds, marker="o", markersize=4,
                       label=MODE_DISPLAY.get(mode, mode), color=cmap(i),
                       linewidth=1.5, capsize=3)
        ax.axhline(25, color="gray", linewidth=1.2, linestyle="--", label="Chance (25%)")
        ax.set_xlabel("SNR (dB, quantile bin midpoint)", fontsize=13)
        ax.set_ylabel("Accuracy (%)", fontsize=13)
        ax.set_title(f"T1 Pooled: accuracy vs SNR ({args.n_bins} equal-count bins), "
                    f"all modes (singles + late fusion)", fontsize=13)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        plt.tight_layout()
        plot_path = os.path.join(args.results, "snr_analysis_pooled.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to {plot_path}")
    except ImportError:
        print("matplotlib not available — skipping plot, JSON results still saved.")


if __name__ == "__main__":
    main()