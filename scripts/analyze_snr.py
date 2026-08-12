"""
analyze_snr.py
----------------

Usage
-----
  # T1 (aad), within-subject split, 30s window
  python analyze_snr.py --task aad --split_setting within \\
      --window_sec 30 --hop_sec 30 \\
      --local_path maestro --cache_dir cache --results results_snr

  # LOSO, 10s window
  python analyze_snr.py --task aad --split_setting loso \\
      --window_sec 10 --hop_sec 5 \\
      --local_path maestro --cache_dir cache --results results_snr

  # Sweep every window size for LOSO (bash)
  for w in 5 10 15 20 30; do
    h=$(python3 -c "print($w/2)")
    python analyze_snr.py --task aad --split_setting loso \\
        --window_sec $w --hop_sec $h \\
        --local_path maestro --cache_dir cache --results results_snr
  done

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
                        load_official_splits, get_official_split_windows,
                        carve_inner_val, carve_inner_val_content,
                        compute_global_content_holdout,
                        WINDOW_SEC as dl_WINDOW_SEC)
from late_fusion import (LateFusionCombiner, train_combiner, evaluate_combined,
                         _active_modalities, _single_modality_forward, _to,
                         _MeanCombiner, _ckpt_path, _load_task,
                         ALL_SINGLE_MODES, MULTI_MODALITY_MODES)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CANONICAL_MODES = ALL_SINGLE_MODES + MULTI_MODALITY_MODES   # all 15, for reference/validation only

MODE_DISPLAY = {
    "eeg": "EEG", "gaze": "Gaze", "imu": "IMU", "video": "Video",
    "eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU", "eeg_video": "EEG+Video",
    "gaze_imu": "Gaze+IMU", "gaze_video": "Gaze+Video", "imu_video": "IMU+Video",
    "eeg_gaze_imu": "EEG+Gaze+IMU", "eeg_gaze_video": "EEG+Gaze+Video",
    "eeg_imu_video": "EEG+IMU+Video", "gaze_imu_video": "Gaze+IMU+Video",
    "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video",
}


# ── quantile-based SNR binning (unchanged from the original script) ────────────
# SNR is normally distributed across trials, so equal-width dB bins leave the
# tails with very few windows -- equal-COUNT (quantile) bins keep every bin's
# accuracy estimate comparably reliable, at the cost of each bin spanning a
# different, data-driven dB range.

def compute_snr_bins(snr_values: np.ndarray, n_bins: int) -> np.ndarray:
    """Returns n_bins+1 bin edges with roughly equal counts of snr_values.
    Computed ONCE from the full dataset (not per-fold, not per-mode), so
    "bin 0" means the same SNR range everywhere results are compared."""
    quantiles = np.linspace(0, 100, n_bins + 1)
    edges = np.percentile(snr_values, quantiles)
    edges[0]  -= 1e-6
    edges[-1] += 1e-6
    return edges


def assign_snr_bins(snr_array: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
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
                           cache_dir: str = None,
                           window_sec: float = dl_WINDOW_SEC,
                           hop_sec: float = None) -> dict:
    """
    Parallel version of dataloader.build_dataset() -- same loop
    structure, same window_sec/hop_sec pass-through to load_trial(), same
    trial_meta_subject/trial_meta_tid tracking (required by
    get_official_split_windows()) -- that additionally reads trials.csv's
    "snr_db" column and returns a per-window "snr" array plus a
    "trial_meta_snr" array (one value per trial, for reference/debugging).
    """
    assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
    root = Path(local_path)

    trials_df   = pd.read_csv(root / "metadata" / "trials.csv")
    audio_meta  = json.loads((root / "metadata" / "audio_layout.json").read_text())
    audio_layout = audio_meta["speakers"]

    if trials == "main":
        trials_df = trials_df[trials_df["kind"] == "main"].copy()

    assert "snr_db" in trials_df.columns, (
        "trials.csv has no 'snr_db' column -- cannot run SNR-stratified analysis. "
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
                window_sec      = window_sec,
                hop_sec         = hop_sec,
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
            trial_meta.append({"trial_id": tid_ctr, "att_idx": att_spk - 1,
                              "subject": s, "tid": tid, "snr": snr})
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
        "trial_meta_subject": np.array([t["subject"]  for t in trial_meta], dtype=np.int64),
        "trial_meta_tid":     np.array([t["tid"]       for t in trial_meta], dtype=object),
        "trial_meta_snr":     np.array([t["snr"]       for t in trial_meta], dtype=np.int64),
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal (all modalities): {n_trials} trials, {n_windows} windows")
    from collections import Counter
    snr_dist = Counter(dataset["trial_meta_snr"].tolist())
    print("SNR dist (trials per level):", dict(sorted(snr_dist.items())))
    return dataset


# ── SNR-aware evaluation ────────────────────────────────────────────────────────

def evaluate_by_snr(models: dict, active_modalities: list, loader: DataLoader,
                    bin_by_window: np.ndarray, combiner=None):
    """
    Same structure as before: runs `models` over `loader` (shuffle=False,
    so batch order matches bin_by_window's order exactly), combining
    outputs via `combiner` (multi-modality) or using the single model
    directly (single-modality, active_modalities has length 1).
    `bin_by_window` holds PRE-COMPUTED quantile bin indices for the exact
    windows this loader iterates -- so correct/total counts are pooled
    within each bin BEFORE any ratio is taken.
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

def run_mode(args, mode: str, data: dict, bin_edges: np.ndarray,
            ModelClass, DatasetClass, task_collate_fn,
            train_content_set=None, heldout_content_set=None) -> dict:
    print(f"\n{'='*60}\nMode: {mode}\n{'='*60}")

    is_single = mode in ALL_SINGLE_MODES
    active_modalities = [mode] if is_single else _active_modalities(mode)
    n_bins = len(bin_edges) - 1

    per_fold_acc = defaultdict(list)
    total_all    = defaultdict(int)

    folds = load_official_splits(args.splits_dir, args.split_setting)

    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if args.split_setting == "loso":
            # Restrict by content on top of the official subject split,
            # exactly matching late_fusion.py's run_one_mode() -- MUST use
            # the same held_out_content_frac/seed as whatever produced
            # the checkpoints being loaded, or this evaluates against a
            # different train/test boundary than they were validated on.
            win_content = data["trial_meta_tid"][
                np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]
            is_train_content   = np.isin(win_content, list(train_content_set))
            is_heldout_content = np.isin(win_content, list(heldout_content_set))
            tr_idx = tr_idx[is_train_content[tr_idx]]
            te_idx = te_idx[is_heldout_content[te_idx]]

        # Load this fold's frozen single-modality checkpoint(s) -- same
        # files whether mode is a single modality or part of a
        # multi-modality combination, via late_fusion.py's own
        # checkpoint-path convention (kept in sync automatically).
        models = {}
        missing = False
        for m in active_modalities:
            try:
                ckpt_path = _ckpt_path(args.ckpt_dir, args.task, m, fold_num, args.split_setting)
            except FileNotFoundError as e:
                print(f"  Fold {fold_num}: {e} -- skipping fold")
                missing = True
                break
            model_m = ModelClass(mode=m).to(DEVICE)
            model_m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[m] = model_m
        if missing:
            continue

        te_ds = DatasetClass(data, te_idx, train=False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=task_collate_fn, num_workers=0)
        # Bin the FINAL TEST windows (the ones actually being SNR-broken-
        # down and reported) using the GLOBAL bin edges computed once in
        # main() from the full dataset, so "bin 0" means the same SNR
        # range in every fold and every mode.
        bin_for_test = assign_snr_bins(data["snr"][te_idx], bin_edges)

        combiner = None
        if not is_single:
            if args.combine == "mean":
                combiner = _MeanCombiner()
            else:
                # Combiner SELECTION only ever sees the inner-val split,
                # carved the same way late_fusion.py carves it -- by
                # subject for loso, by content for within -- so no SNR-
                # bin information from the final test set can leak into
                # which combiner weights get selected.
                if args.split_setting == "loso":
                    inner_tr_idx, inner_vl_idx = carve_inner_val(
                        data, tr_idx, val_frac=args.inner_val_frac, seed=SEED + fold_num)
                else:
                    inner_tr_idx, inner_vl_idx = carve_inner_val_content(
                        data, tr_idx, val_frac=args.inner_val_frac, seed=SEED + fold_num)

                inner_tr_ds = DatasetClass(data, inner_tr_idx, train=True)
                inner_vl_ds = DatasetClass(data, inner_vl_idx, train=False)
                inner_tr_loader = DataLoader(inner_tr_ds, batch_size=args.batch_size, shuffle=True,
                                             collate_fn=task_collate_fn, num_workers=0)
                inner_vl_loader = DataLoader(inner_vl_ds, batch_size=args.batch_size, shuffle=False,
                                             collate_fn=task_collate_fn, num_workers=0)
                combiner, best_inner_val_acc = train_combiner(
                    models, active_modalities, inner_tr_loader, inner_vl_loader,
                    epochs=args.epochs)
                print(f"  Fold {fold_num}: combiner best inner_val={best_inner_val_acc:.4f}")

        correct_by_bin, total_by_bin = evaluate_by_snr(
            models, active_modalities, te_loader, bin_for_test, combiner=combiner)

        for b in total_by_bin:
            total_all[b] += total_by_bin[b]
            per_fold_acc[b].append(correct_by_bin[b] / total_by_bin[b])

        print(f"  Fold {fold_num}: evaluated "
              f"{sum(total_by_bin.values())} test windows across {len(total_by_bin)} SNR bins")

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
              f"acc={np.mean(accs):.4f} +/- {np.std(accs):.4f} "
              f"(n={total_all[b]} windows, {len(accs)} folds/subjects)")

    return snr_results


def find_best_multi_mode(late_fusion_dir: str, task: str, split_setting: str,
                         window_sec: float, hop_sec: float) -> str:
    """
    Scans the EXISTING late_fusion.py output for this exact
    (task, split_setting, window_sec, hop_sec) and returns whichever of
    the 11 multi-modality modes has the highest mean_accuracy, so this
    script only has to train ONE combiner per window size instead of
    all 11 -- the single-modality modes cost nothing extra either way
    (pure inference, no training), so this is purely about avoiding
    redundant combiner-retraining work for modes you don't actually
    need broken down by SNR.

    Uses exact filename parsing plus a cross-check against each file's
    own internal "mode" field before trusting it -- the same safeguard
    used elsewhere in this project, after an earlier loose-glob bug
    caused a real mismatch (a query for one mode silently loading a
    different mode's file).
    """
    import re, glob as _glob
    w = f"{window_sec:g}"
    pattern = re.compile(
        rf"^late_fusion_{task}_{split_setting}_w{re.escape(w)}_h[\d.]+_(.+)_learned\.json$")

    best_mode, best_acc, best_file = None, -1.0, None
    for f in _glob.glob(os.path.join(
            late_fusion_dir, f"late_fusion_{task}_{split_setting}_w{w}_h*_learned.json")):
        m = pattern.match(os.path.basename(f))
        if not m or m.group(1) not in MULTI_MODALITY_MODES:
            continue
        mode = m.group(1)
        with open(f) as fh:
            r = json.load(fh)
        if r.get("mode") != mode:
            print(f"  WARNING: {f} filename implies mode='{mode}' but its own "
                 f"'mode' field says '{r.get('mode')}' -- skipping this file "
                 f"rather than trusting a mismatched result.")
            continue
        acc = r.get("mean_accuracy")
        if acc is not None and acc > best_acc:
            best_mode, best_acc, best_file = mode, acc, f

    if best_mode is None:
        raise RuntimeError(
            f"No late_fusion result files found matching "
            f"late_fusion_{task}_{split_setting}_w{w}_h*_learned.json in "
            f"'{late_fusion_dir}' -- check --late_fusion_dir, or pass "
            f"--modes explicitly instead of --auto_best_multi.")

    print(f"Auto-selected best multimodal mode for {task}/{split_setting}/w={w}s: "
         f"'{best_mode}' (mean_accuracy={best_acc:.4f}, from {best_file})")
    return best_mode


def parse_args():
    p = argparse.ArgumentParser(description="SNR-stratified analysis, official-split-based (singles + late fusion)")
    p.add_argument("--task", default="aad", choices=["aad", "hemisphere", "eccentricity"])
    p.add_argument("--split_setting", default="loso", choices=["loso", "within"])
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--splits_dir", default=None,
                   help="Path to the dataset's splits/ folder. Defaults to <local_path>/splits.")
    p.add_argument("--ckpt_dir",   default=None,
                   help="Directory containing the single-modality checkpoints. Defaults to "
                        "results_{task}_{split_setting}_w{window_sec}_h{hop_sec}, matching "
                        "train_pooled.py/train_hemisphere.py/train_eccentricity.py's own naming.")
    p.add_argument("--results",    default="results_snr")
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec",    type=float, default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--combine",    choices=["mean", "learned"], default="learned",
                   help="Should match how results_late_fusion was generated")
    p.add_argument("--epochs",     type=int, default=30,
                   help="Epochs for re-training each fold's late-fusion combiner "
                        "(only used when --combine learned)")
    p.add_argument("--inner_val_frac", type=float, default=0.2)
    p.add_argument("--held_out_content_frac", type=float, default=0.2,
                   help="LOSO ONLY: MUST match whatever value trained the checkpoints "
                        "being loaded, or this evaluates a different train/test boundary "
                        "than those checkpoints were validated on.")
    p.add_argument("--n_bins",     type=int, default=4,
                   help="Number of quantile (equal-count) SNR bins")
    p.add_argument("--modes", nargs="+", default=None,
                   help="Optional: run exactly these modes instead of the "
                        "default (all 4 single modalities + the single best "
                        "multimodal mode, auto-selected from --late_fusion_dir).")
    p.add_argument("--late_fusion_dir", default="results_late_fusion",
                   help="Directory containing existing late_fusion.py output, "
                        "used to auto-select the best multimodal mode for "
                        "this exact task/split/window/hop (ignored if --modes "
                        "is given explicitly).")
    return p.parse_args()


def main():
    args = parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.local_path, "splits")

    window_sec_eff = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec_eff    = args.hop_sec    if args.hop_sec    is not None else window_sec_eff

    if args.ckpt_dir is None:
        args.ckpt_dir = (f"results_{args.task}_{args.split_setting}"
                         f"_w{window_sec_eff:g}_h{hop_sec_eff:g}")
        print(f"(--ckpt_dir not given, auto-resolved to: {args.ckpt_dir})")

    os.makedirs(args.results, exist_ok=True)

    if args.modes:
        modes_to_run = list(args.modes)
        for m in modes_to_run:
            assert m in VALID_MODES, f"Unknown mode: {m}"
        print(f"Running explicitly-requested modes: {modes_to_run}")
    else:
        best_multi = find_best_multi_mode(
            args.late_fusion_dir, args.task, args.split_setting,
            window_sec_eff, hop_sec_eff)
        modes_to_run = ALL_SINGLE_MODES + [best_multi]
        print(f"Running default mode set (4 singles + auto-selected best "
             f"multimodal): {modes_to_run}")

    ModelClass, DatasetClass, task_collate_fn = _load_task(args.task)

    print("Loading dataset (all 4 modalities, built once, reused for every mode)...")
    data = build_dataset_with_snr(local_path=args.local_path, cache_dir=args.cache_dir,
                                  window_sec=window_sec_eff, hop_sec=hop_sec_eff)

    # Global bin edges from the FULL dataset's SNR distribution (all
    # windows, all subjects), so "bin 0" refers to the same SNR range
    # across every fold and every mode's evaluation.
    bin_edges = compute_snr_bins(data["snr"], args.n_bins)
    print(f"\nSNR quantile bins ({args.n_bins} bins, computed from full dataset):")
    for b in range(args.n_bins):
        print(f"  Bin {b}: {bin_label(b, bin_edges)}")

    train_content_set = heldout_content_set = None
    if args.split_setting == "loso":
        train_content_set, heldout_content_set = compute_global_content_holdout(
            data, held_out_content_frac=args.held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): "
              f"{len(train_content_set)} train-content trials, "
              f"{len(heldout_content_set)} held-out-content trials")

    all_results = {}
    for mode in modes_to_run:
        snr_results = run_mode(args, mode, data, bin_edges, ModelClass, DatasetClass,
                               task_collate_fn, train_content_set, heldout_content_set)
        all_results[mode] = snr_results

    out_path = os.path.join(
        args.results,
        f"snr_analysis_{args.task}_{args.split_setting}"
        f"_w{window_sec_eff:g}_h{hop_sec_eff:g}.json")
    with open(out_path, "w") as f:
        json.dump({
            "task": args.task, "split_setting": args.split_setting,
            "window_sec": window_sec_eff, "hop_sec": hop_sec_eff,
            "combine": args.combine,
            "n_bins": args.n_bins,
            "bin_edges": bin_edges.tolist(),
            "modes_run": modes_to_run,
            "results": all_results,
        }, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()