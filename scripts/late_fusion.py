"""
late_fusion.py
----------------
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn as pooled_collate_fn,
                        load_official_splits, get_official_split_windows,
                        carve_inner_val, carve_inner_val_content,
                        compute_global_content_holdout,
                        mode_uses, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_SINGLE_MODES = ["eeg", "gaze", "imu", "video"]

# The 11 multi-modality combinations eligible for late fusion.
MULTI_MODALITY_MODES = [
    "eeg_gaze", "eeg_imu", "eeg_video", "gaze_imu", "gaze_video", "imu_video",
    "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video", "gaze_imu_video",
    "eeg_gaze_imu_video",
]


def _active_modalities(mode: str) -> list:
    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
    active = []
    if use_eeg:   active.append("eeg")
    if use_gaze:  active.append("gaze")
    if use_imu:   active.append("imu")
    if use_video: active.append("video")
    if len(active) < 2:
        raise ValueError(
            f"mode='{mode}' activates only {active} — late fusion needs "
            f"at least 2 modalities to combine.")
    return active


# ── task registry ──────────────────────────────────────────────────────────────

def _load_task(task: str):
    """Returns (ModelClass, DatasetClass, collate_fn) for the given task."""
    if task == "aad":
        from model_classification import AADModel
        return AADModel, AADDataset, pooled_collate_fn
    elif task == "hemisphere":
        from model_spatial import AADModel
        from train_hemisphere import HemisphereDataset, collate_fn as hem_collate_fn
        return AADModel, HemisphereDataset, hem_collate_fn
    elif task == "eccentricity":
        from model_spatial import AADModel
        from train_eccentricity import EccentricityDataset, collate_fn as ecc_collate_fn
        return AADModel, EccentricityDataset, ecc_collate_fn
    else:
        raise ValueError(f"Unknown task: {task}")


def _ckpt_path(ckpt_dir: str, task: str, mode: str, fold_num: int, split_setting: str) -> str:
    """
    Matches the exact checkpoint filename conventions from train_pooled.py
    / train_hemisphere.py / train_eccentricity.py.
    """
    if task == "aad":
        fname = f"fold_{fold_num}_{mode}_{split_setting}.pt"
    elif task == "hemisphere":
        fname = f"fold_{fold_num}_{mode}_hemisphere_{split_setting}.pt"
    elif task == "eccentricity":
        fname = f"fold_{fold_num}_{mode}_eccentricity_{split_setting}.pt"
    else:
        raise ValueError(f"Unknown task: {task}")
    path = os.path.join(ckpt_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected checkpoint not found: {path}\n"
            f"Make sure you trained mode='{mode}' for task='{task}', "
            f"split_setting='{split_setting}' first.")
    return path


def _result_json_path(ckpt_dir: str, task: str, mode: str, split_setting: str) -> str:
    """Matches the exact single-modality result-file naming from the
    other three scripts, for reading (not writing) existing results."""
    if task == "aad":
        fname = f"results_{mode}_{split_setting}.json"
    elif task == "hemisphere":
        fname = f"results_{mode}_hemisphere_{split_setting}.json"
    elif task == "eccentricity":
        fname = f"results_{mode}_eccentricity_{split_setting}.json"
    else:
        raise ValueError(f"Unknown task: {task}")
    return os.path.join(ckpt_dir, fname)


# ── late fusion combiner ────────────────────────────────────────────────────────

class LateFusionCombiner(nn.Module):
    """Learns one scalar weight per active modality (softmax-normalized)
    applied to that modality's ALREADY-COMPLETE probability vector. The
    single-modality models are kept FROZEN — only these combination
    weights are trained."""

    def __init__(self, n_modalities: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_modalities))

    def forward(self, probs_list: list) -> torch.Tensor:
        w = F.softmax(self.logits, dim=0)
        stacked = torch.stack(probs_list, dim=0)
        combined = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return combined / combined.sum(dim=1, keepdim=True)


class _MeanCombiner:
    """Simple unweighted mean — no training needed."""
    def eval(self):
        pass
    def __call__(self, probs_list):
        return torch.stack(probs_list, dim=0).mean(dim=0)


def _single_modality_forward(model, mode: str, eeg, video, gaze, imu, audio):
    e = eeg   if mode == "eeg"   else None
    v = video if mode == "video" else None
    g = gaze  if mode == "gaze"  else None
    i = imu   if mode == "imu"   else None
    return model(e, v, g, i, audio)


def _to(x):
    return x.to(DEVICE) if x is not None else None


# ── evaluation ──────────────────────────────────────────────────────────────────

def evaluate_combined(models: dict, active_modalities: list,
                      loader: DataLoader, combiner=None) -> float:
    """Runs the (frozen) single-modality models + combiner over `loader`
    and returns accuracy. Used for BOTH inner-val (during combiner
    selection) and the final one-time test evaluation — the caller is
    responsible for making sure test is only ever passed in once."""
    for m in models.values():
        m.eval()
    if combiner is not None:
        combiner.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)

            probs_list = [
                _single_modality_forward(models[m], m, eeg, video, gaze, imu, audio)
                for m in active_modalities
            ]
            combined = combiner(probs_list) if combiner is not None \
                       else torch.stack(probs_list, dim=0).mean(dim=0)

            pred = combined.argmax(dim=1)
            true = labels.argmax(dim=1)
            correct += (pred == true).sum().item()
            total   += labels.size(0)

    return correct / total


def train_combiner(models: dict, active_modalities: list,
                   train_loader: DataLoader, inner_val_loader: DataLoader,
                   epochs: int = 30, lr: float = 1e-2) -> tuple:
    for m in models.values():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    combiner  = LateFusionCombiner(n_modalities=len(active_modalities)).to(DEVICE)
    optimizer = torch.optim.Adam(combiner.parameters(), lr=lr)

    best_inner_val_acc = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        combiner.train()
        for eeg, video, gaze, imu, audio, labels in train_loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)

            with torch.no_grad():
                probs_list = [
                    _single_modality_forward(models[m], m, eeg, video, gaze, imu, audio)
                    for m in active_modalities
                ]

            combined = combiner(probs_list)
            loss = -(labels * torch.log(combined + 1e-8)).sum(dim=1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        inner_val_acc = evaluate_combined(models, active_modalities, inner_val_loader, combiner)
        w = F.softmax(combiner.logits, dim=0).detach().cpu().numpy().round(3)
        print(f"  Ep {epoch:03d} | weights({','.join(active_modalities)})={w} "
              f"| inner_val_acc={inner_val_acc:.4f}")

        if inner_val_acc > best_inner_val_acc:
            best_inner_val_acc = inner_val_acc
            best_weights = combiner.state_dict()

    combiner.load_state_dict(best_weights)
    return combiner, best_inner_val_acc


# ── single-mode result reader (for --mode all) ──────────────────────────────────

def _load_single_modality_result(task: str, ckpt_dir: str, mode: str, split_setting: str) -> dict:
    """Reads a single modality's ALREADY-COMPUTED result JSON (produced
    by train_pooled.py / train_hemisphere.py / train_eccentricity.py)
    rather than re-running anything."""
    path = _result_json_path(ckpt_dir, task, mode, split_setting)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    return {
        "task": task, "mode": mode, "split_setting": split_setting,
        "combine": "single (no fusion)", "active_modalities": [mode],
        "mean_accuracy": r["mean_accuracy"], "std_accuracy": r["std_accuracy"],
    }


# ── per-mode runner ──────────────────────────────────────────────────────────────

def _result_out_path(results_dir: str, task: str, split_setting: str,
                     window_sec_eff: float, hop_sec_eff: float,
                     mode: str, combine: str) -> str:
    """Single source of truth for a per-mode result JSON's path, so the
    --skip_existing check in main() and the actual save in run_one_mode()
    can never silently drift apart."""
    return os.path.join(
        results_dir,
        f"late_fusion_{task}_{split_setting}"
        f"_w{window_sec_eff:g}_h{hop_sec_eff:g}_{mode}_{combine}.json")


def run_one_mode(args, mode: str) -> dict:
    ModelClass, DatasetClass, task_collate_fn = _load_task(args.task)
    active_modalities = _active_modalities(mode)

    print(f"\n{'#'*60}")
    print(f"# Task: {args.task}  Split: {args.split_setting}  Mode: {mode}  "
          f"(modalities: {active_modalities})")
    print(f"{'#'*60}")

    window_sec_eff = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec_eff    = args.hop_sec    if args.hop_sec    is not None else window_sec_eff
    kwargs = {}
    if args.window_sec is not None: kwargs["window_sec"] = args.window_sec
    if args.hop_sec    is not None: kwargs["hop_sec"]    = args.hop_sec

    print("Loading dataset (all 4 modalities, built once)...")
    data = build_dataset(local_path=args.local_path, mode="eeg_gaze_imu_video",
                         cache_dir=args.cache_dir, **kwargs)

    folds = load_official_splits(args.splits_dir, args.split_setting)
    fold_results = {}

    # Global content holdout for loso -- MUST use the same held_out_
    # content_frac and seed as whatever produced the checkpoints being
    # loaded (train_pooled.py / train_hemisphere.py / train_eccentricity.py),
    # or this evaluation will be testing against a DIFFERENT train/test
    # boundary than those checkpoints were actually validated against.
    if args.split_setting == "loso":
        train_content_set, heldout_content_set = compute_global_content_holdout(
            data, held_out_content_frac=args.held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): "
              f"{len(train_content_set)} train-content trials, "
              f"{len(heldout_content_set)} held-out-content trials")

    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if args.split_setting == "loso":
            # Restrict by content ON TOP of the official subject split,
            # exactly matching train_pooled.py's / train_hemisphere.py's /
            # train_eccentricity.py's own loso content-holdout logic.
            win_content = data["trial_meta_tid"][
                np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]
            is_train_content   = np.isin(win_content, list(train_content_set))
            is_heldout_content = np.isin(win_content, list(heldout_content_set))

            tr_idx = tr_idx[is_train_content[tr_idx]]
            te_idx = te_idx[is_heldout_content[te_idx]]

            inner_tr_idx, inner_vl_idx = carve_inner_val(
                data, tr_idx, val_frac=args.inner_val_frac, seed=SEED + fold_num)
        else:
            inner_tr_idx, inner_vl_idx = carve_inner_val_content(
                data, tr_idx, val_frac=args.inner_val_frac, seed=SEED + fold_num)

        models = {}
        for m in active_modalities:
            ckpt_path = _ckpt_path(args.ckpt_dir, args.task, m, fold_num, args.split_setting)
            model_m = ModelClass(mode=m).to(DEVICE)
            model_m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[m] = model_m

        te_ds = DatasetClass(data, te_idx, train=False)
        te_loader = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=task_collate_fn, num_workers=0)

        print(f"\n{'='*60}\n[{mode}] Fold {fold_num}\n{'='*60}")

        if args.combine == "mean":
            test_acc = evaluate_combined(models, active_modalities, te_loader,
                                         combiner=_MeanCombiner())
            print(f"  Mean-combine TEST accuracy: {test_acc:.4f}")
        else:
            inner_tr_ds = DatasetClass(data, inner_tr_idx, train=True)
            inner_vl_ds = DatasetClass(data, inner_vl_idx, train=False)
            inner_tr_loader = DataLoader(inner_tr_ds, batch_size=args.batch_size, shuffle=True,
                                         collate_fn=task_collate_fn, num_workers=0)
            inner_vl_loader = DataLoader(inner_vl_ds, batch_size=args.batch_size, shuffle=False,
                                         collate_fn=task_collate_fn, num_workers=0)

            combiner, best_inner_val_acc = train_combiner(
                models, active_modalities, inner_tr_loader, inner_vl_loader,
                epochs=args.epochs)
            test_acc = evaluate_combined(models, active_modalities, te_loader, combiner)
            final_w = F.softmax(combiner.logits, dim=0).detach().cpu().numpy()
            print(f"  Learned combiner best inner_val: {best_inner_val_acc:.4f} "
                  f"| TEST accuracy (reported): {test_acc:.4f}")
            print(f"  Final weights ({','.join(active_modalities)}): {final_w.round(3)}")

        fold_results[fold_num] = test_acc

    accs = list(fold_results.values())
    summary = {
        "task": args.task, "split_setting": args.split_setting, "mode": mode,
        "window_sec": window_sec_eff, "hop_sec": hop_sec_eff,
        "combine": args.combine, "active_modalities": active_modalities,
        "per_fold_test_accuracy": fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "note": "mean_accuracy computed from official-test accuracy per fold, "
               "never the inner-val score used for combiner selection.",
    }
    print(f"\n{'='*60}")
    print(f"Late Fusion Summary [{args.task}, {args.split_setting}, mode={mode}, {args.combine}]")
    print(f"  Per-fold TEST : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± Std    : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'='*60}")

    out_path = _result_out_path(args.results, args.task, args.split_setting,
                                window_sec_eff, hop_sec_eff, mode, args.combine)
    os.makedirs(args.results, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Late fusion, official-split-based, fold-matched")
    p.add_argument("--task", default='aad',
                   choices=["aad", "hemisphere", "eccentricity"])
    p.add_argument("--split_setting", default="within", choices=["loso", "within"],
                   help="Which official split protocol the checkpoints were "
                        "trained under (default: within)")
    p.add_argument("--mode", default="all",
                   choices=list(VALID_MODES) + ["all"],
                   help="Which multi-modality combination to late-fuse "
                        "(2+ active modalities), or 'all' to sweep all 11 "
                        "multi-modality combinations plus report the 4 "
                        "single-modality results read from their existing "
                        "result JSONs.")
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--splits_dir", default=None,
                   help="Path to the dataset's splits/ folder. Defaults to "
                        "<local_path>/splits if not given explicitly.")
    p.add_argument("--ckpt_dir",   default=None,
                   help="Directory containing the single-modality "
                        "checkpoints AND result JSONs. Defaults to "
                        "results_{task}_{split_setting}_w{window_sec}_h{hop_sec} "
                        "-- e.g. results_eccentricity_loso_w10_h5 -- matching "
                        "exactly what train_pooled.py/train_hemisphere.py/"
                        "train_eccentricity.py save to. Pass explicitly to "
                        "override.")
    p.add_argument("--combine",    choices=["mean", "learned"], default="learned")
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="Fraction of training subjects (loso) or training "
                        "trial content (within) held out for combiner "
                        "selection (default 0.2)")
    p.add_argument("--held_out_content_frac", type=float, default=0.2,
                   help="LOSO ONLY: fraction of trial CONTENT held out "
                        "globally, layered on top of the official "
                        "subject-based loso split. MUST match whatever "
                        "value was used to train the checkpoints being "
                        "loaded (train_pooled.py/train_hemisphere.py/"
                        "train_eccentricity.py's own --held_out_content_frac), "
                        "or this evaluates against a different train/test "
                        "boundary than those checkpoints were validated on. "
                        "Default 0.2. Ignored for --split_setting within.")
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec",    type=float, default=None)
    p.add_argument("--results",    default="results_late_fusion")
    p.add_argument("--skip_existing", action="store_true",
                   help="ONLY affects --mode all. Before training a "
                        "multi-modality mode, check whether its result "
                        "JSON already exists on disk (same task/split/"
                        "window/hop/mode/combine) and, if so, load it "
                        "instead of retraining. Lets an interrupted or "
                        "walltime-killed 'all' sweep be resumed without "
                        "re-running modes that already finished. Off by "
                        "default so existing behavior is unchanged unless "
                        "you opt in.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.local_path, "splits")

    # Computed unconditionally (not just when auto-resolving --ckpt_dir)
    # since the combined "ALL" summary filename below also needs these.
    window_sec_eff = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec_eff    = args.hop_sec    if args.hop_sec    is not None else window_sec_eff

    # Auto-compute ckpt_dir exactly matching train_pooled.py's /
    # train_hemisphere.py's / train_eccentricity.py's own results_dir
    # naming, so you don't have to type it out by hand and keep it in
    # sync manually. E.g. --task eccentricity --split_setting loso
    # --window_sec 10 --hop_sec 5  ->  results_eccentricity_loso_w10_h5
    if args.ckpt_dir is None:
        args.ckpt_dir = (f"results_{args.task}_{args.split_setting}"
                         f"_w{window_sec_eff:g}_h{hop_sec_eff:g}")
        print(f"(--ckpt_dir not given, auto-resolved to: {args.ckpt_dir})")

    os.makedirs(args.results, exist_ok=True)

    print(f"Task     : {args.task}")
    print(f"Split    : {args.split_setting}")
    print(f"Combine  : {args.combine}")
    print(f"Device   : {DEVICE}")

    if args.mode == "all":
        print(f"Mode     : ALL {len(MULTI_MODALITY_MODES)} multi-modality "
              f"combinations + {len(ALL_SINGLE_MODES)} single modalities\n")
        all_summaries = {}

        print(f"{'#'*60}\n# Single-modality results (read from existing JSONs)\n{'#'*60}")
        for mode in ALL_SINGLE_MODES:
            single = _load_single_modality_result(args.task, args.ckpt_dir, mode, args.split_setting)
            if single is None:
                print(f"  {mode:22s} -> result JSON not found, skipping "
                      f"(train it first if you want it in the summary)")
                continue
            print(f"  {mode:22s} -> mean={single['mean_accuracy']:.4f} "
                  f"± {single['std_accuracy']:.4f}")
            all_summaries[mode] = single

        for mode in MULTI_MODALITY_MODES:
            if args.skip_existing:
                existing_path = _result_out_path(
                    args.results, args.task, args.split_setting,
                    window_sec_eff, hop_sec_eff, mode, args.combine)
                if os.path.exists(existing_path):
                    with open(existing_path) as f:
                        summary = json.load(f)
                    # Sanity-check the file actually matches what we think
                    # we're loading, exactly like run_one_mode's own save
                    # implies -- an on-disk file whose internal "mode"
                    # disagrees with its filename is exactly the kind of
                    # silent mismatch that should stop the run, not be
                    # quietly trusted.
                    if summary.get("mode") != mode:
                        raise RuntimeError(
                            f"--skip_existing found {existing_path} but its "
                            f"internal 'mode' field is '{summary.get('mode')}', "
                            f"not '{mode}' -- refusing to reuse a mismatched "
                            f"file. Delete or fix it, or omit --skip_existing.")
                    print(f"  {mode:22s} -> SKIPPED (existing result found: "
                         f"{existing_path}, mean={summary['mean_accuracy']:.4f})")
                    all_summaries[mode] = summary
                    continue
            summary = run_one_mode(args, mode)
            all_summaries[mode] = summary

        print(f"\n{'#'*60}")
        print(f"# FULL SUMMARY [{args.task}, {args.split_setting}, {args.combine}]  "
              f"({len(all_summaries)} of {len(ALL_SINGLE_MODES) + len(MULTI_MODALITY_MODES)} modes)")
        print(f"{'#'*60}")
        print(f"{'Mode':<22} {'Type':<20} {'Mean':>8} {'Std':>8}")
        print("-" * 62)
        for mode in ALL_SINGLE_MODES + MULTI_MODALITY_MODES:
            if mode not in all_summaries:
                continue
            s = all_summaries[mode]
            kind = "single" if mode in ALL_SINGLE_MODES else "late fusion"
            print(f"{mode:<22} {kind:<20} {s['mean_accuracy']:>8.4f} {s['std_accuracy']:>8.4f}")

        combined_path = os.path.join(
            args.results,
            f"late_fusion_{args.task}_{args.split_setting}"
            f"_w{window_sec_eff:g}_h{hop_sec_eff:g}_ALL_{args.combine}.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nFull combined summary saved to {combined_path}")
    else:
        run_one_mode(args, args.mode)