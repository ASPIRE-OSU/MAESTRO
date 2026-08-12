"""
train_aad.py — T1 (4-class attended speaker), runs under EITHER of
the dataset's two official split protocols.

--split_setting loso   : subject-generalization (16 folds, one held-out
                         subject each, official splits/loso/fold_*.json)
--split_setting within : content-generalization, pooled across all 16
                         subjects (5 folds, official splits/within/fold_*.json)
"""

import os, json, argparse
import zlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        load_official_splits, get_official_split_windows,
                        carve_inner_val, carve_inner_val_content,
                        compute_global_content_holdout,
                        N_SPEAKERS, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)
from model_classification import AADModel

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mode_seed(base_seed: int, mode: str) -> int:
    """
    Deterministic mode-dependent seed offset, fixing a seed COLLISION
    bug: per-fold reseeding alone (SEED + fold) is not sufficient when
    two DIFFERENT modes share a fold and produce architecturally
    IDENTICAL model shapes (e.g. gaze and imu are both 6-channel single
    modalities). Uses zlib.crc32 rather than Python's builtin hash(),
    which is randomized per-process (PYTHONHASHSEED) and would silently
    break run-to-run reproducibility.
    """
    return base_seed + (zlib.crc32(mode.encode()) % 10_000)

MODE_LABELS = {
    "eeg":                 "EEG only",
    "gaze":                "Gaze only",
    "imu":                 "IMU only",
    "video":               "Video only",
    "eeg_gaze":            "EEG+Gaze",
    "eeg_imu":             "EEG+IMU",
    "eeg_video":           "EEG+Video",
    "gaze_imu":            "Gaze+IMU",
    "gaze_video":          "Gaze+Video",
    "imu_video":           "IMU+Video",
    "eeg_gaze_imu":        "EEG+Gaze+IMU",
    "eeg_gaze_video":      "EEG+Gaze+Video",
    "eeg_imu_video":       "EEG+IMU+Video",
    "gaze_imu_video":      "Gaze+IMU+Video",
    "eeg_gaze_imu_video":  "EEG+Gaze+IMU+Video",
    "gi":                  "Gaze+IMU",
    "eeg_vg":              "EEG+Gaze+Video",
    "eeg_vgi":             "EEG+Gaze+IMU+Video",
}


def _label_smoothing_ce(probs, targets, smoothing=0.1):
    n_cls      = targets.size(1)
    smooth_tgt = targets * (1 - smoothing) + smoothing / n_cls
    return -(smooth_tgt * torch.log(probs + 1e-8)).sum(dim=1).mean()


def _accuracy(probs, targets):
    return (probs.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()


def _to(x):
    return x.to(DEVICE) if x is not None else None


def _run_epoch(model, loader, optimizer=None, smoothing=0.1, train=True):
    model.train(train)
    total_loss = total_acc = n = 0
    with torch.set_grad_enabled(train):
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg = _to(eeg); video = _to(video)
            gaze = _to(gaze); imu = _to(imu)
            audio = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)
            probs = model(eeg, video, gaze, imu, audio)
            loss  = _label_smoothing_ce(probs, labels, smoothing)
            if train:
                optimizer.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total_loss += loss.item()
            total_acc  += _accuracy(probs, labels)
            n += 1
    return total_loss / n, total_acc / n


def train_model(model, tr_loader, vl_loader, epochs, ckpt_path,
                lr=1e-4, smoothing=0.1):
    """Trains using tr_loader/vl_loader for checkpoint SELECTION only.
    Returns best_inner_val_acc, NOT a final reportable number."""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    best_acc = -1.0; patience_cnt = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, tr_loader, opt, smoothing, True)
        vl_loss, vl_acc = _run_epoch(model, vl_loader, smoothing=smoothing,
                                     train=False)
        sch.step(vl_acc)
        print(f"  Ep {epoch:03d} | tr={tr_acc:.4f} inner_val={vl_acc:.4f}")
        if vl_acc > best_acc:
            best_acc = vl_acc; torch.save(model.state_dict(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                print(f"  Early stopping at epoch {epoch}"); break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best_acc


def _new_model(mode: str, seed: int) -> AADModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return AADModel(mode=mode).to(DEVICE)


@torch.no_grad()
def evaluate_test(model, test_loader):
    """ONE-TIME evaluation of a selected checkpoint on the official test
    partition — called exactly once per fold, after selection is done."""
    model.eval()
    total_acc, n = 0.0, 0
    for eeg, video, gaze, imu, audio, labels in test_loader:
        eeg = _to(eeg); video = _to(video); gaze = _to(gaze); imu = _to(imu)
        audio = [a.to(DEVICE) for a in audio]
        labels = labels.to(DEVICE)
        probs = model(eeg, video, gaze, imu, audio)
        total_acc += _accuracy(probs, labels)
        n += 1
    return total_acc / n


def run_official_splits(data, results_dir, mode="eeg",
                        split_setting="within", splits_dir="splits",
                        epochs=50, batch_size=32, lr=1e-4, smoothing=0.1,
                        inner_val_frac=0.2, held_out_content_frac=0.2):
    """
    Runs T1 under the dataset's official split protocol (loso or
    within), reading splits_dir/{split_setting}/fold_*.json directly.

    For loso specifically: the OFFICIAL protocol only excludes the held-
    out subject's IDENTITY -- its test set is literally "all trials of
    the held-out subject" (see load_official_splits()'s docstring), so
    the held-out subject's trial CONTENT is still shared with the 15
    training subjects. To make the held-out subject novel in BOTH
    identity and content, a GLOBAL trial-content holdout is computed
    ONCE (via compute_global_content_holdout(), same held_out_content_
    frac and seed for every fold) and layered on top of the official
    subject split: the test set is restricted to the held-out subject's
    windows on held-out content ONLY, and the training set (for both the
    15 training subjects AND the inner-val carving) is restricted to
    non-held-out content. This is an ADDITIONAL guarantee beyond what
    the official protocol itself provides, not a replacement for it.
    """
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)
    label = MODE_LABELS[mode]

    folds = load_official_splits(splits_dir, split_setting)
    print(f"\nMode: {label} | Protocol: {split_setting} | {len(folds)} official folds")

    # Global content holdout, computed ONCE, applied identically across
    # every loso fold -- see docstring above for why this is needed on
    # top of the official protocol.
    if split_setting == "loso":
        train_content_set, heldout_content_set = compute_global_content_holdout(
            data, held_out_content_frac=held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): "
              f"{len(train_content_set)} train-content trials, "
              f"{len(heldout_content_set)} held-out-content trials "
              f"(reserved for held-out subjects' test sets only)")

    fold_results = {}
    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if split_setting == "loso":
            # Restrict by content ON TOP of the official subject split:
            # test = held-out subject's windows on held-out content only;
            # train = other 15 subjects' windows on non-held-out content
            # only. Both restrictions use the SAME global content sets
            # computed once above, so no held-out content ever appears
            # in training regardless of which subject it came from.
            win_content = data["trial_meta_tid"][
                np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]
            is_train_content   = np.isin(win_content, list(train_content_set))
            is_heldout_content = np.isin(win_content, list(heldout_content_set))

            tr_idx = tr_idx[is_train_content[tr_idx]]
            te_idx = te_idx[is_heldout_content[te_idx]]

            inner_tr_idx, inner_vl_idx = carve_inner_val(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)
        else:  # within
            inner_tr_idx, inner_vl_idx = carve_inner_val_content(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)

        te_dist = dict(sorted(Counter(data["att_idxs"][te_idx].tolist()).items()))
        print(f"\n{'='*60}")
        print(f"Fold {fold_num}  [{split_setting}, {label}]")
        print(f"  Inner train : {len(inner_tr_idx)} windows")
        print(f"  Inner val   : {len(inner_vl_idx)} windows")
        print(f"  TEST (official) : {len(te_idx)} windows, dist: {te_dist}")
        print(f"{'='*60}")

        tr_ds = AADDataset(data, inner_tr_idx, train=True)
        vl_ds = AADDataset(data, inner_vl_idx, train=False)
        te_ds = AADDataset(data, te_idx,       train=False)
        tr_loader = DataLoader(tr_ds, batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        model     = _new_model(mode=mode, seed=_mode_seed(SEED + fold_num, mode))
        ckpt_path = os.path.join(results_dir, f"fold_{fold_num}_{mode}_{split_setting}.pt")
        best_inner_val_acc = train_model(model, tr_loader, vl_loader,
                                         epochs, ckpt_path, lr, smoothing)
        test_acc = evaluate_test(model, te_loader)

        print(f"  → Fold {fold_num} best inner_val: {best_inner_val_acc:.4f} "
              f"| TEST acc (reported): {test_acc:.4f}")
        fold_results[fold_num] = {
            "test_accuracy": test_acc, "best_inner_val_acc": best_inner_val_acc,
            "n_inner_train_windows": len(inner_tr_idx),
            "n_inner_val_windows":   len(inner_vl_idx),
            "n_test_windows":        len(te_idx),
            "test_speaker_dist": te_dist,
        }

    accs = [v["test_accuracy"] for v in fold_results.values()]
    summary = {
        "mode": label, "split_setting": split_setting, "folds": fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "chance_level":  1 / N_SPEAKERS,
        "note": "mean_accuracy computed from official-test TEST accuracy per "
               "fold, never the inner-val score used for checkpoint selection.",
    }
    print(f"\n{split_setting.upper()} Summary [{label}]: "
          f"{np.mean(accs):.4f} ± {np.std(accs):.4f}")
    out = os.path.join(results_dir, f"results_{mode}_{split_setting}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"Results saved to {out}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_path",  default='maestro')
    p.add_argument("--cache_dir",   default='cache')
    p.add_argument("--mode",        choices=VALID_MODES, default="eeg")
    p.add_argument("--results",     default="results_aad")
    p.add_argument("--split_setting", choices=["loso", "within"], default="within",
                   help="Which official split protocol to use (default: within)")
    p.add_argument("--splits_dir",  default=None,
                   help="Path to the dataset's splits/ folder. "
                        "Defaults to <local_path>/splits (i.e. nested "
                        "inside the dataset root, matching the real "
                        "dataset layout) if not given explicitly.")
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--smoothing",   type=float, default=0.1)
    p.add_argument("--inner_val_frac", type=float, default=0.2,
                   help="Fraction of training subjects (loso) or training "
                        "trial content (within) held out for checkpoint "
                        "selection (default 0.2)")
    p.add_argument("--held_out_content_frac", type=float, default=0.2,
                   help="LOSO ONLY: fraction of trial CONTENT held out "
                        "globally (same set across all 16 folds), layered "
                        "on top of the official subject-based loso split, "
                        "so the held-out subject is novel in both identity "
                        "AND content -- the official protocol alone only "
                        "guarantees identity novelty (default 0.2). "
                        "Ignored for --split_setting within.")
    p.add_argument("--window_sec",  type=float, default=None)
    p.add_argument("--hop_sec",     type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.local_path, "splits")
    print(f"Device: {DEVICE} | Mode: {args.mode} | Split: {args.split_setting}")

    # Resolve the ACTUAL window/hop values that will be used (including
    # dataloader's own defaults when --window_sec/--hop_sec aren't
    # passed), so the results folder name always reflects the true
    # configuration rather than hiding it behind "unset".
    window_sec_eff = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec_eff    = args.hop_sec    if args.hop_sec    is not None else window_sec_eff

    kwargs = {}
    if args.window_sec is not None: kwargs["window_sec"] = args.window_sec
    if args.hop_sec    is not None: kwargs["hop_sec"]    = args.hop_sec
    data = build_dataset(local_path=args.local_path, mode=args.mode,
                         cache_dir=args.cache_dir, **kwargs)
    results_dir = (f"{args.results}_{args.split_setting}"
                   f"_w{window_sec_eff:g}_h{hop_sec_eff:g}")
    run_official_splits(data, results_dir=results_dir, mode=args.mode,
                        split_setting=args.split_setting, splits_dir=args.splits_dir,
                        epochs=args.epochs, batch_size=args.batch_size,
                        lr=args.lr, smoothing=args.smoothing,
                        inner_val_frac=args.inner_val_frac,
                        held_out_content_frac=args.held_out_content_frac)