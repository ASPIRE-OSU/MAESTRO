"""
train_pooled.py — updated for new HuggingFace dataset format.
5-fold stratified CV for 4-speaker AAD, all modes.
"""

import os, json, argparse
import zlib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        get_trial_level_splits, N_SPEAKERS, VALID_MODES)
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
    modalities) — both runs would get identical initial weights,
    identical DataLoader shuffle order, and identical per-batch
    speaker-permutation draws, differing only in the numeric content of
    the input tensors. This was empirically confirmed to produce
    bit-identical outputs (1.0000 prediction agreement, identical
    confusion matrices) between independently-trained gaze and imu
    models sharing a fold. Uses zlib.crc32 rather than Python's builtin
    hash(), which is randomized per-process (PYTHONHASHSEED) and would
    silently break run-to-run reproducibility.
    """
    return base_seed + (zlib.crc32(mode.encode()) % 10_000)

# All 18 mode names (15 canonical + 3 legacy aliases), human-readable
# labels for logging/result-JSON purposes.
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
    "gi":                  "Gaze+IMU",             # alias -> gaze_imu
    "eeg_vg":              "EEG+Gaze+Video",        # alias -> eeg_gaze_video
    "eeg_vgi":             "EEG+Gaze+IMU+Video",     # alias -> eeg_gaze_imu_video
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
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    best_acc = -1.0; patience_cnt = 0
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, tr_loader, opt, smoothing, True)
        vl_loss, vl_acc = _run_epoch(model, vl_loader, smoothing=smoothing,
                                     train=False)
        sch.step(vl_acc)
        print(f"  Ep {epoch:03d} | tr={tr_acc:.4f} vl={vl_acc:.4f}")
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
    """
    Build a fresh model, reseeding immediately beforehand so this fold's
    weight initialization is independent of however many prior folds have
    already run in this process, AND independent of which OTHER mode may
    have used this same fold number (see _mode_seed() above — without
    the mode-dependent offset, two different modes sharing a fold and an
    architecturally-identical model shape would get bit-identical
    initialization, DataLoader shuffling, and training dynamics).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return AADModel(mode=mode).to(DEVICE)


def run_kfold(data, results_dir, mode="eeg", n_splits=5,
              epochs=50, batch_size=32, lr=1e-4, smoothing=0.1):
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)
    label     = MODE_LABELS[mode]
    n_trials  = len(data["trial_meta_ids"])
    n_windows = len(data["audio"][0])
    print(f"\nMode: {label} | {n_trials} trials, {n_windows} windows")
    fold_results = {}
    for fold, tr_idx, vl_idx in get_trial_level_splits(
            data, n_splits=n_splits, seed=SEED):
        n_tr = len(np.unique(data["trial_ids"][tr_idx]))
        n_vl = len(np.unique(data["trial_ids"][vl_idx]))
        vl_dist = dict(sorted(
            Counter(data["att_idxs"][vl_idx].tolist()).items()))
        print(f"\nFold {fold+1}/{n_splits} — train {n_tr} trials, val {n_vl} trials")
        print(f"  Val dist: {vl_dist}")
        tr_ds = AADDataset(data, tr_idx, train=True)
        vl_ds = AADDataset(data, vl_idx, train=False)
        tr_loader = DataLoader(tr_ds, batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        # Reseed per-fold AND per-mode (see _mode_seed) so every fold gets
        # an independent, reproducible initialization instead of
        # inheriting whatever RNG state prior folds happened to leave
        # behind, and so different modes sharing a fold don't collide.
        model     = _new_model(mode=mode, seed=_mode_seed(SEED + fold, mode))
        ckpt_path = os.path.join(results_dir, f"fold_{fold+1}_{mode}.pt")
        best_acc  = train_model(model, tr_loader, vl_loader,
                                epochs, ckpt_path, lr, smoothing)
        print(f"  → Fold {fold+1} best val acc: {best_acc:.4f}")
        fold_results[fold + 1] = {
            "val_accuracy": best_acc, "n_train_trials": n_tr,
            "n_val_trials": n_vl, "n_train_windows": len(tr_idx),
            "n_val_windows": len(vl_idx), "val_speaker_dist": vl_dist,
        }
    accs = [v["val_accuracy"] for v in fold_results.values()]
    summary = {
        "mode": label, "folds": fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "chance_level":  1 / N_SPEAKERS,
    }
    print(f"\n5-Fold Summary [{label}]: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    out = os.path.join(results_dir, f"kfold_results_{mode}.json")
    with open(out, "w") as f: json.dump(summary, f, indent=2)
    print(f"Results saved to {out}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_path",  default='maestro',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",   default='cache',
                   help="Cache directory for video/gaze/IMU features")
    p.add_argument("--mode",        choices=VALID_MODES, default="eeg")
    p.add_argument("--results",     default="results_pooled")
    p.add_argument("--n_splits",    type=int,   default=5)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--smoothing",   type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {DEVICE} | Mode: {args.mode}")
    data = build_dataset(local_path=args.local_path, mode=args.mode,
                         cache_dir=args.cache_dir)
    run_kfold(data, results_dir=f"{args.results}",
              mode=args.mode, n_splits=args.n_splits, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr, smoothing=args.smoothing)