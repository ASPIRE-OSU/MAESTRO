"""
train_reconstruction.py
-----------------------
Envelope reconstruction experiment for T4 (linear backward model only).
"""

import os
import json
import argparse
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataloader import (build_dataset, get_trial_level_splits,
                        N_EEG_CH, N_VIDEO_CH, N_GAZE_CH, N_IMU_CH,
                        N_SPEAKERS, WINDOW_SAMP, TARGET_FS, VALID_MODES)
from model_reconstruction import (LinearModel, pearson_r, pearson_loss,
                                  n_channels_for_mode)


# ── reproducibility ───────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mode_seed(base_seed: int, mode: str) -> int:
    """
    Deterministic mode-dependent seed offset, fixing a seed COLLISION
    bug that is actually WORSE for this file than for the classification
    tasks: LinearModel's architecture is determined purely by
    n_in_channels, and several DIFFERENT modes share identical channel
    counts (e.g. gaze alone and imu alone are both 6ch; eeg_gaze and
    eeg_imu are both 38ch; gaze_video and imu_video are both 10ch).
    Per-fold reseeding alone (SEED + fold) would give every such pair
    identical initial weights, identical DataLoader shuffle order, and
    identical training dynamics whenever they share a fold — this was
    empirically confirmed (1.0000 prediction agreement, identical
    confusion matrices) between independently-trained gaze and imu
    classification models sharing a fold, and the exact same mechanism
    applies here to any pair of modes with equal n_in_channels. Uses
    zlib.crc32 rather than Python's builtin hash(), which is randomized
    per-process (PYTHONHASHSEED) and would silently break run-to-run
    reproducibility.
    """
    return base_seed + (zlib.crc32(mode.encode()) % 10_000)


# ── reconstruction dataset ────────────────────────────────────────────────────

class ReconstructionDataset(Dataset):
    """
    Dataset for the reconstruction experiment.

    Returns (x, attended_envelope) pairs where x is the concatenated
    input modalities. No speaker randomisation — the attended envelope
    is always the reconstruction target.
    """

    def __init__(self, data: dict, window_idx: np.ndarray):
        idx           = window_idx
        self.att_idxs = data["att_idxs"][idx]
        self.eeg   = torch.from_numpy(data["eeg"][idx])   if data["eeg"]   is not None else None
        self.video = torch.from_numpy(data["video"][idx]) if data["video"] is not None else None
        self.gaze  = torch.from_numpy(data["gaze"][idx])  if data["gaze"]  is not None else None
        self.imu   = torch.from_numpy(data["imu"][idx])   if data["imu"]   is not None else None
        self.audio = [torch.from_numpy(data["audio"][i][idx])
                      for i in range(N_SPEAKERS)]

    def __len__(self):
        return len(self.audio[0])

    def __getitem__(self, idx):
        att_idx = int(self.att_idxs[idx])
        att_env = self.audio[att_idx][idx]   # (T, 1) — reconstruction target

        modalities = []
        if self.eeg   is not None: modalities.append(self.eeg[idx])
        if self.gaze  is not None: modalities.append(self.gaze[idx])
        if self.imu   is not None: modalities.append(self.imu[idx])
        if self.video is not None: modalities.append(self.video[idx])

        x = torch.cat(modalities, dim=1)   # (T, C_total)
        return x, att_env


def recon_collate(batch):
    x      = torch.stack([b[0] for b in batch])   # (B, T, C_total)
    target = torch.stack([b[1] for b in batch])   # (B, T, 1)
    return x, target


# ── one epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer=None, train=True):
    model.train(train)
    total_loss = total_r = n = 0.0

    with torch.set_grad_enabled(train):
        for x, target in loader:
            x      = x.to(DEVICE)
            target = target.to(DEVICE)
            pred   = model(x)
            loss   = pearson_loss(target, pred)
            r      = pearson_r(target, pred).mean().item()

            if train:
                optimizer.zero_grad()
                loss.backward()
                # Skip update if loss is NaN to prevent weight corruption
                if not torch.isnan(loss):
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            total_loss += loss.item() if not torch.isnan(loss) else 0.0
            total_r    += r if not (r != r) else 0.0   # check for NaN
            n          += 1

    return total_loss / n, total_r / n


# ── training loop ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                epochs, ckpt_path, lr=1e-4, weight_decay=1e-3):
    """
    Adam with weight_decay (L2 penalty), approximating ridge regression —
    the standard regularization for linear backward AAD models, which
    are normally fit via ridge-regularized closed-form least squares
    rather than unconstrained gradient descent.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6)
    best_val_r   = -1.0
    patience_cnt = 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_r = _run_epoch(model, train_loader, optimizer, train=True)
        vl_loss, vl_r = _run_epoch(model, val_loader,             train=False)
        scheduler.step(vl_r)

        print(f"  Ep {epoch:03d} | "
              f"tr_loss={tr_loss:.4f} tr_r={tr_r:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_r={vl_r:.4f}")

        if vl_r > best_val_r:
            best_val_r   = vl_r
            torch.save(model.state_dict(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= 10:
                print(f"  Early stopping at epoch {epoch}")
                break

    # Only load checkpoint if it was saved at least once
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best_val_r


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def _new_model(mode: str, n_in: int, seed: int) -> LinearModel:
    """
    Build a fresh model, reseeding immediately beforehand so this fold's
    weight initialization is independent of however many prior folds have
    already run in this process, AND independent of which OTHER mode may
    have used this same fold number and happens to share the same
    n_in_channels (see _mode_seed() above).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return LinearModel(integration_window=32, n_in_channels=n_in).to(DEVICE)


def run_kfold(data: dict,
              results_dir: str,
              mode: str          = "eeg",
              n_splits: int      = 5,
              epochs: int        = 50,
              batch_size: int    = 32,
              lr: float          = 1e-4,
              weight_decay: float = 1e-3):
    """5-fold stratified trial-level CV for the linear reconstruction model."""
    os.makedirs(results_dir, exist_ok=True)

    n_trials  = len(data["trial_meta_ids"])
    n_windows = len(data["audio"][0])
    print(f"\nMode         : {mode}")
    print(f"Dataset      : {n_trials} trials, {n_windows} windows")
    print(f"Weight decay : {weight_decay}")
    print(f"Running {n_splits}-fold stratified CV\n")

    # Total input channels, derived from dataloader.mode_uses() via
    # n_channels_for_mode() rather than manually summing per-modality
    # constants for each active modality.
    n_in = n_channels_for_mode(mode)
    print(f"Input channels: {n_in}")

    fold_results = {}

    for fold, train_idx, val_idx in get_trial_level_splits(
            data, n_splits=n_splits, seed=SEED):

        n_tr = len(np.unique(data["trial_ids"][train_idx]))
        n_vl = len(np.unique(data["trial_ids"][val_idx]))

        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}")
        print(f"  Train : {n_tr} trials, {len(train_idx)} windows")
        print(f"  Val   : {n_vl} trials, {len(val_idx)} windows")
        print(f"{'='*60}")

        tr_ds = ReconstructionDataset(data, train_idx)
        vl_ds = ReconstructionDataset(data, val_idx)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=recon_collate, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=recon_collate, num_workers=0)

        # Reseed per-fold AND per-mode (see _mode_seed) so every fold gets
        # an independent, reproducible initialization instead of
        # inheriting whatever RNG state prior folds happened to leave
        # behind, and so different modes sharing a fold AND an equal
        # n_in_channels don't collide.
        model     = _new_model(mode=mode, n_in=n_in, seed=_mode_seed(SEED + fold, mode))
        ckpt_path = os.path.join(results_dir, f"fold_{fold+1}_linear_{mode}.pt")

        best_r = train_model(model, tr_loader, vl_loader,
                             epochs=epochs, ckpt_path=ckpt_path,
                             lr=lr, weight_decay=weight_decay)

        print(f"\n  → Fold {fold+1} best val Pearson r: {best_r:.4f}")
        fold_results[fold + 1] = {
            "val_pearson_r":   best_r,
            "n_train_trials":  n_tr,
            "n_val_trials":    n_vl,
            "n_train_windows": len(train_idx),
            "n_val_windows":   len(val_idx),
        }

    rs = [v["val_pearson_r"] for v in fold_results.values()]
    summary = {
        "model":          f"linear_{mode}",
        "weight_decay":   weight_decay,
        "folds":          fold_results,
        "mean_pearson_r": float(np.mean(rs)),
        "std_pearson_r":  float(np.std(rs)),
    }

    print(f"\n{'='*60}")
    print(f"5-Fold Summary [linear, {mode}]")
    print(f"  Per-fold r : {[f'{r:.4f}' for r in rs]}")
    print(f"  Mean ± Std : {np.mean(rs):.4f} ± {np.std(rs):.4f}")
    print(f"{'='*60}")

    out_path = os.path.join(results_dir,
                            f"reconstruction_results_linear_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Linear envelope reconstruction for AAD (T4)"
    )
    p.add_argument("--local_path",   default='maestro',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",    default='cache',
                   help="Cache directory for video/gaze/IMU features")
    p.add_argument("--mode",         choices=VALID_MODES, default="eeg",
                   help="Input modality mode")
    p.add_argument("--results",      default="results_reconstruction")
    p.add_argument("--n_splits",     type=int,   default=5)
    p.add_argument("--epochs",       type=int,   default=50)
    p.add_argument("--batch_size",   type=int,   default=32)
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-3,
                   help="L2 penalty on Adam, approximating ridge "
                        "regression for the linear backward model "
                        "(default 1e-3)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"Mode   : {args.mode}")
    print("\nLoading dataset...")
    data = build_dataset(local_path=args.local_path, mode=args.mode,
                         cache_dir=args.cache_dir)
    run_kfold(data=data, results_dir=args.results, mode=args.mode,
              n_splits=args.n_splits, epochs=args.epochs,
              batch_size=args.batch_size, lr=args.lr,
              weight_decay=args.weight_decay)