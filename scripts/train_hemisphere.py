"""
train_hemisphere.py
-------------------
T1 (Attended Hemisphere) — binary AAD experiment.

Task
----
  Predict whether the attended speaker is in the left or right hemisphere:
    Left  (label 0): S1 (-67.5°) or S2 (-22.5°)
    Right (label 1): S3 (+22.5°) or S4 (+67.5°)

  Chance level: 0.5

Audio grouping
--------------
  env_left  = mean(env_S1, env_S2)   → left hemisphere envelope
  env_right = mean(env_S3, env_S4)   → right hemisphere envelope

  The model compares EEG/multimodal embedding against these two grouped
  envelopes via cosine similarity → binary softmax.

Modes
-----
  eeg     : EEG only
  eeg_vgi : EEG + Video + Gaze + IMU

Evaluation
----------
  5-fold stratified trial-level CV (stratified on binary hemisphere label).
  Reports binary accuracy mean ± std across folds.
  Directly comparable to prior binary AAD work.

Usage
-----
  python train_hemisphere.py --root /data --trials trials.csv --mode eeg

  python train_hemisphere.py --root /data --trials trials.csv --mode eeg_vgi \\
                             --video_root /video --cache_dir /cache
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold

from dataloader import (build_dataset, N_EEG_CH, N_VIDEO_CH,
                        N_GAZE_CH, N_IMU_CH, N_SPEAKERS, WINDOW_SAMP,
                        VALID_MODES)
from model_spatial import AADModel


# ── constants ─────────────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Supported modes for spatial tasks
SPATIAL_MODES = ("eeg", "eeg_vgi")

# T1 hemisphere label mapping (0-based speaker index → binary label)
# S1(0)=left, S2(1)=left, S3(2)=right, S4(3)=right
HEMISPHERE_LABEL = {0: 0, 1: 0, 2: 1, 3: 1}

N_CLASSES = 2   # binary: left=0, right=1

MODE_LABELS = {
    "eeg":     "EEG only",
    "eeg_vgi": "EEG+Video+Gaze+IMU",
}


# ── label helpers ─────────────────────────────────────────────────────────────

def get_hemisphere_labels(att_idxs: np.ndarray) -> np.ndarray:
    """Convert attended speaker indices (0-based) to hemisphere labels."""
    return np.array([HEMISPHERE_LABEL[i] for i in att_idxs], dtype=np.int64)


def group_audio_hemisphere(audio: list) -> list:
    """
    Group 4 speaker envelopes into 2 hemisphere envelopes.

    audio : list of 4 tensors, each (B, T, 1)
            [S1, S2, S3, S4] — original order (not rotated)

    Returns
    -------
    [env_left, env_right] each (B, T, 1)
    """
    env_left  = (audio[0] + audio[1]) / 2.0   # mean(S1, S2)
    env_right = (audio[2] + audio[3]) / 2.0   # mean(S3, S4)
    return [env_left, env_right]


# ── dataset ───────────────────────────────────────────────────────────────────

class HemisphereDataset(Dataset):
    """
    Dataset for T1 hemisphere binary decoding.

    Returns (eeg, video, gaze, imu, [env_left, env_right], label)
    where label ∈ {0=left, 1=right}.

    Audio envelopes are grouped BEFORE batching so the model always
    sees the correct hemisphere grouping. No speaker randomisation —
    S1–S4 always map to their fixed hemisphere.
    """

    def __init__(self, data: dict, window_idx: np.ndarray, train: bool = True):
        idx        = window_idx
        self.train = train

        self.eeg   = torch.from_numpy(data["eeg"][idx])   if data["eeg"]   is not None else None
        self.video = torch.from_numpy(data["video"][idx]) if data["video"] is not None else None
        self.gaze  = torch.from_numpy(data["gaze"][idx])  if data["gaze"]  is not None else None
        self.imu   = torch.from_numpy(data["imu"][idx])   if data["imu"]   is not None else None

        # Original 4-speaker envelopes (in fixed S1–S4 order)
        self.audio = [torch.from_numpy(data["audio"][i][idx])
                      for i in range(N_SPEAKERS)]

        # Binary hemisphere labels
        hem_labels      = get_hemisphere_labels(data["att_idxs"][idx])
        self.labels     = torch.from_numpy(
            np.eye(N_CLASSES, dtype=np.float32)[hem_labels]
        )  # one-hot (N, 2)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Group envelopes into hemispheres
        env_left  = (self.audio[0][idx] + self.audio[1][idx]) / 2.0
        env_right = (self.audio[2][idx] + self.audio[3][idx]) / 2.0

        return (
            self.eeg[idx]   if self.eeg   is not None else None,
            self.video[idx] if self.video is not None else None,
            self.gaze[idx]  if self.gaze  is not None else None,
            self.imu[idx]   if self.imu   is not None else None,
            [env_left, env_right],
            self.labels[idx],
        )


def collate_fn(batch):
    def _stack(i):
        return torch.stack([b[i] for b in batch]) if batch[0][i] is not None else None
    eeg    = _stack(0)
    video  = _stack(1)
    gaze   = _stack(2)
    imu    = _stack(3)
    audio  = [torch.stack([b[4][i] for b in batch]) for i in range(N_CLASSES)]
    labels = torch.stack([b[5] for b in batch])
    return eeg, video, gaze, imu, audio, labels


# ── model ─────────────────────────────────────────────────────────────────────

def _new_model(mode: str) -> AADModel:
    """Binary (2-class) AADModel for hemisphere decoding."""
    return AADModel(mode=mode).to(DEVICE)


# ── helpers ───────────────────────────────────────────────────────────────────

def _label_smoothing_ce(probs, targets, smoothing=0.1):
    n_cls      = targets.size(1)
    smooth_tgt = targets * (1 - smoothing) + smoothing / n_cls
    return -(smooth_tgt * torch.log(probs + 1e-8)).sum(dim=1).mean()


def _accuracy(probs, targets):
    return (probs.argmax(dim=1) == targets.argmax(dim=1)).float().mean().item()


def _to(x):
    return x.to(DEVICE) if x is not None else None


# ── one epoch ─────────────────────────────────────────────────────────────────

def _run_epoch(model, loader, optimizer=None,
               label_smoothing=0.1, train=True):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0

    with torch.set_grad_enabled(train):
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg    = _to(eeg)
            video  = _to(video)
            gaze   = _to(gaze)
            imu    = _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE)

            probs = model(eeg, video, gaze, imu, audio)
            loss  = _label_smoothing_ce(probs, labels, label_smoothing)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            total_acc  += _accuracy(probs, labels)
            n          += 1

    return total_loss / n, total_acc / n


# ── training loop ─────────────────────────────────────────────────────────────

def train_model(model, train_loader, val_loader,
                epochs, ckpt_path, lr=1e-4, label_smoothing=0.1):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
    )
    best_val_acc = -1.0
    patience_cnt = 0
    patience     = 10

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = _run_epoch(model, train_loader, optimizer,
                                     label_smoothing=label_smoothing, train=True)
        vl_loss, vl_acc = _run_epoch(model, val_loader,
                                     label_smoothing=label_smoothing, train=False)
        scheduler.step(vl_acc)

        print(f"  Ep {epoch:03d} | "
              f"tr_loss={tr_loss:.4f} tr_acc={tr_acc:.4f} | "
              f"vl_loss={vl_loss:.4f} vl_acc={vl_acc:.4f}")

        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            torch.save(model.state_dict(), ckpt_path)
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best_val_acc


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def run_kfold(data: dict,
              results_dir: str,
              mode: str       = "eeg",
              n_splits: int   = 5,
              epochs: int     = 50,
              batch_size: int = 32,
              lr: float       = 1e-4,
              label_smoothing: float = 0.1):
    """
    5-fold stratified CV for T1 hemisphere decoding.
    Stratified on binary hemisphere label (not 4-class attended speaker).
    """
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)

    mode_label = MODE_LABELS[mode]

    # Compute binary hemisphere labels at trial level for stratification
    trial_ids      = data["trial_meta_ids"]
    trial_att_idxs = data["trial_meta_att_idx"]
    trial_hem_labels = np.array(
        [HEMISPHERE_LABEL[i] for i in trial_att_idxs], dtype=np.int64
    )

    n_trials  = len(trial_ids)
    n_windows = len(data["audio"][0])

    print(f"\nTask    : T1 Hemisphere (left vs right)")
    print(f"Mode    : {mode_label}")
    print(f"Dataset : {n_trials} trials, {n_windows} windows")
    print(f"Label dist: {dict(sorted(Counter(trial_hem_labels.tolist()).items()))}")
    print(f"Running {n_splits}-fold stratified CV\n")

    skf          = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_results = {}

    for fold, (tr_t_idx, vl_t_idx) in enumerate(skf.split(trial_ids, trial_hem_labels)):
        tr_trial_ids = trial_ids[tr_t_idx]
        vl_trial_ids = trial_ids[vl_t_idx]

        win_trial_ids = data["trial_ids"]
        train_idx = np.where(np.isin(win_trial_ids, tr_trial_ids))[0]
        val_idx   = np.where(np.isin(win_trial_ids, vl_trial_ids))[0]

        # Val hemisphere distribution
        val_hem = get_hemisphere_labels(data["att_idxs"][val_idx])
        vl_dist = dict(sorted(Counter(val_hem.tolist()).items()))

        print(f"\n{'='*60}")
        print(f"Fold {fold+1}/{n_splits}  [T1 Hemisphere — {mode_label}]")
        print(f"  Train : {len(tr_trial_ids)} trials, {len(train_idx)} windows")
        print(f"  Val   : {len(vl_trial_ids)} trials, {len(val_idx)} windows")
        print(f"  Val hemisphere dist (0=left, 1=right): {vl_dist}")
        print(f"{'='*60}")

        tr_ds = HemisphereDataset(data, train_idx, train=True)
        vl_ds = HemisphereDataset(data, val_idx,   train=False)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        model     = _new_model(mode=mode)
        ckpt_path = os.path.join(results_dir, f"fold_{fold+1}_{mode}_hemisphere.pt")

        best_acc = train_model(
            model, tr_loader, vl_loader,
            epochs=epochs, ckpt_path=ckpt_path,
            lr=lr, label_smoothing=label_smoothing,
        )

        print(f"\n  → Fold {fold+1} best val accuracy: {best_acc:.4f}")
        fold_results[fold + 1] = {
            "val_accuracy":   best_acc,
            "n_train_trials": len(tr_trial_ids),
            "n_val_trials":   len(vl_trial_ids),
            "val_hem_dist":   vl_dist,
        }

    accs = [v["val_accuracy"] for v in fold_results.values()]
    summary = {
        "task":          "T1_hemisphere",
        "mode":          mode_label,
        "folds":         fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "chance_level":  0.5,
    }

    print(f"\n{'='*60}")
    print(f"5-Fold CV Summary  [T1 Hemisphere — {mode_label}]")
    print(f"  Per-fold : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean±Std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Chance   : 0.5000")
    print(f"{'='*60}")

    out_path = os.path.join(results_dir, f"hemisphere_results_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="T1 Hemisphere binary AAD decoding"
    )
    p.add_argument("--local_path",            default='maestro-eeg-dataset',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",       default=None,
                   help="Directory to cache preprocessed video/gaze/IMU features. "
                        "First run computes and saves; subsequent runs load instantly.")
    p.add_argument("--mode",            choices=SPATIAL_MODES, default="eeg")
    p.add_argument("--results",         default="results_hemisphere")
    p.add_argument("--n_splits",        type=int,   default=5)
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"Task   : T1 Hemisphere (left vs right)  — chance=0.5")
    print(f"Mode   : {args.mode}  ({MODE_LABELS[args.mode]})")
    if args.cache_dir:
        print(f"Cache  : {args.cache_dir}")

    print("\nLoading dataset...")
    data = build_dataset(local_path=args.local_path, mode=args.mode, cache_dir=args.cache_dir)

    run_kfold(
        data            = data,
        results_dir     = args.results,
        mode            = args.mode,
        n_splits        = args.n_splits,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        label_smoothing = args.label_smoothing,
    )