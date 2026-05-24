"""
train_loso.py
-------------
Leave-One-Subject-Out (LOSO) evaluation for 4-speaker AAD.

Protocol
--------
  For each of the 16 subjects:
    - Train  : remaining 15 subjects (all 100 eval trials each)
    - Val    : held-out subject (all 100 eval trials)
    - Early stopping on held-out subject val accuracy
    - Report best val accuracy as test accuracy for that subject

  Final result: mean ± std across 16 held-out subject accuracies.

Supported modes
---------------
  eeg     : EEG only
  eeg_vgi : EEG + Video + Gaze + IMU

Usage
-----
  python train_loso.py --root /data --trials trials.csv --mode eeg

  python train_loso.py --root /data --trials trials.csv --mode eeg_vgi \\
                       --video_root /video --cache_dir /cache
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        N_SPEAKERS, VALID_MODES)
from model_classification import AADModel


# ── constants ─────────────────────────────────────────────────────────────────

LOSO_MODES = ("eeg", "eeg_vgi")   # only these two modes for LOSO

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE_LABELS = {
    "eeg":     "EEG only",
    "eeg_vgi": "EEG+Video+Gaze+IMU",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_model(mode: str) -> AADModel:
    return AADModel(mode=mode).to(DEVICE)


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
    """
    Train with Adam + ReduceLROnPlateau + early stopping.
    Val set = held-out subject (true test set in LOSO).
    Saves best weights by val accuracy.
    Returns best val accuracy.
    """
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


# ── LOSO ──────────────────────────────────────────────────────────────────────

def run_loso(data: dict,
             results_dir: str,
             mode: str       = "eeg",
             epochs: int     = 50,
             batch_size: int = 32,
             lr: float       = 1e-4,
             label_smoothing: float = 0.1):
    """
    Leave-One-Subject-Out evaluation.

    Parameters
    ----------
    data        : dict from build_dataset() — must contain "subject_ids" key
    results_dir : directory for checkpoints and result JSON
    mode        : "eeg" or "eeg_vgi"
    """
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)

    mode_label  = MODE_LABELS[mode]
    subject_ids = np.unique(data["subject_ids"])
    n_subjects  = len(subject_ids)
    n_windows   = len(data["audio"][0])

    print(f"\nMode       : {mode_label}")
    print(f"Subjects   : {n_subjects}")
    print(f"Windows    : {n_windows}")
    print(f"Running LOSO ({n_subjects} runs)\n")

    loso_results = {}

    for run_idx, test_sid in enumerate(subject_ids):
        print(f"\n{'='*60}")
        print(f"Run {run_idx+1}/{n_subjects}  —  Held-out Subject {test_sid}  [{mode_label}]")
        print(f"{'='*60}")

        # Split windows by subject
        test_mask  = data["subject_ids"] == test_sid
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(test_mask)[0]

        n_tr_trials = len(np.unique(data["trial_ids"][train_idx]))
        n_vl_trials = len(np.unique(data["trial_ids"][val_idx]))
        vl_dist     = dict(sorted(
            Counter(data["att_idxs"][val_idx].tolist()).items()
        ))

        print(f"  Train : {n_subjects-1} subjects, "
              f"{n_tr_trials} trials, {len(train_idx)} windows")
        print(f"  Val   : Subject {test_sid}, "
              f"{n_vl_trials} trials, {len(val_idx)} windows")
        print(f"  Val speaker dist (0-based): {vl_dist}")

        tr_ds = AADDataset(data, train_idx, train=True)
        vl_ds = AADDataset(data, val_idx,   train=False)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        model     = _new_model(mode=mode)
        ckpt_path = os.path.join(results_dir,
                                 f"loso_subj{test_sid}_{mode}.pt")

        best_acc = train_model(
            model, tr_loader, vl_loader,
            epochs=epochs, ckpt_path=ckpt_path,
            lr=lr, label_smoothing=label_smoothing,
        )

        print(f"\n  → Subject {test_sid} test accuracy: {best_acc:.4f}")
        loso_results[int(test_sid)] = {
            "test_accuracy":  best_acc,
            "n_train_trials": n_tr_trials,
            "n_val_trials":   n_vl_trials,
            "n_train_windows": len(train_idx),
            "n_val_windows":   len(val_idx),
            "val_speaker_dist": vl_dist,
        }

    # Summary
    accs = [v["test_accuracy"] for v in loso_results.values()]
    summary = {
        "mode":           mode_label,
        "loso_results":   loso_results,
        "mean_accuracy":  float(np.mean(accs)),
        "std_accuracy":   float(np.std(accs)),
        "chance_level":   1 / N_SPEAKERS,
        "per_subject":    {str(sid): f"{acc:.4f}"
                           for sid, acc in zip(subject_ids, accs)},
    }

    print(f"\n{'='*60}")
    print(f"LOSO Summary  [{mode_label}]")
    print(f"  Per-subject : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± Std  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Chance      : {1/N_SPEAKERS:.4f}")
    print(f"{'='*60}")

    out_path = os.path.join(results_dir, f"loso_results_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


# ── dataset builder with subject IDs ─────────────────────────────────────────

def build_dataset_loso(local_path: str,
                       mode: str,
                       cache_dir: str = None) -> dict:
    """
    Build pooled dataset for all 16 subjects, including subject_ids array
    needed for LOSO splitting.
    """
    from pathlib import Path
    import json
    import pandas as pd
    from dataloader import (load_trial, N_SPEAKERS, VALID_MODES, WINDOW_SAMP)

    assert mode in VALID_MODES
    root = Path(local_path)

    trials_df    = pd.read_csv(root / "metadata" / "trials.csv")
    audio_meta   = json.loads((root / "metadata" / "audio_layout.json").read_text())
    audio_layout = audio_meta["speakers"]
    trials_df    = trials_df[trials_df["kind"] == "main"].copy()

    use_eeg   = mode in ("eeg",   "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi")
    use_video = mode in ("video", "eeg_video", "eeg_vg",   "eeg_vgi")
    use_gaze  = mode in ("gaze",  "gi",    "eeg_gaze", "eeg_vg", "eeg_vgi")
    use_imu   = mode in ("imu",   "gi",    "eeg_vgi")

    all_eeg         = [] if use_eeg   else None
    all_video       = [] if use_video else None
    all_gaze        = [] if use_gaze  else None
    all_imu         = [] if use_imu   else None
    all_audio       = [[] for _ in range(N_SPEAKERS)]
    all_att_idxs    = []
    all_trial_ids   = []
    all_subject_ids = []
    trial_meta      = []
    trial_id        = 0

    for s in range(1, 17):
        sid = f"S{s:02d}"
        for _, row in trials_df.iterrows():
            tid     = row["trial_id"]
            att_spk = int(row["attended_speaker"])

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
            all_trial_ids.append(np.full(n_win, trial_id, dtype=np.int64))
            all_subject_ids.append(np.full(n_win, s, dtype=np.int64))
            trial_meta.append({"trial_id": trial_id, "att_idx": att_spk - 1})
            trial_id += 1

        print(f"Subject {s}: loaded")

    def _cat(lst):
        if lst is None or not lst: return None
        lst = [x for x in lst if x is not None]
        return np.concatenate(lst, axis=0) if lst else None

    dataset = {
        "eeg":         _cat(all_eeg),
        "video":       _cat(all_video),
        "gaze":        _cat(all_gaze),
        "imu":         _cat(all_imu),
        "audio":       [np.concatenate(all_audio[i], axis=0) for i in range(N_SPEAKERS)],
        "att_idxs":    np.concatenate(all_att_idxs,    axis=0),
        "trial_ids":   np.concatenate(all_trial_ids,   axis=0),
        "subject_ids": np.concatenate(all_subject_ids, axis=0),
        "trial_meta_ids":     np.array([t["trial_id"] for t in trial_meta], dtype=np.int64),
        "trial_meta_att_idx": np.array([t["att_idx"]  for t in trial_meta], dtype=np.int64),
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal ({mode}): {n_trials} trials, {n_windows} windows")
    print(f"Subjects: {sorted(np.unique(dataset['subject_ids']).tolist())}")
    return dataset


# ── CLI # ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO evaluation for 4-speaker AAD"
    )
    p.add_argument("--local_path",      default='maestro-eeg-dataset',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",       default=None,
                   help="Directory to cache preprocessed video/gaze/IMU features. "
                        "First run computes and saves; subsequent runs load instantly.")
    p.add_argument("--mode",            choices=LOSO_MODES,
                   default="eeg",
                   help=f"One of: {LOSO_MODES}")
    p.add_argument("--results",         default="results_loso")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"Mode   : {args.mode}  ({MODE_LABELS[args.mode]})")
    if args.cache_dir:
        print(f"Cache  : {args.cache_dir}")

    print("\nLoading dataset...")
    data = build_dataset_loso(
        local_path = args.local_path,
        mode       = args.mode,
        cache_dir  = args.cache_dir,
    )

    run_loso(
        data            = data,
        results_dir     = args.results,
        mode            = args.mode,
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
        label_smoothing = args.label_smoothing,
    )