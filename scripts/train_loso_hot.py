"""
train_loso_hot.py
-------------
Leave-One-Subject-Out (LOSO) evaluation for 4-speaker AAD, combined with a
held-out trial split.
"""

import os
import json
import argparse
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        N_SPEAKERS, VALID_MODES)
from model_classification import AADModel


# ── constants ─────────────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _mode_seed(base_seed: int, mode: str) -> int:
    """
    Deterministic mode-dependent seed offset, fixing a seed COLLISION
    bug: per-fold reseeding alone (SEED + run_idx) is not sufficient
    when two DIFFERENT modes share a fold/subject and produce
    architecturally IDENTICAL model shapes (e.g. gaze and imu are both
    6-channel single modalities) — both runs would get identical
    initial weights, identical DataLoader shuffle order, and identical
    per-batch speaker-permutation draws, differing only in the numeric
    content of the input tensors. This was empirically confirmed to
    produce bit-identical outputs (1.0000 prediction agreement,
    identical confusion matrices) between independently-trained gaze
    and imu models sharing a fold. Uses zlib.crc32 rather than Python's
    builtin hash(), which is randomized per-process (PYTHONHASHSEED)
    and would silently break run-to-run reproducibility.
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


# ── helpers ───────────────────────────────────────────────────────────────────

def _new_model(mode: str, seed: int) -> AADModel:
    """
    Build a fresh model, reseeding immediately beforehand so this fold's
    weight initialization is independent of however many prior folds have
    already run in this process. `seed` should vary per fold (e.g.
    SEED + run_idx) so folds don't all get identical initial weights,
    while still being fully reproducible run-to-run.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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
    Val set = held-out subject on held-out trial content (true test set).
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


# ── trial-content split ────────────────────────────────────────────────────────

def split_trial_tids(trial_tids: np.ndarray,
                     trial_att_labels: np.ndarray,
                     held_out_trial_frac: float,
                     seed: int = SEED) -> tuple:
    """
    Stratified split of unique trial-content IDs (e.g. 'eval_014') into
    train_tids and heldout_tids, stratified by attended speaker label.

    This split is computed ONCE and applied identically across all 16 LOSO
    folds, so that the same trial content is withheld from training
    regardless of which subject is held out.

    Parameters
    ----------
    trial_tids        : (n_unique_trials,) array of unique tid strings
    trial_att_labels  : (n_unique_trials,) attended-speaker label per tid,
                        used only for stratification
    held_out_trial_frac : fraction of trial content to reserve as heldout

    Returns
    -------
    train_tids, heldout_tids : disjoint arrays of tid strings
    """
    from sklearn.model_selection import train_test_split
    train_tids, heldout_tids = train_test_split(
        trial_tids,
        test_size=held_out_trial_frac,
        stratify=trial_att_labels,
        random_state=seed,
    )
    return np.array(train_tids), np.array(heldout_tids)


# ── LOSO ──────────────────────────────────────────────────────────────────────

def run_loso(data: dict,
             results_dir: str,
             mode: str       = "eeg",
             epochs: int     = 50,
             batch_size: int = 32,
             lr: float       = 1e-4,
             label_smoothing: float = 0.1,
             held_out_trial_frac: float = 0.2):
    """
    Leave-One-Subject-Out evaluation, combined with a held-out trial split.

    Parameters
    ----------
    data        : dict from build_dataset_loso() — must contain
                  "subject_ids" and "trial_tids" keys
    results_dir : directory for checkpoints and result JSON
    mode        : one of VALID_MODES
    held_out_trial_frac : fraction of trial content reserved as heldout,
                  never trained on by any subject (default 0.2)
    """
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)

    mode_label  = MODE_LABELS[mode]
    subject_ids = np.unique(data["subject_ids"])
    n_subjects  = len(subject_ids)
    n_windows   = len(data["audio"][0])

    # ── Compute the trial-content split ONCE, shared across all folds ──────────
    unique_tids = np.unique(data["trial_tids"])
    # attended-speaker label per unique tid, for stratification
    tid_to_att = {}
    for tid, att in zip(data["trial_tids"], data["att_idxs"]):
        tid_to_att.setdefault(tid, att)
    att_per_tid = np.array([tid_to_att[t] for t in unique_tids])

    train_tids, heldout_tids = split_trial_tids(
        unique_tids, att_per_tid, held_out_trial_frac, seed=SEED)

    print(f"\nMode                : {mode_label}")
    print(f"Subjects             : {n_subjects}")
    print(f"Windows              : {n_windows}")
    print(f"Unique trial content : {len(unique_tids)}")
    print(f"  Train-content tids : {len(train_tids)} "
          f"({100*(1-held_out_trial_frac):.0f}%)")
    print(f"  Held-out tids      : {len(heldout_tids)} "
          f"({100*held_out_trial_frac:.0f}%) — never trained on by any subject")
    print(f"Running LOSO ({n_subjects} runs)\n")

    train_tid_set   = set(train_tids.tolist())
    heldout_tid_set = set(heldout_tids.tolist())
    is_train_tid    = np.array([t in train_tid_set   for t in data["trial_tids"]])
    is_heldout_tid  = np.array([t in heldout_tid_set for t in data["trial_tids"]])

    loso_results = {}

    for run_idx, test_sid in enumerate(subject_ids):
        print(f"\n{'='*60}")
        print(f"Run {run_idx+1}/{n_subjects}  —  Held-out Subject {test_sid}  [{mode_label}]")
        print(f"{'='*60}")

        # Train : other 15 subjects, restricted to train-content trials only
        train_mask = (data["subject_ids"] != test_sid) & is_train_tid
        # Val   : held-out subject, restricted to held-out-content trials only
        val_mask   = (data["subject_ids"] == test_sid) & is_heldout_tid

        train_idx = np.where(train_mask)[0]
        val_idx   = np.where(val_mask)[0]

        if len(val_idx) == 0:
            print(f"  WARNING: no held-out-trial windows for subject {test_sid} — skipping")
            continue

        n_tr_trials = len(np.unique(data["trial_ids"][train_idx]))
        n_vl_trials = len(np.unique(data["trial_ids"][val_idx]))
        vl_dist     = dict(sorted(
            Counter(data["att_idxs"][val_idx].tolist()).items()
        ))

        print(f"  Train : {n_subjects-1} subjects, train-content tids only, "
              f"{n_tr_trials} trials, {len(train_idx)} windows")
        print(f"  Val   : Subject {test_sid}, held-out-content tids only, "
              f"{n_vl_trials} trials, {len(val_idx)} windows")
        print(f"  Val speaker dist (0-based): {vl_dist}")

        tr_ds = AADDataset(data, train_idx, train=True)
        vl_ds = AADDataset(data, val_idx,   train=False)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        # Reseed per-fold AND per-mode (see _mode_seed) so every fold gets
        # an independent, reproducible initialization instead of
        # inheriting whatever RNG state prior folds happened to leave
        # behind, and so different modes sharing a fold/subject don't
        # collide.
        model     = _new_model(mode=mode, seed=_mode_seed(SEED + run_idx, mode))
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
        "mode":                 mode_label,
        "held_out_trial_frac":  held_out_trial_frac,
        "n_train_content_tids": len(train_tids),
        "n_heldout_content_tids": len(heldout_tids),
        "loso_results":         loso_results,
        "mean_accuracy":        float(np.mean(accs)),
        "std_accuracy":         float(np.std(accs)),
        "chance_level":         1 / N_SPEAKERS,
        "per_subject":          {str(sid): f"{acc:.4f}"
                                 for sid, acc in zip(loso_results.keys(), accs)},
    }

    print(f"\n{'='*60}")
    print(f"LOSO Summary  [{mode_label}]  (subject + trial content held out)")
    print(f"  Per-subject : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± Std  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Chance      : {1/N_SPEAKERS:.4f}")
    print(f"{'='*60}")

    out_path = os.path.join(results_dir, f"loso_results_{mode}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


# ── dataset builder with subject IDs AND trial content IDs ────────────────────

def build_dataset_loso(local_path: str,
                       mode: str,
                       cache_dir: str = None) -> dict:
    """
    Build pooled dataset for all 16 subjects, including subject_ids and
    trial_tids arrays needed for the combined subject + trial-content split.

    trial_tids stores the ORIGINAL trial content identifier (e.g. 'eval_014'
    from trials.csv), shared across all subjects who saw that trial — this
    is distinct from trial_ids, which is a unique integer per
    (subject, trial) recording instance used only for internal indexing.
    """
    from pathlib import Path
    import json
    import pandas as pd
    from dataloader import (load_trial, N_SPEAKERS, VALID_MODES, WINDOW_SAMP,
                            mode_uses)

    assert mode in VALID_MODES
    root = Path(local_path)

    trials_df    = pd.read_csv(root / "metadata" / "trials.csv")
    audio_meta   = json.loads((root / "metadata" / "audio_layout.json").read_text())
    audio_layout = audio_meta["speakers"]
    trials_df    = trials_df[trials_df["kind"] == "main"].copy()

    # Derived from dataloader's single source of truth, rather than a
    # separately-hardcoded set of tuples that could drift out of sync.
    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)

    all_eeg         = [] if use_eeg   else None
    all_video       = [] if use_video else None
    all_gaze        = [] if use_gaze  else None
    all_imu         = [] if use_imu   else None
    all_audio       = [[] for _ in range(N_SPEAKERS)]
    all_att_idxs    = []
    all_trial_ids   = []
    all_subject_ids = []
    all_trial_tids  = []     # original trial content ID per window
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
            all_trial_tids.append(np.full(n_win, tid, dtype=object))
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
        "trial_tids":  np.concatenate(all_trial_tids,  axis=0),
        "trial_meta_ids":     np.array([t["trial_id"] for t in trial_meta], dtype=np.int64),
        "trial_meta_att_idx": np.array([t["att_idx"]  for t in trial_meta], dtype=np.int64),
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal ({mode}): {n_trials} trials, {n_windows} windows")
    print(f"Subjects: {sorted(np.unique(dataset['subject_ids']).tolist())}")
    print(f"Unique trial content IDs: {len(np.unique(dataset['trial_tids']))}")
    return dataset


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="LOSO evaluation for 4-speaker AAD, "
                    "with combined subject + trial-content hold-out"
    )
    p.add_argument("--local_path",      default='maestro',
                   help="Root of the MAESTRO HuggingFace dataset")
    p.add_argument("--cache_dir",       default='cache',
                   help="Directory to cache preprocessed video/gaze/IMU features. "
                        "First run computes and saves; subsequent runs load instantly.")
    p.add_argument("--mode",            choices=VALID_MODES,
                   default="eeg",
                   help=f"One of: {VALID_MODES}")
    p.add_argument("--results",         default="results_loso")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--held_out_trial_frac", type=float, default=0.1,
                   help="Fraction of trial content reserved as heldout, "
                        "never trained on by any subject (default 0.2)")
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
        data                 = data,
        results_dir          = args.results,
        mode                 = args.mode,
        epochs               = args.epochs,
        batch_size           = args.batch_size,
        lr                   = args.lr,
        label_smoothing      = args.label_smoothing,
        held_out_trial_frac  = args.held_out_trial_frac,
    )