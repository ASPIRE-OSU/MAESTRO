"""
train_hemisphere.py — T2 (Attended Hemisphere), runs under EITHER of
the dataset's two official split protocols.

--split_setting loso   : subject-generalization (16 folds)
--split_setting within : content-generalization, pooled across subjects (5 folds)

See train_aad.py's module docstring for the full explanation of what
each protocol controls for; this file mirrors that same pattern for the
binary hemisphere task.
"""

import os
import json
import argparse
import zlib

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from dataloader import (build_dataset, load_official_splits,
                        get_official_split_windows, carve_inner_val,
                        carve_inner_val_content, compute_global_content_holdout,
                        N_EEG_CH, N_VIDEO_CH, N_GAZE_CH, N_IMU_CH,
                        N_SPEAKERS, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)
from model_spatial import AADModel


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
    bug: per-fold reseeding alone is not sufficient when two DIFFERENT
    modes share a fold and produce architecturally IDENTICAL model
    shapes (e.g. gaze and imu are both 6-channel single modalities).
    Uses zlib.crc32 rather than Python's builtin hash(), which is
    randomized per-process (PYTHONHASHSEED) and would silently break
    run-to-run reproducibility.
    """
    return base_seed + (zlib.crc32(mode.encode()) % 10_000)

# T2 hemisphere label mapping (0-based speaker index → binary label)
# S1(0)=left, S2(1)=left, S3(2)=right, S4(3)=right
HEMISPHERE_LABEL = {0: 0, 1: 0, 2: 1, 3: 1}

N_CLASSES = 2   # binary: left=0, right=1

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


# ── label helpers ─────────────────────────────────────────────────────────────

def get_hemisphere_labels(att_idxs: np.ndarray) -> np.ndarray:
    return np.array([HEMISPHERE_LABEL[i] for i in att_idxs], dtype=np.int64)


def group_audio_hemisphere(audio: list) -> list:
    env_left  = (audio[0] + audio[1]) / 2.0
    env_right = (audio[2] + audio[3]) / 2.0
    return [env_left, env_right]


# ── dataset ───────────────────────────────────────────────────────────────────

class HemisphereDataset(Dataset):
    """
    Dataset for T2 hemisphere binary decoding.
    Returns (eeg, video, gaze, imu, [env_left, env_right], label)
    where label ∈ {0=left, 1=right}. No speaker randomisation — S1–S4
    always map to their fixed hemisphere.
    """

    def __init__(self, data: dict, window_idx: np.ndarray, train: bool = True):
        idx        = window_idx
        self.train = train

        self.eeg   = torch.from_numpy(data["eeg"][idx])   if data["eeg"]   is not None else None
        self.video = torch.from_numpy(data["video"][idx]) if data["video"] is not None else None
        self.gaze  = torch.from_numpy(data["gaze"][idx])  if data["gaze"]  is not None else None
        self.imu   = torch.from_numpy(data["imu"][idx])   if data["imu"]   is not None else None

        self.audio = [torch.from_numpy(data["audio"][i][idx])
                      for i in range(N_SPEAKERS)]

        hem_labels  = get_hemisphere_labels(data["att_idxs"][idx])
        self.labels = torch.from_numpy(
            np.eye(N_CLASSES, dtype=np.float32)[hem_labels]
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
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

def _new_model(mode: str, seed: int) -> AADModel:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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


# ── training loop (inner-val for selection ONLY) ────────────────────────────────

def train_model(model, train_loader, val_loader,
                epochs, ckpt_path, lr=1e-4, label_smoothing=0.1):
    """Trains with train_loader/val_loader for checkpoint SELECTION
    ONLY. Returns best_inner_val_acc, NOT a final reportable number."""
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
              f"inner_val_loss={vl_loss:.4f} inner_val_acc={vl_acc:.4f}")

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


# ── official splits runner ──────────────────────────────────────────────────────

def run_official_splits(data: dict, results_dir: str, mode: str = "eeg",
                        split_setting: str = "within", splits_dir: str = "splits",
                        epochs: int = 50, batch_size: int = 32,
                        lr: float = 1e-4, label_smoothing: float = 0.1,
                        inner_val_frac: float = 0.2,
                        held_out_content_frac: float = 0.2):
    """
    Runs T2 hemisphere decoding under the dataset's official split
    protocol (loso or within), reading splits_dir/{split_setting}/
    fold_*.json directly as the authoritative train/test definition.

    For loso specifically: the OFFICIAL protocol only excludes the held-
    out subject's IDENTITY -- its test set is "all trials of the held-
    out subject", so trial CONTENT is still shared with the 15 training
    subjects. A GLOBAL content holdout (computed ONCE, same across all
    16 folds) is layered on top so the held-out subject is novel in
    BOTH identity and content, matching train_aad.py's treatment.
    """
    from collections import Counter
    os.makedirs(results_dir, exist_ok=True)
    mode_label = MODE_LABELS[mode]

    folds = load_official_splits(splits_dir, split_setting)
    print(f"\nTask    : T2 Hemisphere (left vs right)")
    print(f"Mode    : {mode_label}")
    print(f"Protocol: {split_setting}  ({len(folds)} official folds)")

    if split_setting == "loso":
        train_content_set, heldout_content_set = compute_global_content_holdout(
            data, held_out_content_frac=held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): "
              f"{len(train_content_set)} train-content trials, "
              f"{len(heldout_content_set)} held-out-content trials")

    fold_results = {}
    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if split_setting == "loso":
            win_content = data["trial_meta_tid"][
                np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]
            is_train_content   = np.isin(win_content, list(train_content_set))
            is_heldout_content = np.isin(win_content, list(heldout_content_set))

            tr_idx = tr_idx[is_train_content[tr_idx]]
            te_idx = te_idx[is_heldout_content[te_idx]]

            inner_tr_idx, inner_vl_idx = carve_inner_val(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)
        else:
            inner_tr_idx, inner_vl_idx = carve_inner_val_content(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)

        te_hem  = get_hemisphere_labels(data["att_idxs"][te_idx])
        te_dist = dict(sorted(Counter(te_hem.tolist()).items()))

        print(f"\n{'='*60}")
        print(f"Fold {fold_num}  [{split_setting}, T2 Hemisphere — {mode_label}]")
        print(f"  Inner train : {len(inner_tr_idx)} windows")
        print(f"  Inner val   : {len(inner_vl_idx)} windows")
        print(f"  TEST (official) : {len(te_idx)} windows, dist (0=left,1=right): {te_dist}")
        print(f"{'='*60}")

        tr_ds = HemisphereDataset(data, inner_tr_idx, train=True)
        vl_ds = HemisphereDataset(data, inner_vl_idx, train=False)
        te_ds = HemisphereDataset(data, te_idx,       train=False)

        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        te_loader = DataLoader(te_ds, batch_size=batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        model     = _new_model(mode=mode, seed=_mode_seed(SEED + fold_num, mode))
        ckpt_path = os.path.join(results_dir, f"fold_{fold_num}_{mode}_hemisphere_{split_setting}.pt")

        best_inner_val_acc = train_model(
            model, tr_loader, vl_loader,
            epochs=epochs, ckpt_path=ckpt_path,
            lr=lr, label_smoothing=label_smoothing,
        )
        test_acc = evaluate_test(model, te_loader)

        print(f"\n  → Fold {fold_num} best inner_val: {best_inner_val_acc:.4f} "
              f"| TEST acc (reported): {test_acc:.4f}")
        fold_results[fold_num] = {
            "test_accuracy":      test_acc,
            "best_inner_val_acc": best_inner_val_acc,
            "n_inner_train_windows": len(inner_tr_idx),
            "n_inner_val_windows":   len(inner_vl_idx),
            "n_test_windows":        len(te_idx),
            "test_hem_dist":         te_dist,
        }

    accs = [v["test_accuracy"] for v in fold_results.values()]
    summary = {
        "task":          "T2_hemisphere",
        "mode":          mode_label,
        "split_setting": split_setting,
        "folds":         fold_results,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
        "chance_level":  0.5,
        "note": "mean_accuracy computed from official-test TEST accuracy per fold, "
               "never the inner-val score used for checkpoint selection.",
    }

    print(f"\n{'='*60}")
    print(f"{split_setting.upper()} Summary  [T2 Hemisphere — {mode_label}]")
    print(f"  Per-fold : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean±Std : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Chance   : 0.5000")
    print(f"{'='*60}")

    out_path = os.path.join(results_dir, f"results_{mode}_hemisphere_{split_setting}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to {out_path}")

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="T2 Hemisphere binary AAD decoding")
    p.add_argument("--local_path",      default='maestro')
    p.add_argument("--cache_dir",       default='cache')
    p.add_argument("--mode",            choices=VALID_MODES, default="eeg")
    p.add_argument("--results",         default="results_hemisphere")
    p.add_argument("--split_setting",   choices=["loso", "within"], default="within",
                   help="Which official split protocol to use (default: within)")
    p.add_argument("--splits_dir",      default=None,
                   help="Path to the dataset's splits/ folder. "
                        "Defaults to <local_path>/splits (i.e. nested "
                        "inside the dataset root, matching the real "
                        "dataset layout) if not given explicitly.")
    p.add_argument("--epochs",          type=int,   default=50)
    p.add_argument("--batch_size",      type=int,   default=32)
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--inner_val_frac",  type=float, default=0.2)
    p.add_argument("--held_out_content_frac", type=float, default=0.2,
                   help="LOSO ONLY: fraction of trial CONTENT held out "
                        "globally, layered on top of the official "
                        "subject-based loso split (default 0.2). "
                        "Ignored for --split_setting within.")
    p.add_argument("--window_sec",      type=float, default=None)
    p.add_argument("--hop_sec",         type=float, default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.local_path, "splits")
    print(f"Device : {DEVICE}")
    print(f"Seed   : {SEED}")
    print(f"Task   : T2 Hemisphere (left vs right)  — chance=0.5")
    print(f"Mode   : {args.mode}  ({MODE_LABELS[args.mode]})")
    print(f"Split  : {args.split_setting}")

    print("\nLoading dataset...")
    kwargs = {}
    if args.window_sec is not None: kwargs["window_sec"] = args.window_sec
    if args.hop_sec    is not None: kwargs["hop_sec"]    = args.hop_sec
    data = build_dataset(local_path=args.local_path, mode=args.mode,
                         cache_dir=args.cache_dir, **kwargs)

    window_sec_eff = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec_eff    = args.hop_sec    if args.hop_sec    is not None else window_sec_eff
    results_dir = (f"{args.results}_{args.split_setting}"
                   f"_w{window_sec_eff:g}_h{hop_sec_eff:g}")
    run_official_splits(
        data=data, results_dir=results_dir, mode=args.mode,
        split_setting=args.split_setting, splits_dir=args.splits_dir,
        epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr, label_smoothing=args.label_smoothing,
        inner_val_frac=args.inner_val_frac,
        held_out_content_frac=args.held_out_content_frac,
    )