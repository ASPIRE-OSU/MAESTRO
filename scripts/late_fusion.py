"""
late_fusion.py
----------------
late fusion — combines the OUTPUT PROBABILITIES of
independently-trained single-modality models, rather than fusing
embeddings before a decision is made (contrast with the gated / fixed-
concatenation fusion in model_classification.py / model_spatial.py,
both of which fuse INSIDE the network, before the classification
decision).
"""

import os
import json
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloader import (build_dataset, AADDataset, collate_fn,
                        get_trial_level_splits, N_SPEAKERS, VALID_MODES,
                        mode_uses)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ALL_SINGLE_MODES = ["eeg", "gaze", "imu", "video"]

CKPT_PATTERN = {
    "pooled":       "fold_{fold}_{mode}.pt",
    "hemisphere":   "fold_{fold}_{mode}_hemisphere.pt",
    "eccentricity": "fold_{fold}_{mode}_eccentricity.pt",
    # LOSO uses a different pattern entirely (per-subject, not per-fold),
    # handled separately in run_loso_late_fusion() below.
}

# The 11 multi-modality combinations eligible for late fusion (excludes
# the 4 single modalities, since late-fusing one model is a no-op), used
# by --mode all to sweep everything in one invocation. Excludes legacy
# aliases (gi/eeg_vg/eeg_vgi) since they're redundant with their
# canonical equivalents (gaze_imu/eeg_gaze_video/eeg_gaze_imu_video).
MULTI_MODALITY_MODES = [
    "eeg_gaze", "eeg_imu", "eeg_video", "gaze_imu", "gaze_video", "imu_video",
    "eeg_gaze_imu", "eeg_gaze_video", "eeg_imu_video", "gaze_imu_video",
    "eeg_gaze_imu_video",
]


def _active_modalities(mode: str) -> list:
    """
    Which single modalities does `mode` late-fuse, in a fixed canonical
    order (eeg, gaze, imu, video) — derived from dataloader.mode_uses()
    so this stays in sync with the rest of the codebase's mode registry.
    """
    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
    active = []
    if use_eeg:   active.append("eeg")
    if use_gaze:  active.append("gaze")
    if use_imu:   active.append("imu")
    if use_video: active.append("video")
    if len(active) < 2:
        raise ValueError(
            f"mode='{mode}' activates only {active} — late fusion needs "
            f"at least 2 modalities to combine. Use the plain training "
            f"scripts directly for single-modality results.")
    return active


# ── task registry ──────────────────────────────────────────────────────────────

def _load_task(task: str):
    """
    Returns (ModelClass, label_map). label_map maps the raw 4-way
    attended-speaker index (0-3) to whatever label space this task uses.
    None for pooled/loso (4-class, no grouping); a dict for hemisphere/
    eccentricity's binary grouping, matching the exact mappings used in
    train_hemisphere.py / train_eccentricity.py.
    """
    if task in ("pooled", "loso"):
        from model_classification import AADModel
        return AADModel, None
    elif task == "hemisphere":
        from model_spatial import AADModel
        HEMISPHERE_LABEL = {0: 0, 1: 0, 2: 1, 3: 1}
        return AADModel, HEMISPHERE_LABEL
    elif task == "eccentricity":
        from model_spatial import AADModel
        ECCENTRICITY_LABEL = {0: 1, 1: 0, 2: 0, 3: 1}
        return AADModel, ECCENTRICITY_LABEL
    else:
        raise ValueError(f"Unknown task: {task}")


# ── late fusion combiner ────────────────────────────────────────────────────────

class LateFusionCombiner(nn.Module):
    """
    Learns one scalar weight per active modality (softmax-normalized)
    applied to that modality's ALREADY-COMPLETE probability vector. The
    single-modality models are kept FROZEN — only these combination
    weights are trained, and only on this fold/subject's own training
    split.
    """

    def __init__(self, n_modalities: int):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_modalities))

    def forward(self, probs_list: list) -> torch.Tensor:
        w = F.softmax(self.logits, dim=0)
        stacked = torch.stack(probs_list, dim=0)
        combined = (stacked * w.view(-1, 1, 1)).sum(dim=0)
        return combined / combined.sum(dim=1, keepdim=True)


# ── forward helpers ────────────────────────────────────────────────────────────

def _single_modality_forward(model, mode: str, eeg, video, gaze, imu, audio):
    e = eeg   if mode == "eeg"   else None
    v = video if mode == "video" else None
    g = gaze  if mode == "gaze"  else None
    i = imu   if mode == "imu"   else None
    return model(e, v, g, i, audio)


def _to(x):
    return x.to(DEVICE) if x is not None else None


def _group_labels(labels: torch.Tensor, label_map: dict) -> torch.Tensor:
    att_idx = labels.argmax(dim=1).cpu().numpy()
    grouped = np.array([label_map[i] for i in att_idx], dtype=np.int64)
    return torch.from_numpy(np.eye(2, dtype=np.float32)[grouped])


# ── evaluation ──────────────────────────────────────────────────────────────────

def evaluate_late_fusion(models: dict,
                         active_modalities: list,
                         loader: DataLoader,
                         label_map: dict = None,
                         combiner: LateFusionCombiner = None) -> float:
    for m in models.values():
        m.eval()
    if combiner is not None:
        combiner.eval()

    correct, total = 0, 0
    with torch.no_grad():
        for eeg, video, gaze, imu, audio, labels in loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE) if label_map is None \
                    else _group_labels(labels, label_map).to(DEVICE)

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


def train_combiner(models: dict,
                   active_modalities: list,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   label_map: dict = None,
                   epochs: int = 30,
                   lr: float = 1e-2) -> tuple:
    """Train ONLY the combiner's softmax weights on THIS fold/subject's
    training split; single-modality models stay frozen throughout."""
    for m in models.values():
        m.eval()
        for p in m.parameters():
            p.requires_grad = False

    combiner  = LateFusionCombiner(n_modalities=len(active_modalities)).to(DEVICE)
    optimizer = torch.optim.Adam(combiner.parameters(), lr=lr)

    best_val_acc = -1.0
    best_weights = None

    for epoch in range(1, epochs + 1):
        combiner.train()
        for eeg, video, gaze, imu, audio, labels in train_loader:
            eeg, video, gaze, imu = _to(eeg), _to(video), _to(gaze), _to(imu)
            audio  = [a.to(DEVICE) for a in audio]
            labels = labels.to(DEVICE) if label_map is None \
                    else _group_labels(labels, label_map).to(DEVICE)

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

        val_acc = evaluate_late_fusion(models, active_modalities, val_loader,
                                       label_map, combiner)
        w = F.softmax(combiner.logits, dim=0).detach().cpu().numpy().round(3)
        print(f"  Ep {epoch:03d} | weights({','.join(active_modalities)})={w} "
              f"| val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = combiner.state_dict()

    combiner.load_state_dict(best_weights)
    return combiner, best_val_acc


def _find_ckpt(ckpt_dir: str, task: str, fold: int, mode: str) -> str:
    pattern = CKPT_PATTERN[task]
    path = os.path.join(ckpt_dir, pattern.format(fold=fold, mode=mode))
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected checkpoint not found: {path}\n"
            f"Make sure you trained mode='{mode}' with {task} training "
            f"for all folds first.")
    return path


# ── task-specific dataset/collate loaders ───────────────────────────────────────

def _get_dataset_and_collate(task: str):
    """
    hemisphere/eccentricity group the 4 raw speaker envelopes into 2
    (left/right or inner/outer) BEFORE the model ever sees them, and
    their labels are already binary one-hot — using the generic 4-speaker
    AADDataset/collate_fn here would silently feed the model 4 raw
    envelopes instead of the 2 grouped ones it expects, and would need
    manual label grouping. Import each task's OWN dataset/collate classes
    instead, so audio grouping and label shape exactly match what the
    underlying single-modality checkpoints were actually trained on.
    """
    if task in ("pooled", "loso"):
        from dataloader import AADDataset, collate_fn
        return AADDataset, collate_fn
    elif task == "hemisphere":
        from train_hemisphere import HemisphereDataset, collate_fn
        return HemisphereDataset, collate_fn
    elif task == "eccentricity":
        from train_eccentricity import EccentricityDataset, collate_fn
        return EccentricityDataset, collate_fn
    else:
        raise ValueError(f"Unknown task: {task}")


# ── pooled / hemisphere / eccentricity (5-fold trial-level CV) ────────────────

def run_kfold_late_fusion(args, mode, active_modalities, ModelClass, label_map):
    DatasetClass, collate_fn = _get_dataset_and_collate(args.task)

    print("\nLoading dataset (all 4 modalities)...")
    data = build_dataset(local_path=args.local_path, mode="eeg_gaze_imu_video",
                         cache_dir=args.cache_dir)

    fold_accs = []
    for fold, tr_idx, vl_idx in get_trial_level_splits(
            data, n_splits=args.n_splits, seed=SEED):

        fold_num = fold + 1

        models = {}
        for m in active_modalities:
            ckpt_path = _find_ckpt(args.ckpt_dir, args.task, fold_num, m)
            model_m = ModelClass(mode=m).to(DEVICE)
            model_m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[m] = model_m

        # Task-specific dataset: for hemisphere/eccentricity this already
        # groups audio into 2 envelopes and returns binary one-hot labels,
        # so label_map is NOT applied again here (it would double-convert).
        tr_ds = DatasetClass(data, tr_idx, train=True)
        vl_ds = DatasetClass(data, vl_idx, train=False)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        print(f"\n{'='*60}\n[{mode}] Fold {fold_num}/{args.n_splits}\n{'='*60}")

        # label_map already baked into the dataset for hemisphere/eccentricity,
        # so pass None here regardless of task to avoid double-grouping.
        if args.combine == "mean":
            acc = evaluate_late_fusion(models, active_modalities, vl_loader, None)
            print(f"  Mean-combine val accuracy: {acc:.4f}")
        else:
            combiner, acc = train_combiner(
                models, active_modalities, tr_loader, vl_loader, None,
                epochs=args.epochs)
            final_w = F.softmax(combiner.logits, dim=0).detach().cpu().numpy()
            print(f"  Learned combiner val accuracy: {acc:.4f}")
            print(f"  Final weights ({','.join(active_modalities)}): {final_w.round(3)}")

        fold_accs.append(acc)

    return fold_accs


# ── LOSO (leave-one-subject-out + held-out trial content) ──────────────────────

def run_loso_late_fusion(args, mode, active_modalities, ModelClass, label_map):
    """
    Reuses train_loso.py's OWN build_dataset_loso() and split_trial_tids()
    so the exact same train/val split (subject held out AND trial content
    held out) is used here as was used to produce the checkpoints being
    loaded — no risk of a subtly different split introducing leakage or
    an unfair comparison.
    """
    from train_loso_hot import build_dataset_loso, split_trial_tids

    print("\nLoading dataset (all 4 modalities, LOSO)...")
    data = build_dataset_loso(local_path=args.local_path, mode="eeg_gaze_imu_video",
                              cache_dir=args.cache_dir)

    subject_ids = np.unique(data["subject_ids"])
    unique_tids = np.unique(data["trial_tids"])
    tid_to_att  = {}
    for tid, att in zip(data["trial_tids"], data["att_idxs"]):
        tid_to_att.setdefault(tid, att)
    att_per_tid = np.array([tid_to_att[t] for t in unique_tids])

    train_tids, heldout_tids = split_trial_tids(
        unique_tids, att_per_tid, args.held_out_trial_frac, seed=SEED)
    train_tid_set   = set(train_tids.tolist())
    heldout_tid_set = set(heldout_tids.tolist())
    is_train_tid    = np.array([t in train_tid_set   for t in data["trial_tids"]])
    is_heldout_tid  = np.array([t in heldout_tid_set for t in data["trial_tids"]])

    subj_accs = []
    for test_sid in subject_ids:
        print(f"\n{'='*60}\n[{mode}] Held-out Subject {test_sid}\n{'='*60}")

        models = {}
        for m in active_modalities:
            ckpt_path = os.path.join(
                args.ckpt_dir, f"loso_subj{test_sid}_{m}.pt")
            if not os.path.exists(ckpt_path):
                raise FileNotFoundError(
                    f"Expected LOSO checkpoint not found: {ckpt_path}\n"
                    f"Make sure you trained mode='{m}' with train_loso.py "
                    f"for all 16 subjects first.")
            model_m = ModelClass(mode=m).to(DEVICE)
            model_m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
            models[m] = model_m

        train_mask = (data["subject_ids"] != test_sid) & is_train_tid
        val_mask   = (data["subject_ids"] == test_sid) & is_heldout_tid
        train_idx  = np.where(train_mask)[0]
        val_idx    = np.where(val_mask)[0]

        if len(val_idx) == 0:
            print(f"  WARNING: no held-out-trial windows for subject {test_sid} — skipping")
            continue

        tr_ds = AADDataset(data, train_idx, train=True)
        vl_ds = AADDataset(data, val_idx,   train=False)
        tr_loader = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                               collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(vl_ds, batch_size=args.batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        if args.combine == "mean":
            acc = evaluate_late_fusion(models, active_modalities, vl_loader, label_map)
            print(f"  Mean-combine test accuracy: {acc:.4f}")
        else:
            combiner, acc = train_combiner(
                models, active_modalities, tr_loader, vl_loader, label_map,
                epochs=args.epochs)
            final_w = F.softmax(combiner.logits, dim=0).detach().cpu().numpy()
            print(f"  Learned combiner test accuracy: {acc:.4f}")
            print(f"  Final weights ({','.join(active_modalities)}): {final_w.round(3)}")

        subj_accs.append(acc)

    return subj_accs


# ── single-mode runner ───────────────────────────────────────────────────────────

def run_one_mode(args, mode: str) -> dict:
    """Run late fusion for exactly one multi-modality mode, return its summary dict."""
    active_modalities = _active_modalities(mode)
    ModelClass, label_map = _load_task(args.task)

    print(f"\n{'#'*60}")
    print(f"# Mode: {mode}  (modalities: {active_modalities})")
    print(f"{'#'*60}")

    if args.task == "loso":
        accs = run_loso_late_fusion(args, mode, active_modalities, ModelClass, label_map)
    else:
        accs = run_kfold_late_fusion(args, mode, active_modalities, ModelClass, label_map)

    summary = {
        "task": args.task, "mode": mode, "combine": args.combine,
        "active_modalities": active_modalities,
        "accuracies": accs,
        "mean_accuracy": float(np.mean(accs)),
        "std_accuracy":  float(np.std(accs)),
    }
    print(f"\n{'='*60}")
    print(f"Late Fusion Summary [{args.task}, mode={mode}, {args.combine}]  (no leakage)")
    print(f"  Per-fold/subject : {[f'{a:.4f}' for a in accs]}")
    print(f"  Mean ± Std       : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"{'='*60}")

    out_path = os.path.join(args.results, f"late_fusion_{args.task}_{mode}_{args.combine}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out_path}")

    return summary


RESULT_FILE_PATTERN = {
    "pooled":       ("kfold_results_{mode}.json",       "mean_accuracy", "std_accuracy"),
    "loso":         ("loso_results_{mode}.json",        "mean_accuracy", "std_accuracy"),
    "hemisphere":   ("hemisphere_results_{mode}.json",  "mean_accuracy", "std_accuracy"),
    "eccentricity": ("eccentricity_results_{mode}.json","mean_accuracy", "std_accuracy"),
}


def _load_single_modality_result(task: str, ckpt_dir: str, mode: str) -> dict:
    """
    Read a single modality's ALREADY-COMPUTED result JSON (produced by
    train_pooled.py / train_loso.py / train_hemisphere.py /
    train_eccentricity.py when that single modality was trained) rather
    than re-running anything. Returns None if the file isn't found, so
    the summary can note it's missing instead of failing outright.
    """
    fname_pattern, mean_key, std_key = RESULT_FILE_PATTERN[task]
    path = os.path.join(ckpt_dir, fname_pattern.format(mode=mode))
    if not os.path.exists(path):
        return None
    with open(path) as f:
        r = json.load(f)
    return {
        "task": task, "mode": mode, "combine": "single (no fusion)",
        "active_modalities": [mode],
        "mean_accuracy": r[mean_key],
        "std_accuracy":  r[std_key],
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Late fusion (Option 2), fold/subject-matched")
    p.add_argument("--task", default='pooled',
                   choices=["pooled", "loso", "hemisphere", "eccentricity"])
    p.add_argument("--mode", default="all",
                   help="Which multi-modality combination to late-fuse "
                        "(any of dataloader.VALID_MODES with 2+ active "
                        "modalities), or 'all' to sweep all 11 multi-"
                        "modality combinations plus report the 4 single-"
                        "modality results (read from their existing "
                        "result JSONs, not re-trained) in one combined "
                        "summary file.")
    p.add_argument("--local_path", default="maestro")
    p.add_argument("--cache_dir",  default="cache")
    p.add_argument("--ckpt_dir",   default='results_pooled',
                   help="Directory containing the single-modality checkpoints "
                        "AND their result JSONs for this task "
                        "(fold_k_{mode}.pt + kfold_results_{mode}.json, or "
                        "loso_subj{sid}_{mode}.pt + loso_results_{mode}.json "
                        "for --task loso)")
    p.add_argument("--combine",    choices=["mean", "learned"], default="learned")
    p.add_argument("--n_splits",   type=int, default=5)
    p.add_argument("--epochs",     type=int, default=30)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--held_out_trial_frac", type=float, default=0.2,
                   help="Only used for --task loso, must match the value "
                        "used when the LOSO checkpoints were trained")
    p.add_argument("--results",    default="results_late_fusion")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.results, exist_ok=True)

    print(f"Task     : {args.task}")
    print(f"Combine  : {args.combine}")
    print(f"Device   : {DEVICE}")

    if args.mode == "all":
        print(f"Mode     : ALL {len(MULTI_MODALITY_MODES)} multi-modality "
              f"combinations + {len(ALL_SINGLE_MODES)} single modalities\n")
        all_summaries = {}

        # Single modalities: read existing results, no training/inference.
        print(f"{'#'*60}\n# Single-modality results (read from existing JSONs)\n{'#'*60}")
        for mode in ALL_SINGLE_MODES:
            single = _load_single_modality_result(args.task, args.ckpt_dir, mode)
            if single is None:
                print(f"  {mode:22s} -> result JSON not found, skipping "
                      f"(train it first if you want it in the summary)")
                continue
            print(f"  {mode:22s} -> mean={single['mean_accuracy']:.4f} "
                  f"± {single['std_accuracy']:.4f}")
            all_summaries[mode] = single

        # Multi-modality combinations: actually run late fusion.
        for mode in MULTI_MODALITY_MODES:
            summary = run_one_mode(args, mode)
            all_summaries[mode] = summary

        print(f"\n{'#'*60}")
        print(f"# FULL SUMMARY [{args.task}, {args.combine}]  "
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
            args.results, f"late_fusion_{args.task}_ALL_{args.combine}.json")
        with open(combined_path, "w") as f:
            json.dump(all_summaries, f, indent=2)
        print(f"\nFull combined summary (singles + late fusion) saved to {combined_path}")
    else:
        run_one_mode(args, args.mode)