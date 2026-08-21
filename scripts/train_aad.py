"""
train_aad.py — attended-talker decoding (T1), under either official split.

--split_setting loso   : listener generalisation (16 folds, one held-out
                         listener each, splits/loso/fold_*.json), with a global
                         stimulus-content holdout layered on top so the held-out
                         listener is novel in identity AND content.
--split_setting within : content generalisation, pooled across all 16 listeners
                         (5 folds, splits/within/fold_*.json, setting "intra").

WHAT CHANGED, AND WHY IT MATTERS FOR ANY NUMBER THIS SCRIPT PRINTS
------------------------------------------------------------------
The previous revision reported ~0.50 four-way accuracy against a 0.25 chance
level.  That number was not attributable to the physiological recording:
permuting the recordings across test windows changed it by 0.0009, and feeding
zeros instead changed it by nothing.  Three changes here make the reported
number mean what it appears to mean.

1. CANDIDATES.  `--candidates qmatch` (default) equalises the four candidates'
   amplitude distributions, removing an acoustic cue that identified the
   attended talker without any recording.  `--candidates raw` reproduces the
   previous, confounded construction and is retained only for that purpose.
   The audio-only probe printed at startup certifies the choice BEFORE training:
   it must land near 1/K.

2. OBJECTIVE.  The loss now contains the permutation control itself, so a
   decoder that ignores the recording cannot reach a low training loss.  See
   `losses.py`.

3. SELECTION.  The retained checkpoint maximises (validation accuracy - validation
   permuted accuracy), not validation accuracy.  Where a shortcut exists,
   accuracy selection picks the epoch that exploits it best; once the candidates
   are clean the two criteria agree to within 0.005, so this is a safeguard that
   is inactive on clean data and cannot manufacture an effect.

Every fold reports accuracy, the permutation null, and their difference.  Report
all three.  An accuracy on its own does not distinguish a decoder that reads the
recording from one that reads the candidates.
"""

import argparse
import json
import os
import zlib
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataloader import (build_dataset, build_dataset_cached, AADDataset, collate_fn,
                        SubjectBatchSampler, make_candidate_bank,
                        audio_only_probe, load_official_splits,
                        get_official_split_windows, carve_inner_val,
                        carve_inner_val_content, compute_global_content_holdout,
                        subject_per_window, content_per_window,
                        position_in_trial, N_SPEAKERS, VALID_MODES,
                        WINDOW_SEC as dl_WINDOW_SEC)
from model_classification import AADModel
from losses import total_loss, DEFAULTS as LOSS_DEFAULTS
from evaluation import Evaluator

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODE_LABELS = {
    "eeg": "EEG only", "gaze": "Gaze only", "imu": "IMU only",
    "video": "Video only", "eeg_gaze": "EEG+Gaze", "eeg_imu": "EEG+IMU",
    "eeg_video": "EEG+Video", "gaze_imu": "Gaze+IMU",
    "gaze_video": "Gaze+Video", "imu_video": "IMU+Video",
    "eeg_gaze_imu": "EEG+Gaze+IMU", "eeg_gaze_video": "EEG+Gaze+Video",
    "eeg_imu_video": "EEG+IMU+Video", "gaze_imu_video": "Gaze+IMU+Video",
    "eeg_gaze_imu_video": "EEG+Gaze+IMU+Video", "gi": "Gaze+IMU",
    "eeg_vg": "EEG+Gaze+Video", "eeg_vgi": "EEG+Gaze+IMU+Video",
}


def _mode_seed(base_seed: int, mode: str) -> int:
    """Deterministic mode-dependent seed offset.  Per-fold reseeding alone is
    not sufficient when two different modes produce architecturally identical
    models (gaze and imu are both 6-channel).  zlib.crc32 rather than hash(),
    which is randomised per process."""
    return base_seed + (zlib.crc32(mode.encode()) % 10_000)


def _to(x):
    return x.to(DEVICE) if x is not None else None


def run_epoch(model, loader, head, optimizer=None, loss_cfg=None, train=True):
    model.train(train)
    total, nb = 0.0, 0
    with torch.set_grad_enabled(train):
        for eeg, video, gaze, imu, audio, label, spk_of_slot, att, subj in loader:
            eeg, video = _to(eeg), _to(video)
            gaze, imu = _to(gaze), _to(imu)
            audio = [a.to(DEVICE) for a in audio]
            label, att = label.to(DEVICE), att.to(DEVICE)
            spk_of_slot, subj = spk_of_slot.to(DEVICE), subj.to(DEVICE)

            out = model(eeg=eeg, video=video, gaze=gaze, imu=imu,
                        audio=audio if model.couple_mod else None,
                        spk_of_slot=spk_of_slot)
            loss, _ = total_loss(out, label, subj, head, loss_cfg, spk_label=att)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += float(loss.detach()); nb += 1
    return total / max(nb, 1)


def train_model(model, tr_loader, vl_loader, epochs, ckpt_path,
                lr=1e-3, weight_decay=1e-4, patience=12, loss_cfg=None,
                val_shuffles=3, verbose_every=5):
    """Trains, selecting the checkpoint on the CONTRIBUTION (validation accuracy
    minus validation permuted accuracy) rather than on validation accuracy.
    Returns the best validation contribution -- NOT a reportable number."""
    head = getattr(model, "head", None)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-6)

    best, bad = -9e9, 0
    for epoch in range(1, epochs + 1):
        tr_loss = run_epoch(model, tr_loader, head, opt, loss_cfg, True)
        b = Evaluator(model, vl_loader, DEVICE).battery(n_shuffle=val_shuffles,
                                                        seed=7)
        sch.step(b["contribution"])
        if b["contribution"] > best:
            best = b["contribution"]
            torch.save(model.state_dict(), ckpt_path)
            bad = 0
        else:
            bad += 1
        if epoch % verbose_every == 0 or epoch == 1:
            print(f"  Ep {epoch:03d} | loss={tr_loss:.4f} "
                  f"val_acc={b['accuracy']:.4f} val_null={b['null_mean']:.4f} "
                  f"val_contribution={b['contribution']:+.4f}", flush=True)
        if bad >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
    model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    return best


def run_official_splits(data, results_dir, mode="eeg", split_setting="within",
                        splits_dir="splits", epochs=50, batch_size=32, lr=1e-3,
                        inner_val_frac=0.2, held_out_content_frac=0.2,
                        candidates="qmatch", n_cand=N_SPEAKERS,
                        window_sec=None, hop_sec=None, loss_cfg=None,
                        test_shuffles=20, val_shuffles=3):
    os.makedirs(results_dir, exist_ok=True)
    label = MODE_LABELS[mode]

    # ── certify the candidate set BEFORE training ─────────────────────────────
    bank = make_candidate_bank(data, candidates, window_sec or dl_WINDOW_SEC,
                               hop_sec, n_cand=n_cand, seed=SEED)
    probe = audio_only_probe(bank, content_per_window(data))
    chance = 1.0 / n_cand
    print(f"\n[audio-only probe] construction='{candidates}' K={n_cand}: "
          f"{probe:.4f} (chance {chance:.4f}, excess {probe - chance:+.4f})")
    if probe - chance > 0.05:
        print("  WARNING: the candidate set is acoustically confounded. Any "
              "accuracy below is partly obtainable without the recording. Use "
              "--candidates qmatch or shifted_qm.")

    folds = load_official_splits(splits_dir, split_setting)
    print(f"\nMode: {label} | Protocol: {split_setting} | {len(folds)} folds")

    if split_setting == "loso":
        train_content, heldout_content = compute_global_content_holdout(
            data, held_out_content_frac=held_out_content_frac, seed=SEED)
        print(f"Global content holdout (loso only): {len(train_content)} train / "
              f"{len(heldout_content)} held-out content trials")

    win_content = content_per_window(data)
    win_position = position_in_trial(data)
    fold_results = {}

    for fold_info in folds:
        fold_num = fold_info["fold"]
        tr_idx, te_idx = get_official_split_windows(data, fold_info)

        if split_setting == "loso":
            tr_idx = tr_idx[np.isin(win_content[tr_idx], list(train_content))]
            te_idx = te_idx[np.isin(win_content[te_idx], list(heldout_content))]
            inner_tr, inner_vl = carve_inner_val(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)
        else:
            inner_tr, inner_vl = carve_inner_val_content(
                data, tr_idx, val_frac=inner_val_frac, seed=SEED + fold_num)

        if len(te_idx) < 5 or len(inner_tr) < 20:
            print(f"Fold {fold_num}: too few windows, skipping")
            continue

        te_dist = dict(sorted(Counter(data["att_idxs"][te_idx].tolist()).items()))
        print(f"\n{'='*62}\nFold {fold_num}  [{split_setting}, {label}]")
        print(f"  Inner train : {len(inner_tr)} windows")
        print(f"  Inner val   : {len(inner_vl)} windows")
        print(f"  TEST        : {len(te_idx)} windows, dist: {te_dist}")
        print(f"{'='*62}")

        mk = lambda idx, tr: AADDataset(data, idx, bank, train=tr, n_cand=n_cand)
        tr_loader = DataLoader(
            mk(inner_tr, True),
            batch_sampler=SubjectBatchSampler(subject_per_window(data)[inner_tr],
                                              batch_size, seed=SEED + fold_num),
            collate_fn=collate_fn, num_workers=0)
        vl_loader = DataLoader(mk(inner_vl, False), batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)
        te_loader = DataLoader(mk(te_idx, False), batch_size, shuffle=False,
                               collate_fn=collate_fn, num_workers=0)

        torch.manual_seed(_mode_seed(SEED + fold_num, mode))
        np.random.seed(_mode_seed(SEED + fold_num, mode))
        model = AADModel(mode=mode).to(DEVICE)

        ckpt = os.path.join(results_dir,
                            f"fold_{fold_num}_{mode}_{split_setting}.pt")
        best_val = train_model(model, tr_loader, vl_loader, epochs, ckpt, lr=lr,
                               loss_cfg=loss_cfg, val_shuffles=val_shuffles)

        strata = {"position": win_position[te_idx],
                  "trial": data["trial_ids"][te_idx]}
        res = Evaluator(model, te_loader, DEVICE,
                        strata=strata).battery(n_shuffle=test_shuffles)
        res["best_val_contribution"] = best_val
        res["n_inner_train_windows"] = int(len(inner_tr))
        res["n_inner_val_windows"] = int(len(inner_vl))
        res["test_speaker_dist"] = te_dist

        print(f"  -> Fold {fold_num}: acc={res['accuracy']:.4f} "
              f"null={res['null_mean']:.4f} "
              f"contribution={res['contribution']:+.4f} "
              f"p={res['p_permutation']:.3f} collapse={res['collapse']:.3f}")
        fold_results[fold_num] = res

    keys = ["accuracy", "null_mean", "contribution", "zeros_accuracy",
            "flip_rate", "collapse", "p_permutation",
            "null_position", "contribution_position",
            "null_trial", "contribution_trial"]
    vals = list(fold_results.values())
    summary = {
        "mode": label, "split_setting": split_setting,
        "candidates": candidates, "n_candidates": n_cand,
        "window_sec": window_sec, "hop_sec": hop_sec,
        "audio_only_probe": probe, "chance_level": chance,
        "folds": fold_results,
        "mean": {k: [float(np.mean([v[k] for v in vals])),
                     float(np.std([v[k] for v in vals]))]
                 for k in keys if vals and k in vals[0]},
        "note": "Report accuracy WITH null_mean and contribution. Accuracy "
                "alone does not distinguish a decoder that uses the recording "
                "from one that reads the candidates.",
    }
    m = summary["mean"]
    print(f"\n{split_setting.upper()} summary [{label}] "
          f"window={window_sec}s candidates={candidates}:")
    print(f"  accuracy     = {m['accuracy'][0]:.4f} +- {m['accuracy'][1]:.4f} "
          f"(chance {chance:.4f})")
    print(f"  permuted     = {m['null_mean'][0]:.4f}")
    print(f"  contribution = {m['contribution'][0]:+.4f}")

    out = os.path.join(results_dir, f"results_{mode}_{split_setting}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Results saved to {out}")
    return summary


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--local_path", default="maestro",
                   help="Dataset root (contains data/, media/, metadata/, splits/)")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--dataset_cache", default=None,
                   help="Optional dir memoising the assembled dataset, so a "
                        "window/modality sweep does not re-extract the audio "
                        "envelopes from FLAC on every run.")
    p.add_argument("--mode", choices=VALID_MODES, default="eeg")
    p.add_argument("--results", default="results_aad")
    p.add_argument("--split_setting", choices=["loso", "within"], default="within")
    p.add_argument("--splits_dir", default=None,
                   help="Defaults to <local_path>/splits")
    p.add_argument("--candidates", default="qmatch",
                   choices=["raw", "qmatch", "shifted", "shifted_qm"],
                   help="Candidate construction. 'qmatch' (default) removes the "
                        "acoustic confound; 'raw' reproduces the previous, "
                        "confounded construction; 'shifted*' use same-talker "
                        "temporal negatives.")
    p.add_argument("--n_candidates", type=int, default=N_SPEAKERS,
                   help="K. Must be <= 3 for the shifted constructions at "
                        "window/hop = 2, since a 30 s trial yields five windows "
                        "and an overlapping negative would be partly correct.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--inner_val_frac", type=float, default=0.2)
    p.add_argument("--held_out_content_frac", type=float, default=0.2)
    p.add_argument("--window_sec", type=float, default=None)
    p.add_argument("--hop_sec", type=float, default=None)
    p.add_argument("--test_shuffles", type=int, default=20)
    p.add_argument("--val_shuffles", type=int, default=3)
    for k, v in LOSS_DEFAULTS.items():
        p.add_argument(f"--{k}", type=float, default=v)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.splits_dir is None:
        args.splits_dir = os.path.join(args.local_path, "splits")
    print(f"Device: {DEVICE} | Mode: {args.mode} | Split: {args.split_setting}")

    window_sec = args.window_sec if args.window_sec is not None else dl_WINDOW_SEC
    hop_sec = args.hop_sec if args.hop_sec is not None else window_sec

    data = build_dataset_cached(local_path=args.local_path, mode=args.mode,
                                cache_dir=args.cache_dir,
                                window_sec=window_sec, hop_sec=hop_sec,
                                dataset_cache=args.dataset_cache)

    results_dir = (f"{args.results}_{args.split_setting}"
                   f"_w{window_sec:g}_h{hop_sec:g}_{args.candidates}")
    loss_cfg = {k: getattr(args, k) for k in LOSS_DEFAULTS}
    run_official_splits(
        data, results_dir=results_dir, mode=args.mode,
        split_setting=args.split_setting, splits_dir=args.splits_dir,
        epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        inner_val_frac=args.inner_val_frac,
        held_out_content_frac=args.held_out_content_frac,
        candidates=args.candidates, n_cand=args.n_candidates,
        window_sec=window_sec, hop_sec=hop_sec, loss_cfg=loss_cfg,
        test_shuffles=args.test_shuffles, val_shuffles=args.val_shuffles)
