"""
dataloader.py
-------------
"""

from __future__ import annotations

import json
import os
from math import gcd
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.interpolate import interp1d
from scipy.signal import butter, hilbert, iirnotch, filtfilt, resample_poly, sosfiltfilt
from torch.utils.data import Dataset, Sampler

# ── constants ─────────────────────────────────────────────────────────────────

EEG_FS_RAW   = 500
AUDIO_FS_RAW = 16_000
TARGET_FS    = 64
WINDOW_SEC   = 30
WINDOW_SAMP  = TARGET_FS * WINDOW_SEC   # 1920

N_EEG_CH   = 32
N_VIDEO_CH  = 4
N_GAZE_CH   = 6
N_IMU_CH    = 6
N_SPEAKERS  = 4


MODE_MODALITIES = {
    # singles (4)
    "eeg":               {"eeg"},
    "gaze":               {"gaze"},
    "imu":                {"imu"},
    "video":               {"video"},
    # pairs (6)
    "eeg_gaze":           {"eeg", "gaze"},
    "eeg_imu":            {"eeg", "imu"},
    "eeg_video":          {"eeg", "video"},
    "gaze_imu":           {"gaze", "imu"},
    "gaze_video":         {"gaze", "video"},
    "imu_video":          {"imu", "video"},
    # triples (4)
    "eeg_gaze_imu":       {"eeg", "gaze", "imu"},
    "eeg_gaze_video":     {"eeg", "gaze", "video"},
    "eeg_imu_video":      {"eeg", "imu", "video"},
    "gaze_imu_video":     {"gaze", "imu", "video"},
    # full combination (1)
    "eeg_gaze_imu_video": {"eeg", "gaze", "imu", "video"},
}


MODE_ALIASES = {
    "gi":       "gaze_imu",
    "eeg_vg":   "eeg_gaze_video",
    "eeg_vgi":  "eeg_gaze_imu_video",
}


VALID_MODES = tuple(MODE_MODALITIES.keys()) + tuple(MODE_ALIASES.keys())


def _canonical_mode(mode: str) -> str:
    """Resolve a legacy alias (e.g. 'eeg_vgi') to its canonical name
    (e.g. 'eeg_gaze_imu_video'); canonical names pass through unchanged."""
    return MODE_ALIASES.get(mode, mode)


def mode_uses(mode: str) -> tuple:
    """
    Returns (use_eeg, use_gaze, use_imu, use_video) booleans for a given
    mode name (canonical or legacy alias). This is the single source of
    truth for "which modalities does mode X activate" — every model/
    training script should call this instead of hardcoding its own
    tuple-membership checks.
    """
    canon = _canonical_mode(mode)
    if canon not in MODE_MODALITIES:
        raise ValueError(
            f"Unknown mode '{mode}'. Valid modes: {VALID_MODES}")
    m = MODE_MODALITIES[canon]
    return ("eeg" in m, "gaze" in m, "imu" in m, "video" in m)


# EEG electrode montage / bad-channel constants
MASTOIDS = ("M1", "M2")
ADC_CLIP = 0.0839   # amplifier ADC clip voltage, per the dataset's acquisition spec

try:
    import mne
    _HAVE_MNE = True
except ImportError:
    _HAVE_MNE = False


# ── signal helpers ─────────────────────────────────────────────────────────────

def _butter_bp(lo, hi, fs, order=4):
    nyq = fs / 2
    return butter(order, [lo / nyq, hi / nyq], btype="band", output="sos")


def _butter_lp(cutoff, fs, order=4):
    return butter(order, cutoff / (fs / 2), btype="low", output="sos")


def _resample(x: np.ndarray, fs_in: float, fs_out: int) -> np.ndarray:
    fs_in_i = int(round(fs_in))
    g    = gcd(fs_in_i, fs_out)
    up   = fs_out    // g
    down = fs_in_i   // g
    return resample_poly(x, up, down, axis=0).astype(np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    mu  = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0,  keepdims=True) + 1e-8
    return ((x - mu) / std).astype(np.float32)


def _interp_grid(ts: np.ndarray, vals: np.ndarray,
                 t0: float, t1: float, fs: int) -> np.ndarray:
    """Interpolate (ts, vals) onto a regular fs-Hz grid [t0, t1)."""
    grid = np.arange(t0, t1, 1.0 / fs)
    C    = vals.shape[1]
    out  = np.zeros((len(grid), C), dtype=np.float32)
    for c in range(C):
        f = interp1d(ts, vals[:, c], kind="linear",
                     bounds_error=False,
                     fill_value=(vals[0, c], vals[-1, c]))
        out[:, c] = f(grid).astype(np.float32)
    return out


# ── EEG bad-channel detection ──────────────────────────────────────────────────

def _detect_bad_channels(eeg_tc: np.ndarray, ch_names: list) -> list:

    data = eeg_tc.T   # (C, T) for per-channel statistics
    stds = data.std(axis=1)
    sat  = (np.abs(data) >= ADC_CLIP).mean(axis=1)
    bad  = set()
    for i, ch in enumerate(ch_names):
        if stds[i] < 1e-9 or sat[i] >= 0.1:
            bad.add(ch)
    rem = [i for i, ch in enumerate(ch_names) if ch not in bad]
    if len(rem) >= 3:
        ac  = np.diff(data[rem], axis=1)
        v   = np.var(ac, axis=1)
        med = np.median(v)
        mad = np.median(np.abs(v - med)) or 1e-30
        for j, i in enumerate(rem):
            if abs((v[j] - med) / (1.4826 * mad)) > 6.0:
                bad.add(ch_names[i])
    return sorted(bad)


# ── EEG preprocessing ──────────────────────────────────────────────────────────

def filter_reference_eeg(eeg_raw: np.ndarray,
                         ch_names: list,
                         fs_in: int = EEG_FS_RAW) -> np.ndarray:
    """

    Parameters
    ----------
    eeg_raw  : (T, C) raw EEG, time-first, FULL unmasked trial recording
    ch_names : list of C channel names (e.g. "Fp1", ..., "M1", "M2", ...)

    Returns
    -------
    (T, C) float32, same length as input (not resampled, not masked),
    per-channel z-scored
    """
    eeg = eeg_raw.astype(np.float64)

    # Notch BEFORE bandpass, matching preprocess.py's _preprocess_mne order
    b, a = iirnotch(60.0 / (fs_in / 2), Q=30)
    eeg  = filtfilt(b, a, eeg, axis=0)

    sos = _butter_bp(1.0, 40.0, fs_in)
    eeg = sosfiltfilt(sos, eeg, axis=0)

    bads = _detect_bad_channels(eeg, ch_names)

    if _HAVE_MNE:
        try:
            info = mne.create_info(list(ch_names), fs_in, ch_types="eeg")
            try:
                info.set_montage("standard_1020", match_case=False,
                                 on_missing="ignore")
            except Exception:
                pass
            raw = mne.io.RawArray(eeg.T, info, verbose="ERROR")   # MNE wants (C,T)
            raw.info["bads"] = list(bads)

            good_mastoids = [m for m in MASTOIDS
                             if m in ch_names and m not in bads]
            if good_mastoids:
                raw.set_eeg_reference(ref_channels=good_mastoids, verbose="ERROR")
            else:
                raw.set_eeg_reference("average", projection=False, verbose="ERROR")

            if raw.info["bads"]:
                try:
                    raw.interpolate_bads(reset_bads=True, verbose="ERROR")
                except Exception:
                    pass

            eeg = raw.get_data().T   # back to (T, C)
        except Exception:
            eeg = eeg - eeg.mean(axis=1, keepdims=True)
    else:
        eeg = eeg - eeg.mean(axis=1, keepdims=True)

    eeg = eeg.astype(np.float32)
    # Per-channel z-score over the full trial, matching gaze/IMU/video/audio
    return _zscore(eeg)


def preprocess_eeg(eeg_raw: np.ndarray,
                   ch_names: list,
                   fs_in: int = EEG_FS_RAW) -> np.ndarray:

    eeg = filter_reference_eeg(eeg_raw, ch_names, fs_in)
    return _resample(eeg, fs_in, TARGET_FS)




# ── audio envelope ─────────────────────────────────────────────────────────────

def extract_envelope(audio: np.ndarray, fs_in: int = AUDIO_FS_RAW,
                     target_rms: float | None = None) -> np.ndarray:
    """Speech amplitude envelope, standardised.

    NOTE ON `target_rms`.  This argument is a NO-OP and is retained only for
    call-site compatibility.  Every step below is linear, so E(alpha*x) =
    alpha*E(x); and the closing z-score is invariant to any affine map, so
    Z(alpha*e) = Z(e).  Composing, Z(E(alpha*x)) = Z(E(x)): a level difference
    is removed EXACTLY by the z-score alone, whether or not this prescale runs.
    Verified on real recordings -- scaling by +-15 dB changes the returned array
    by at most 1.6e-05, i.e. float32 rounding.

    The consequence is important and was previously missed.  Because a pure
    level difference provably cannot survive, the acoustic cue that let a
    decoder identify the attended talker without any physiological input is NOT
    loudness.  It is a difference in the SHAPE of the envelope distribution
    (kurtosis, skew, sparsity, dynamic range), which is invariant to affine
    rescaling and therefore passes through untouched.  No per-candidate
    normalisation can remove it; see `quantile_match_candidates`.
    """
    audio = audio.astype(np.float64)
    if target_rms is not None:
        current_rms = np.sqrt(np.mean(audio ** 2)) + 1e-8
        audio = audio * (target_rms / current_rms)   # cancelled by _zscore below

    env = np.abs(hilbert(audio)).astype(np.float32)
    sos = _butter_lp(20.0, fs_in)
    env = sosfiltfilt(sos, env).astype(np.float32)
    env = _resample(env, fs_in, TARGET_FS)
    return _zscore(env)[:, np.newaxis]


# ── candidate construction ─────────────────────────────────────────────────────

def quantile_match_candidates(A: np.ndarray, chunk: int = 1000) -> np.ndarray:
    """Force the K candidates of each window onto a common value distribution.

    A : (N, K, T) standardised envelopes -> (N, K, T)

    In this stimulus material the attended talker was prepared differently from
    its competitors: it is ~15 dB louder, and -- because that difference is one
    of dynamics rather than gain (crest factor differs by 9.4 dB, and crest is
    gain-invariant) -- it also has systematically lower kurtosis, lower skew,
    lower temporal sparsity and a wider inter-quantile range.  Those statistics
    are affine-invariant, so they survive the z-score in `extract_envelope`, and
    a logistic probe on eight such features picks the attended talker 56 % of
    the time on content-disjoint folds against a 25 % chance level -- more than
    the network itself extracts from EEG.  No subject or content split removes
    this, because it is a property of the target ROLE.

    The remedy: replace each candidate's samples by the shared "vocabulary" of
    values `ref` (the average order statistics of the K candidates), indexed by
    that candidate's own ranks.  Every candidate then holds an identical multiset
    of values, so every statistic computed from the value multiset -- all
    moments, all quantiles, kurtosis, skew, Gini sparsity, dynamic range,
    silence fraction -- is identical across candidates by construction.  Only
    the temporal ORDERING differs, which is exactly the property a neural
    response tracks.

    Residual: relative band powers depend on ordering and therefore survive, so
    the audio-only probe lands at 0.26 rather than exactly 0.25.  Any operation
    that also flattened the spectra would destroy the signal of interest.  The
    fully confound-free alternative is same-talker temporal negatives
    (`build_shifted_candidates`), whose probe is 0.5002 against 0.5000.
    """
    N, K, T = A.shape
    out = np.empty_like(A, dtype=np.float32)
    ar = np.arange(T)
    for s0 in range(0, N, chunk):
        a = A[s0:s0 + chunk]
        n = a.shape[0]
        order = np.argsort(a, axis=2, kind="stable")
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(ar, (n, K, T)), axis=2)
        ref = np.sort(a, axis=2).mean(axis=1, keepdims=True)          # (n,1,T)
        matched = np.take_along_axis(np.broadcast_to(ref, (n, K, T)),
                                     ranks, axis=2)
        mu = matched.mean(axis=2, keepdims=True)
        sd = matched.std(axis=2, keepdims=True) + 1e-8
        out[s0:s0 + chunk] = ((matched - mu) / sd).astype(np.float32)
    return out


def build_shifted_candidates(trial_ids: np.ndarray, window_sec: float,
                             hop_sec: float, n_neg: int = 2,
                             seed: int = 0) -> tuple:
    """Same-talker temporal negatives: the fully confound-free construction.

    For each window, the positive is the attended talker's envelope on that
    window and the negatives are the SAME talker's envelope on non-overlapping
    windows of the same trial.  The candidates are then exchangeable, so the
    audio-only Bayes accuracy is exactly 1/K.

    Two details that matter.  Negatives are drawn UNIFORMLY among the admissible
    windows: taking the temporally furthest one biases them toward trial edges,
    whose onset/offset statistics are themselves distinctive, and that alone
    lifted the audio-only probe to 0.60 on a chance-0.50 task.  And a negative
    must never overlap the positive -- with hop < window it would be partly
    correct -- so K is reduced rather than allowing overlap.  A 30 s trial at
    window/hop = 2 yields five windows, supporting at most two disjoint
    negatives.

    Returns (imposter_idx (N, n_neg), n_fallback).
    """
    gap = max(1, int(np.ceil(window_sec / hop_sec)))
    out = np.full((len(trial_ids), n_neg), -1, dtype=np.int64)
    rng = np.random.default_rng(seed)
    n_fallback = 0
    for t in np.unique(trial_ids):
        idx = np.where(trial_ids == t)[0]        # contiguous, temporal order
        n = len(idx)
        for p in range(n):
            valid = [q for q in range(n) if abs(q - p) >= gap]
            rng.shuffle(valid)
            if len(valid) < n_neg:
                n_fallback += 1
                extra = [q for q in range(n) if q != p and q not in valid]
                valid = valid + list(rng.permutation(extra))
            if not valid:                        # single-window trial
                out[idx[p]] = idx[p]
                continue
            pick = (valid * n_neg)[:n_neg]
            out[idx[p]] = idx[np.asarray(pick)]
    assert (out >= 0).all(), "imposter index not assigned"
    return out, n_fallback


def make_candidate_bank(data: dict, construction: str = "qmatch",
                        window_sec: float = WINDOW_SEC,
                        hop_sec: float | None = None,
                        n_cand: int = N_SPEAKERS, seed: int = 0) -> dict:
    """Assemble the candidate arrays once for the whole dataset.

    construction:
      "raw"        the K co-present talkers, standardised.  CONFOUNDED -- an
                   audio-only probe reaches 0.56 against 0.25 chance.  Retained
                   only for reproducing the previous result.
      "qmatch"     the same K talkers, distribution-matched.  DEFAULT.
      "shifted"    same-talker temporal negatives.
      "shifted_qm" same-talker negatives, additionally distribution-matched.

    Returns {"construction", "A": (N,K,T), "pos": (N,), "spk_meaningful": bool}
    where `pos[i]` indexes the correct candidate and `spk_meaningful` records
    whether candidate index still corresponds to a loudspeaker (it does not for
    same-talker constructions, where every candidate is the same talker).
    """
    hop_sec = hop_sec if hop_sec is not None else window_sec
    N = len(data["trial_ids"])

    if construction in ("raw", "qmatch"):
        A = np.stack([data["audio"][k][:, :, 0] for k in range(n_cand)], axis=1)
        if construction == "qmatch":
            A = quantile_match_candidates(A)
        return {"construction": construction,
                "A": np.ascontiguousarray(A, dtype=np.float32),
                "pos": data["att_idxs"].astype(np.int64),
                "spk_meaningful": True}

    if construction in ("shifted", "shifted_qm"):
        att = np.stack([data["audio"][a][i, :, 0]
                        for i, a in enumerate(data["att_idxs"])]).astype(np.float32)
        imp, n_fb = build_shifted_candidates(data["trial_ids"], window_sec,
                                             hop_sec, n_neg=n_cand - 1, seed=seed)
        A = np.stack([att] + [att[imp[:, j]] for j in range(n_cand - 1)], axis=1)
        if construction == "shifted_qm":
            A = quantile_match_candidates(A)
        if n_fb:
            print(f"  shifted candidates: {n_fb} windows ({100*n_fb/max(N,1):.1f} %) "
                  f"needed the overlap fallback")
        return {"construction": construction,
                "A": np.ascontiguousarray(A, dtype=np.float32),
                "pos": np.zeros(N, dtype=np.int64),
                "spk_meaningful": False}

    raise ValueError(f"unknown construction: {construction}")


# ── video optical flow ─────────────────────────────────────────────────────────

def extract_optical_flow(video_path: str,
                         t_start: float,
                         t_end: float) -> np.ndarray:

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    start_frame = max(0, int(t_start * fps) - 1)
    end_frame   = int(t_end * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    features, prev_gray = [], None
    for _ in range(start_frame, end_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        small = cv2.resize(frame, (160, 90))
        gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            features.append([mag.mean(), mag.std(),
                              flow[..., 0].mean(), flow[..., 1].mean()])
        prev_gray = gray
    cap.release()

    n_out = int((t_end - t_start) * TARGET_FS)
    if len(features) < 2:
        return np.zeros((n_out, N_VIDEO_CH), dtype=np.float32)
    arr = np.array(features, dtype=np.float32)
    out = _resample(arr, fps, TARGET_FS)
    if len(out) >= n_out:
        out = out[:n_out]
    else:
        pad = np.zeros((n_out - len(out), N_VIDEO_CH), dtype=np.float32)
        out = np.concatenate([out, pad], axis=0)
    return _zscore(out)


# ── gaze preprocessing ─────────────────────────────────────────────────────────

def preprocess_gaze(gaze_raw: np.ndarray,
                    t_sec: np.ndarray,
                    t_start: float,
                    t_end: float) -> np.ndarray:

    mask = (t_sec >= t_start - 0.1) & (t_sec <= t_end + 0.1)
    ts, vals = t_sec[mask], gaze_raw[mask]
    n_out = int((t_end - t_start) * TARGET_FS)
    C = vals.shape[1] if len(vals) > 0 else gaze_raw.shape[1]
    if len(ts) < 4:
        return np.zeros((n_out, C), dtype=np.float32)

    grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)
    out  = np.zeros((len(grid), C), dtype=np.float32)
    for c in range(C):
        valid = ~np.isnan(vals[:, c])
        if valid.sum() < 4:
            continue
        f = interp1d(ts[valid], vals[valid, c], kind="linear",
                     bounds_error=False,
                     fill_value=(vals[valid, c][0], vals[valid, c][-1]))
        out[:, c] = f(grid).astype(np.float32)

    sos = _butter_lp(10.0, TARGET_FS)
    out = sosfiltfilt(sos, out, axis=0).astype(np.float32)
    return _zscore(out)


# ── IMU preprocessing ──────────────────────────────────────────────────────────

def preprocess_imu(imu_raw: np.ndarray,
                   t_sec: np.ndarray,
                   t_start: float,
                   t_end: float) -> np.ndarray:

    mask = (t_sec >= t_start - 0.1) & (t_sec <= t_end + 0.1)
    ts, vals = t_sec[mask], imu_raw[mask]
    n_out = int((t_end - t_start) * TARGET_FS)
    C = vals.shape[1] if len(vals) > 0 else imu_raw.shape[1]
    if len(ts) < 4:
        return np.zeros((n_out, C), dtype=np.float32)

    fs_imu = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 130.0

    grid_imu = np.arange(t_start, t_end, 1.0 / fs_imu)
    out_imu  = np.zeros((len(grid_imu), C), dtype=np.float32)
    for c in range(C):
        valid = ~np.isnan(vals[:, c])
        if valid.sum() < 4:
            continue
        f = interp1d(ts[valid], vals[valid, c], kind="linear",
                     bounds_error=False,
                     fill_value=(vals[valid, c][0], vals[valid, c][-1]))
        out_imu[:, c] = f(grid_imu).astype(np.float32)

    out = _resample(out_imu, fs_imu, TARGET_FS)
    sos = _butter_lp(20.0, TARGET_FS)
    out = sosfiltfilt(sos, out, axis=0).astype(np.float32)
    return _zscore(out)


# ── sync helper ────────────────────────────────────────────────────────────────

def _load_sync(timing_path: str) -> dict:
    with open(timing_path) as f:
        t = json.load(f)

    align   = t["align"]
    tobii   = t["tobii"]
    anchor_unix = align["anchor_unix"]
    end_unix    = align["end_unix"]

    recording_start = tobii["recording_start_unix"]
    gaze_trim_sec  = (anchor_unix - recording_start) + tobii.get("gaze_t_first", 0.0)
    imu_trim_sec   = (anchor_unix - recording_start) + tobii.get("imu_t_first",  0.0)
    video_trim_sec = (anchor_unix - recording_start)

    trial_end_sec = align.get("overlap_sec", end_unix - anchor_unix)

    RAW_TO_LOGICAL_DEVICE = {6: 1, 5: 2, 3: 3}
    audio_device_t0 = {
        RAW_TO_LOGICAL_DEVICE.get(d["device_id_raw"], d["device_id_raw"]):
            d["playback_start_unix"]
        for d in t["audio"].get("devices", [])
        if "device_id_raw" in d and "playback_start_unix" in d
    }

    return {
        "anchor_unix":            anchor_unix,
        "end_unix":               end_unix,
        "trial_end_sec":          trial_end_sec,
        "eeg_first_sample_unix":  t["eeg"]["first_sample_unix"],
        "eeg_t0_internal_sec":    t["eeg"]["t0_internal_sec"],
        "gaze_trim_sec":          gaze_trim_sec,
        "imu_trim_sec":           imu_trim_sec,
        "video_trim_sec":         video_trim_sec,
        "audio_t0_unix":          t["audio"]["t0_unix"],   # fallback
        "audio_device_t0":       audio_device_t0,          # NEW
    }


# ── cache helpers (video, gaze, IMU — one .npz per (subject, trial)) ─────────

def _save_cache(path: str, arrays: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    existing = {}
    if os.path.exists(path):
        try:
            d = np.load(path)
            existing = {k: d[k] for k in d.files}
        except Exception:
            pass
    existing.update({k: v for k, v in arrays.items() if v is not None})
    np.savez_compressed(path, **existing)


def _load_cache(path: str, required_keys: list,
                expected_len: int | None = None) -> dict | None:
    if not os.path.exists(path):
        return None
    try:
        d      = np.load(path)
        loaded = {k: d[k] for k in d.files}
    except Exception:
        return None
    if not all(k in loaded for k in required_keys):
        return None
    return loaded


# ── trial loader ───────────────────────────────────────────────────────────────

def load_trial(local_path: str,
               sid: str,
               tid: str,
               audio_layout: list,
               attended_speaker: int,
               mode: str,
               cache_dir: str | None = None,
               window_sec: float = WINDOW_SEC,
               hop_sec: float | None = None) -> dict | None:
    root = Path(local_path)

    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)

    # ── Sync ──────────────────────────────────────────────────────────────────
    timing_path = (root / "media" / "timing" /
                   f"subject={sid}" / f"trial={tid}.json")
    if not timing_path.exists():
        print(f"  Skipping {sid}/{tid}: timing file not found")
        return None
    try:
        sync = _load_sync(str(timing_path))
    except Exception as e:
        print(f"  Skipping {sid}/{tid}: sync error: {e}")
        return None

    anchor_unix    = sync["anchor_unix"]
    gaze_trim_sec  = sync["gaze_trim_sec"]
    imu_trim_sec   = sync["imu_trim_sec"]
    video_trim_sec = sync["video_trim_sec"]
    trial_end_sec  = sync["trial_end_sec"]

    eeg_proc = None
    eeg_cpath = (os.path.join(cache_dir, f"{sid}_{tid}.npz")
                if cache_dir else None)
    eeg_cached = _load_cache(eeg_cpath, ["eeg"], WINDOW_SAMP) \
                if (use_eeg and eeg_cpath) else None

    if use_eeg and eeg_cached is not None:
        eeg_proc = eeg_cached["eeg"]
    elif use_eeg:
        eeg_path = (root / "data" / "eeg" /
                    f"subject={sid}" / f"trial={tid}.parquet")
        if not eeg_path.exists():
            print(f"  Skipping {sid}/{tid}: EEG parquet not found")
            return None
        try:
            df       = pd.read_parquet(eeg_path)
            ch_cols  = [c for c in df.columns if c.startswith("ch_")]
            ch_names = [c[3:] for c in ch_cols]
            eeg_raw  = df[ch_cols].to_numpy(dtype=np.float32)

            t_sec    = df["t_sec"].to_numpy()
            eeg_unix = sync["eeg_first_sample_unix"] + \
                      (t_sec - sync["eeg_t0_internal_sec"])
            mask     = (eeg_unix >= anchor_unix) & \
                      (eeg_unix <= anchor_unix + trial_end_sec)

            eeg_filtered = filter_reference_eeg(eeg_raw, ch_names, EEG_FS_RAW)
            eeg_proc     = _resample(eeg_filtered[mask], EEG_FS_RAW, TARGET_FS)

            if eeg_cpath:
                _save_cache(eeg_cpath, {"eeg": eeg_proc})
        except Exception as e:
            print(f"  Skipping {sid}/{tid}: EEG error: {e}")
            return None

    audio_dir       = root / "media" / "audio" / tid
    audio_t0_shared = sync["audio_t0_unix"]
    audio_device_t0 = sync.get("audio_device_t0", {})
    envs = []
    try:
        attendable = sorted(
            [spk for spk in audio_layout if spk.get("attendable", True)],
            key=lambda x: x["speaker"]
        )

        raw_waveforms = []
        for spk in attendable:
            spk_n   = spk["speaker"]
            dev_id  = spk.get("device")
            audio_t0 = audio_device_t0.get(dev_id, audio_t0_shared)
            if dev_id is not None and dev_id not in audio_device_t0 and audio_device_t0:
                print(f"  WARNING {sid}/{tid}: speaker {spk_n}'s device={dev_id} "
                      f"not found in translated audio_device_t0 keys "
                      f"{list(audio_device_t0.keys())} -- falling back to "
                      f"shared audio_t0_unix. This should be rare; if you see "
                      f"this often, the inferred RAW_TO_LOGICAL_DEVICE mapping "
                      f"in _load_sync() may not hold for this trial/subject.")

            matches = sorted(audio_dir.glob(f"speaker{spk_n}_*.flac"))
            if not matches:
                raise FileNotFoundError(
                    f"No audio file found for speaker {spk_n} in {audio_dir}")
            wav, sr = sf.read(str(matches[0]), dtype="float32", always_2d=False)

            i0 = max(0, int(round((anchor_unix - audio_t0) * sr)))
            i1 = int(round((anchor_unix + trial_end_sec - audio_t0) * sr))
            raw_waveforms.append((wav[i0:i1], sr))

        trial_target_rms = float(np.mean([
            np.sqrt(np.mean(w.astype(np.float64) ** 2)) for w, _ in raw_waveforms
        ]))

        # Second pass: envelope extraction, equalized to the shared trial target.
        for wav, sr in raw_waveforms:
            envs.append(extract_envelope(wav, sr, target_rms=trial_target_rms))
    except Exception as e:
        print(f"  Skipping {sid}/{tid}: audio error: {e}")
        return None

    # ── Common length so far ──────────────────────────────────────────────────
    lengths = [len(e) for e in envs]
    if eeg_proc is not None:
        lengths.append(len(eeg_proc))

    # ── Video / Gaze / IMU ────────────────────────────────────────────────────
    video_proc = gaze_proc = imu_proc = None

    if use_video or use_gaze or use_imu:
        cpath    = (os.path.join(cache_dir, f"{sid}_{tid}.npz")
                    if cache_dir else None)
        req_keys = (["video"] if use_video else []) + \
                   (["gaze"]  if use_gaze  else []) + \
                   (["imu"]   if use_imu   else [])
        cached   = _load_cache(cpath, req_keys, WINDOW_SAMP) \
                   if cpath else None

        if cached is None:
            min_sec = trial_end_sec if trial_end_sec is not None \
                     else min(lengths) / TARGET_FS
            computed = {}

            if use_video:
                v_start = video_trim_sec
                v_end   = v_start + min_sec
                vpath = (root / "media" / "video" /
                         f"subject={sid}" / f"{tid}.mp4")
                try:
                    computed["video"] = extract_optical_flow(
                        str(vpath), v_start, v_end)
                except Exception as e:
                    print(f"  Skipping {sid}/{tid}: video error: {e}")
                    return None

            if use_gaze:
                g_start = gaze_trim_sec
                g_end   = g_start + min_sec
                gpath = (root / "data" / "gaze" /
                         f"subject={sid}" / f"trial={tid}.parquet")
                try:
                    gdf   = pd.read_parquet(gpath)
                    t_sec = gdf["t"].to_numpy()
                    gdf["pupil"] = gdf["R_pupil"].fillna(gdf["L_pupil"])
                    gcols = ["gaze2d_x", "gaze2d_y",
                             "gaze3d_x", "gaze3d_y", "gaze3d_z",
                             "pupil"]
                    computed["gaze"] = preprocess_gaze(
                        gdf[gcols].to_numpy(dtype=np.float32),
                        t_sec, g_start, g_end)
                except Exception as e:
                    print(f"  Skipping {sid}/{tid}: gaze error: {e}")
                    return None

            if use_imu:
                i_start = imu_trim_sec
                i_end   = i_start + min_sec
                ipath = (root / "data" / "imu" /
                         f"subject={sid}" / f"trial={tid}.parquet")
                try:
                    idf   = pd.read_parquet(ipath)
                    t_sec = idf["t"].to_numpy()
                    icols = ["ax", "ay", "az", "gx", "gy", "gz"]
                    computed["imu"] = preprocess_imu(
                        idf[icols].to_numpy(dtype=np.float32),
                        t_sec, i_start, i_end)
                except Exception as e:
                    print(f"  Skipping {sid}/{tid}: IMU error: {e}")
                    return None

            if cpath:
                _save_cache(cpath, computed)
            cached = computed

        if use_video: video_proc = cached.get("video")
        if use_gaze:  gaze_proc  = cached.get("gaze")
        if use_imu:   imu_proc   = cached.get("imu")

    # ── Align to shortest ─────────────────────────────────────────────────────
    for arr in (video_proc, gaze_proc, imu_proc):
        if arr is not None:
            lengths.append(len(arr))
    min_len = min(lengths)

    envs = [e[:min_len] for e in envs]
    if eeg_proc   is not None: eeg_proc   = eeg_proc[:min_len]
    if video_proc is not None: video_proc = video_proc[:min_len]
    if gaze_proc  is not None: gaze_proc  = gaze_proc[:min_len]
    if imu_proc   is not None: imu_proc   = imu_proc[:min_len]

    # ── Windowing (configurable size/overlap) ─────────────────────────────────
    hop_sec_eff  = hop_sec if hop_sec is not None else window_sec
    window_samp  = int(round(window_sec * TARGET_FS))
    hop_samp     = int(round(hop_sec_eff * TARGET_FS))
    if window_samp <= 0 or hop_samp <= 0:
        print(f"  Skipping {sid}/{tid}: invalid window_sec/hop_sec "
              f"({window_sec}/{hop_sec_eff})")
        return None

    pad_to = ((min_len + window_samp - 1) // window_samp) * window_samp
    if pad_to - min_len <= TARGET_FS:
        min_len = pad_to
    if min_len < window_samp:
        return None

    starts = list(range(0, min_len - window_samp + 1, hop_samp))
    n_win  = len(starts)
    if n_win == 0:
        return None

    def _win(x, C):
        if x is None:
            return None
        if len(x) < min_len:
            pad = np.zeros((min_len - len(x), C), dtype=np.float32)
            x   = np.concatenate([x, pad], axis=0)
        out = np.zeros((n_win, window_samp, C), dtype=np.float32)
        for i, s in enumerate(starts):
            out[i] = x[s:s + window_samp]
        return out

    return {
        "eeg":      _win(eeg_proc,   N_EEG_CH),
        "video":    _win(video_proc, N_VIDEO_CH),
        "gaze":     _win(gaze_proc,  N_GAZE_CH),
        "imu":      _win(imu_proc,   N_IMU_CH),
        "audio":    [_win(e, 1) for e in envs],
        "att_idxs": np.full(n_win, attended_speaker - 1, dtype=np.int64),
    }


# ── dataset builder ────────────────────────────────────────────────────────────

def build_dataset(local_path: str,
                  mode: str,
                  subjects: list | str = "all",
                  trials: str = "main",
                  cache_dir: str | None = None,
                  window_sec: float = WINDOW_SEC,
                  hop_sec: float | None = None) -> dict:
    assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
    root = Path(local_path)

    trials_df   = pd.read_csv(root / "metadata" / "trials.csv")
    audio_meta  = json.loads(
        (root / "metadata" / "audio_layout.json").read_text())
    audio_layout = audio_meta["speakers"]

    if trials == "main":
        trials_df = trials_df[trials_df["kind"] == "main"].copy()

    subj_list = list(range(1, 17)) if subjects == "all" else list(subjects)

    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)

    all_eeg      = [] if use_eeg   else None
    all_video    = [] if use_video else None
    all_gaze     = [] if use_gaze  else None
    all_imu      = [] if use_imu   else None
    all_audio    = [[] for _ in range(N_SPEAKERS)]
    all_att_idxs = []
    all_trial_ids= []
    trial_meta   = []
    tid_ctr      = 0

    for s in subj_list:
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
                window_sec      = window_sec,
                hop_sec         = hop_sec,
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
            all_trial_ids.append(
                np.full(n_win, tid_ctr, dtype=np.int64))
            trial_meta.append({"trial_id": tid_ctr,
                                "att_idx":  att_spk - 1,
                                "subject":  s,        # NEW
                                "tid":      tid})     # NEW — original content ID
            tid_ctr += 1

        print(f"Subject {s}: loaded")

    def _cat(lst):
        if lst is None or not lst:
            return None
        lst = [x for x in lst if x is not None]
        return np.concatenate(lst, axis=0) if lst else None

    dataset = {
        "eeg":   _cat(all_eeg),
        "video": _cat(all_video),
        "gaze":  _cat(all_gaze),
        "imu":   _cat(all_imu),
        "audio": [np.concatenate(all_audio[i], axis=0)
                  for i in range(N_SPEAKERS)],
        "att_idxs":           np.concatenate(all_att_idxs,  axis=0),
        "trial_ids":          np.concatenate(all_trial_ids, axis=0),
        "trial_meta_ids":     np.array([t["trial_id"] for t in trial_meta],
                                        dtype=np.int64),
        "trial_meta_att_idx": np.array([t["att_idx"]  for t in trial_meta],
                                        dtype=np.int64),
        "trial_meta_subject": np.array([t["subject"]  for t in trial_meta],
                                        dtype=np.int64),   # NEW
        "trial_meta_tid":     np.array([t["tid"]       for t in trial_meta],
                                        dtype=object),      # NEW
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal ({mode}): {n_trials} trials, {n_windows} windows")
    from collections import Counter
    dist = Counter(dataset["trial_meta_att_idx"].tolist())
    print("Attended speaker dist (0-based):", dict(sorted(dist.items())))
    return dataset


def build_dataset_cached(local_path: str, mode: str, cache_dir: str | None,
                         window_sec: float, hop_sec: float | None = None,
                         dataset_cache: str | None = None, **kwargs) -> dict:
    """`build_dataset` memoised to a single .npz.

    The per-trial cache under `cache_dir` stores preprocessed signals but not the
    audio envelopes, which are re-extracted from FLAC on every build -- ~6400
    reads per call.  When sweeping windows and modality sets that dominates the
    runtime, so the assembled dataset is memoised here as well.  Keyed by mode,
    window and hop; delete the .npz to force a rebuild.
    """
    if dataset_cache is None:
        return build_dataset(local_path=local_path, mode=mode,
                             cache_dir=cache_dir, window_sec=window_sec,
                             hop_sec=hop_sec, **kwargs)

    hop = hop_sec if hop_sec is not None else window_sec
    path = os.path.join(dataset_cache,
                        f"dataset__{mode}_w{window_sec:g}_h{hop:g}.npz")
    if os.path.exists(path):
        print(f"[build] {mode} w={window_sec:g} <- cached {path}", flush=True)
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files if not k.startswith("audio_")}
        d["audio"] = [z[f"audio_{i}"] for i in range(N_SPEAKERS)]
        for k in ("eeg", "video", "gaze", "imu"):
            d.setdefault(k, None)
        return d

    d = build_dataset(local_path=local_path, mode=mode, cache_dir=cache_dir,
                      window_sec=window_sec, hop_sec=hop_sec, **kwargs)
    os.makedirs(dataset_cache, exist_ok=True)
    save = {k: v for k, v in d.items() if k != "audio" and v is not None}
    save["trial_meta_tid"] = save["trial_meta_tid"].astype("<U24")
    save.update({f"audio_{i}": a for i, a in enumerate(d["audio"])})
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez(tmp, **save)
    os.replace(tmp, path)
    print(f"[build] cached -> {path}", flush=True)
    return d


def subject_per_window(data: dict) -> np.ndarray:
    """Listener id for every window."""
    return data["trial_meta_subject"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]


def content_per_window(data: dict) -> np.ndarray:
    """Stimulus-content id for every window."""
    return data["trial_meta_tid"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"])]


def position_in_trial(data: dict) -> np.ndarray:
    """Index of each window within its trial, for the position-stratified null."""
    tr = data["trial_ids"]
    _, first = np.unique(tr, return_index=True)
    start = np.zeros(len(tr), dtype=np.int64)
    start[first] = first
    np.maximum.accumulate(start, out=start)
    return np.arange(len(tr)) - start


# ── K-fold splitting ───────────────────────────────────────────────────────────

def get_trial_level_splits(data: dict, n_splits: int = 5, seed: int = 42,
                           held_out_content_frac: float = 0.2):

    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    trial_ids  = data["trial_meta_ids"]
    trial_att  = data["trial_meta_att_idx"]
    trial_subj = data["trial_meta_subject"]
    trial_tid  = data["trial_meta_tid"]
    win_ids    = data["trial_ids"]

    # ── Step 1: global trial-CONTENT holdout, computed ONCE ─────────────────
    unique_content = np.unique(trial_tid)
    content_att = np.array([trial_att[trial_tid == c][0] for c in unique_content])
    train_content, heldout_content = train_test_split(
        unique_content, test_size=held_out_content_frac,
        stratify=content_att, random_state=seed)
    train_content_set   = set(train_content.tolist())
    heldout_content_set = set(heldout_content.tolist())
    is_train_content   = np.array([t in train_content_set   for t in trial_tid])
    is_heldout_content = np.array([t in heldout_content_set for t in trial_tid])

    # ── Step 2: subject-grouped K-fold on top ───────────────────────────────
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_t, vl_t) in enumerate(
            sgkf.split(trial_ids, trial_att, groups=trial_subj)):
        train_trial_ids = set(trial_ids[tr_t].tolist())
        val_trial_ids   = set(trial_ids[vl_t].tolist())

        win_trial_lookup = np.searchsorted(trial_ids, win_ids)

        # Train windows: train-side subjects AND non-held-out content
        train_mask = (np.isin(win_ids, list(train_trial_ids)) &
                     is_train_content[win_trial_lookup])
        # Val windows: val-side subjects AND held-out content only
        val_mask = (np.isin(win_ids, list(val_trial_ids)) &
                   is_heldout_content[win_trial_lookup])

        yield fold, np.where(train_mask)[0], np.where(val_mask)[0]


def carve_inner_val(data: dict, window_idx: np.ndarray,
                    val_frac: float = 0.2, seed: int = 42) -> tuple:

    subj_lookup = data["trial_meta_subject"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"][window_idx])]
    unique_subjects = np.unique(subj_lookup)

    rng = np.random.default_rng(seed)
    n_val_subj = max(1, int(round(val_frac * len(unique_subjects))))
    val_subjects = set(rng.choice(unique_subjects, size=n_val_subj,
                                  replace=False).tolist())

    is_val = np.isin(subj_lookup, list(val_subjects))
    inner_val_idx   = window_idx[is_val]
    inner_train_idx = window_idx[~is_val]
    return inner_train_idx, inner_val_idx


def load_official_splits(splits_dir: str, setting: str) -> list:

    import json as _json
    from pathlib import Path as _Path

    setting_dir = _Path(splits_dir) / setting
    fold_files = sorted(setting_dir.glob("fold_*.json"))
    if not fold_files:
        raise FileNotFoundError(
            f"No fold_*.json files found in {setting_dir}. "
            f"Expected the dataset's splits/{setting}/ folder.")

    folds = []
    for f in fold_files:
        with open(f) as fh:
            folds.append(_json.load(fh))
    folds.sort(key=lambda d: d["fold"])
    return folds


def get_official_split_windows(data: dict, fold: dict) -> tuple:

    setting = fold["setting"]
    win_trial_lookup = np.searchsorted(data["trial_meta_ids"], data["trial_ids"])

    if setting == "loso":
        # Subject IDs in the split files are strings like "S01"; data's
        # trial_meta_subject is an int (e.g. 1) — convert consistently.
        test_subj_ints  = {int(s.lstrip("S")) for s in fold["test_subjects"]}
        train_subj_ints = {int(s.lstrip("S")) for s in fold["train_subjects"]}

        subj_per_win = data["trial_meta_subject"][win_trial_lookup]
        train_idx = np.where(np.isin(subj_per_win, list(train_subj_ints)))[0]
        test_idx  = np.where(np.isin(subj_per_win, list(test_subj_ints)))[0]

    elif setting == "intra":
        test_content_set  = set(fold["test_trials"])
        train_content_set = set(fold["train_trials"])

        content_per_win = data["trial_meta_tid"][win_trial_lookup]
        train_idx = np.where(np.isin(content_per_win, list(train_content_set)))[0]
        test_idx  = np.where(np.isin(content_per_win, list(test_content_set)))[0]

    else:
        raise ValueError(f"Unknown official split setting: {setting}")

    return train_idx, test_idx


def compute_global_content_holdout(data: dict, held_out_content_frac: float = 0.2,
                                   seed: int = 42) -> tuple:

    from sklearn.model_selection import train_test_split

    unique_content = np.unique(data["trial_meta_tid"])
    content_att = np.array([
        data["trial_meta_att_idx"][data["trial_meta_tid"] == c][0]
        for c in unique_content
    ])
    train_content, heldout_content = train_test_split(
        unique_content, test_size=held_out_content_frac,
        stratify=content_att, random_state=seed)
    return set(train_content.tolist()), set(heldout_content.tolist())


def carve_inner_val_content(data: dict, window_idx: np.ndarray,
                            val_frac: float = 0.2, seed: int = 42) -> tuple:

    content_lookup = data["trial_meta_tid"][
        np.searchsorted(data["trial_meta_ids"], data["trial_ids"][window_idx])]
    unique_content = np.unique(content_lookup)

    rng = np.random.default_rng(seed)
    n_val_content = max(1, int(round(val_frac * len(unique_content))))
    val_content = set(rng.choice(unique_content, size=n_val_content,
                                 replace=False).tolist())

    is_val = np.isin(content_lookup, list(val_content))
    inner_val_idx   = window_idx[is_val]
    inner_train_idx = window_idx[~is_val]
    return inner_train_idx, inner_val_idx


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class AADDataset(Dataset):
    """Windows plus their candidate set.

    Differences from the previous revision:
      * candidates come from a `bank` built by `make_candidate_bank`, so the
        acoustic confound can be removed (see `quantile_match_candidates`);
      * returns the candidate PERMUTATION, so an orientation head predicting a
        fixed loudspeaker index can be mapped into slot order;
      * returns the SUBJECT id, so the contrastive loss can restrict its
        in-batch negatives to one listener -- raw EEG identifies the listener
        with 0.90 accuracy (16-way, chance 0.0625), so a cross-listener batch is
        solved by identity alone and teaches nothing about attention;
      * the label is an integer index rather than a one-hot vector.
    """

    def __init__(self, data: dict, window_idx: np.ndarray, bank: dict,
                 train: bool = True, n_cand: int = N_SPEAKERS):
        self.gidx = np.asarray(window_idx)
        self.train = train
        self.K = n_cand
        self.eeg   = torch.from_numpy(data["eeg"][self.gidx])   \
                     if data["eeg"]   is not None else None
        self.video = torch.from_numpy(data["video"][self.gidx]) \
                     if data["video"] is not None else None
        self.gaze  = torch.from_numpy(data["gaze"][self.gidx])  \
                     if data["gaze"]  is not None else None
        self.imu   = torch.from_numpy(data["imu"][self.gidx])   \
                     if data["imu"]   is not None else None
        self.A     = torch.from_numpy(bank["A"][self.gidx])      # (n, K, T)
        self.pos   = bank["pos"][self.gidx]
        self.spk_meaningful = bank["spk_meaningful"]
        self.att_idxs = data["att_idxs"][self.gidx].astype(np.int64)
        self.subject  = subject_per_window(data)[self.gidx].astype(np.int64)

    def __len__(self):
        return len(self.gidx)

    def _perm(self, i):
        if self.train:
            return torch.randperm(self.K)
        # deterministic at eval, seeded by the GLOBAL window index so every
        # configuration under comparison sees the identical slot assignment
        return torch.from_numpy(
            np.random.default_rng(int(self.gidx[i])).permutation(self.K))

    def __getitem__(self, i):
        perm  = self._perm(i)
        cands = self.A[i][perm]                                  # (K, T)
        label = int((perm == int(self.pos[i])).nonzero()[0].item())
        spk_of_slot = (perm.clone() if self.spk_meaningful else
                       torch.full((self.K,), int(self.att_idxs[i]),
                                  dtype=torch.long))
        return (
            self.eeg[i]   if self.eeg   is not None else None,
            self.video[i] if self.video is not None else None,
            self.gaze[i]  if self.gaze  is not None else None,
            self.imu[i]   if self.imu   is not None else None,
            cands.unsqueeze(-1).contiguous(),                    # (K, T, 1)
            label, spk_of_slot, int(self.att_idxs[i]), int(self.subject[i]),
        )


class SubjectBatchSampler(Sampler):
    """Every batch is drawn from a single listener.

    The contrastive term's in-batch negatives must be within-listener; see the
    AADDataset docstring.
    """

    def __init__(self, subjects, batch_size, shuffle=True, seed=0, min_batch=4):
        self.groups = [np.where(subjects == s)[0] for s in np.unique(subjects)]
        self.bs, self.shuffle = batch_size, shuffle
        self.seed, self.min_batch = seed, min_batch
        self.epoch = 0

    def __iter__(self):
        rng = np.random.default_rng(self.seed + self.epoch)
        self.epoch += 1
        batches = []
        for g in self.groups:
            g = g.copy()
            if self.shuffle:
                rng.shuffle(g)
            for s0 in range(0, len(g), self.bs):
                b = g[s0:s0 + self.bs]
                if len(b) >= self.min_batch:
                    batches.append(b.tolist())
        if self.shuffle:
            rng.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return sum(max(0, len(g) // self.bs) for g in self.groups)


def collate_fn(batch):
    def _stack(i):
        return torch.stack([b[i] for b in batch]) \
               if batch[0][i] is not None else None
    cands = torch.stack([b[4] for b in batch])                   # (B, K, T, 1)
    return (
        _stack(0), _stack(1), _stack(2), _stack(3),
        [cands[:, k] for k in range(cands.shape[1])],            # K x (B, T, 1)
        torch.tensor([b[5] for b in batch], dtype=torch.long),   # label
        torch.stack([b[6] for b in batch]),                      # spk_of_slot
        torch.tensor([b[7] for b in batch], dtype=torch.long),   # attended spk
        torch.tensor([b[8] for b in batch], dtype=torch.long),   # subject
    )


# ── audio-only acceptance probe ────────────────────────────────────────────────

def _shape_features(E: np.ndarray) -> np.ndarray:
    """Eight affine-invariant shape statistics of standardised envelopes.

    Affine-invariant means unchanged by x -> a*x + b, so these survive the
    z-score in `extract_envelope` untouched.  That is precisely why level
    normalisation cannot remove the confound and this probe is needed.
    E: (M, T) -> (M, 8)
    """
    from scipy.signal import welch
    from scipy.stats import kurtosis, skew
    M, T = E.shape
    p5, p95 = np.percentile(E, [5, 95], axis=1)
    f, P = welch(E, fs=TARGET_FS, nperseg=min(256, T), axis=1)
    tot = P.sum(1) + 1e-12

    def band(lo, hi):
        return P[:, (f >= lo) & (f < hi)].sum(1) / tot

    a = np.sort(np.abs(E), axis=1)
    w = np.arange(1, T + 1)
    gini = 2 * (a * w).sum(1) / (T * a.sum(1) + 1e-12) - (T + 1) / T
    return np.stack([kurtosis(E, axis=1), skew(E, axis=1), p95 - p5,
                     (E < -0.5).mean(1), band(0.5, 4), band(4, 8),
                     band(8, 20), gini], axis=1).astype(np.float64)


def audio_only_probe(bank: dict, groups: np.ndarray,
                     n_splits: int = 5) -> float:
    """Can the correct candidate be identified from the AUDIO ALONE?

    Fits a logistic classifier on the shape statistics above, on
    content-disjoint folds, then takes the per-window argmax over candidates.
    A construction free of acoustic confounding must score 1/K.

    This is model-independent and should be run BEFORE training: it certifies
    the task, not the network.  Reference values on this dataset --
    raw 0.5597 (chance 0.25), qmatch 0.2600, shifted_qm binary 0.5002
    (chance 0.50).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GroupKFold

    A, labels = bank["A"], bank["pos"]
    N, K, T = A.shape
    X = _shape_features(A.reshape(N * K, T))
    y = np.zeros(N * K, dtype=np.int64)
    y[np.arange(N) * K + labels] = 1
    g = np.repeat(groups, K)

    oof = np.zeros(N * K)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=g):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return float((oof.reshape(N, K).argmax(1) == labels).mean())


# ── T2 / T3 grouped candidate references ──────────────────────────────────────

# Speaker index (0-based) -> group, for the two binary spatial tasks.
# T2 hemisphere : S1,S2 = left(0), S3,S4 = right(1)
# T3 eccentricity: S2,S3 = inner(0), S1,S4 = outer(1)
SPATIAL_GROUPS = {
    "hemisphere":   {0: 0, 1: 0, 2: 1, 3: 1},
    "eccentricity": {0: 1, 1: 0, 2: 0, 3: 1},
}
SPATIAL_GROUP_NAMES = {
    "hemisphere":   ("left", "right"),
    "eccentricity": ("inner", "outer"),
}


def group_labels(att_idxs: np.ndarray, task: str) -> np.ndarray:
    """Attended loudspeaker index (0..3) -> binary group label for `task`."""
    g = SPATIAL_GROUPS[task]
    return np.array([g[int(i)] for i in att_idxs], dtype=np.int64)


def make_grouped_candidate_bank(data: dict, task: str,
                                construction: str = "qmatch") -> dict:
    """Candidate bank for the binary spatial tasks T2 and T3.

    The task's two references are the per-group MEANS of the four co-present
    talker envelopes, as in the original formulation:

        T2   a_left  = (a_1 + a_2)/2      a_right = (a_3 + a_4)/2
        T3   a_inner = (a_2 + a_3)/2      a_outer = (a_1 + a_4)/2

    WHY THIS NEEDS THE SAME TREATMENT AS T1.  The attended talker is prepared
    differently from its competitors -- not by gain (crest factor, which no gain
    can alter, differs by 9.4 dB) but in the SHAPE of its amplitude envelope,
    and shape statistics are invariant to affine rescaling, so they survive the
    per-candidate z-score untouched.  Averaging two talkers does not remove
    that: whichever reference contains the attended talker inherits its
    signature, and the task is binary, so a decoder with a learned audio encoder
    only has to decide which of two references looks "attended".  Left
    unaddressed this is a strictly easier shortcut than in T1.

    construction:
      "raw"     the two group means, standardised.  CONFOUNDED; retained only to
                reproduce the previous revision.
      "qmatch"  the two group means, distribution-matched (default).  Both
                references then carry the identical multiset of values -- same
                kurtosis, skew, sparsity, dynamic range, silence fraction --
                and differ only in temporal ordering, which is the property a
                neural response tracks.

    Returns the same structure as `make_candidate_bank`, with K = 2 and `pos`
    the attended group, so `AADDataset` consumes it unchanged.
    """
    if task not in SPATIAL_GROUPS:
        raise ValueError(f"unknown task '{task}'; expected one of "
                         f"{tuple(SPATIAL_GROUPS)}")
    if construction not in ("raw", "qmatch"):
        raise ValueError("grouped references support 'raw' or 'qmatch' only; "
                         "same-talker negatives are not defined for a task "
                         "whose classes are spatial groups")

    members = {0: [], 1: []}
    for spk, grp in SPATIAL_GROUPS[task].items():
        members[grp].append(spk)

    A = np.stack([
        np.mean([data["audio"][s][:, :, 0] for s in members[g]], axis=0)
        for g in (0, 1)
    ], axis=1).astype(np.float32)                       # (N, 2, T)

    # standardise each reference, then (optionally) equalise their marginals
    mu = A.mean(axis=2, keepdims=True)
    sd = A.std(axis=2, keepdims=True) + 1e-8
    A = (A - mu) / sd
    if construction == "qmatch":
        A = quantile_match_candidates(A)

    return {"construction": f"{task}_{construction}",
            "A": np.ascontiguousarray(A, dtype=np.float32),
            "pos": group_labels(data["att_idxs"], task),
            "spk_meaningful": True}
