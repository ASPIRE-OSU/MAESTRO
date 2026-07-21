"""
dataloader.py
-------------
Loads, synchronises, preprocesses and windows EEG + audio + video + gaze + IMU
for 4-speaker AAD. Built on top of the new HuggingFace dataset format.
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
from torch.utils.data import Dataset

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

# ── mode / modality registry ────────────────────────────────────────────────────
# Canonical mode -> {active modalities} mapping. All other files should
# call mode_uses(mode) rather than re-deriving this themselves.
#
# Canonical names use full underscored modality lists in a fixed order
# (eeg, gaze, imu, video).

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

# Backward-compatible short aliases -> canonical name
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
    """
    Lightweight bad-channel detection: flat, saturated, or variance-outlier
    channels. eeg_tc is (T, C) time-first. Returns sorted list of bad channel
    names.
    """
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
    """
    Convenience wrapper: filter_reference_eeg() + resample to TARGET_FS.
    Kept for any external callers expecting the old single-call interface;
    load_trial() calls filter_reference_eeg() and _resample() separately
    so filtering can happen on the full trial BEFORE masking to the
    anchor-end alignment window.
    """
    eeg = filter_reference_eeg(eeg_raw, ch_names, fs_in)
    return _resample(eeg, fs_in, TARGET_FS)




# ── audio envelope ─────────────────────────────────────────────────────────────

def extract_envelope(audio: np.ndarray, fs_in: int = AUDIO_FS_RAW) -> np.ndarray:
    """Hilbert, LP 20 Hz, downsample, z-score. Returns (T, 1)."""
    env = np.abs(hilbert(audio.astype(np.float64))).astype(np.float32)
    sos = _butter_lp(20.0, fs_in)
    env = sosfiltfilt(sos, env).astype(np.float32)
    env = _resample(env, fs_in, TARGET_FS)
    return _zscore(env)[:, np.newaxis]


# ── video optical flow ─────────────────────────────────────────────────────────

def extract_optical_flow(video_path: str,
                         t_start: float,
                         t_end: float) -> np.ndarray:
    """Farneback dense optical flow, 4 features/frame, resample to 64 Hz.
    Returns (T_out, 4) float32."""
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
    """Interpolate gaze to 64 Hz grid, LP 10 Hz, z-score. Returns (T, C).
    NaN values are handled per channel by dropping invalid samples before
    interpolation."""
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
    """Interpolate IMU, resample to 64 Hz, LP 20 Hz, z-score. Returns (T, C).
    NaN values are handled per channel by dropping invalid samples before
    interpolation."""
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
    """
    Load timing JSON and derive the per-modality alignment references

    Returns
    -------
    dict with anchor_unix, end_unix, trial_end_sec,
             eeg_first_sample_unix, eeg_t0_internal_sec,
             gaze_trim_sec, imu_trim_sec, video_trim_sec, audio_t0_unix
    """
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

    return {
        "anchor_unix":            anchor_unix,
        "end_unix":               end_unix,
        "trial_end_sec":          trial_end_sec,
        "eeg_first_sample_unix":  t["eeg"]["first_sample_unix"],
        "eeg_t0_internal_sec":    t["eeg"]["t0_internal_sec"],
        "gaze_trim_sec":          gaze_trim_sec,
        "imu_trim_sec":           imu_trim_sec,
        "video_trim_sec":         video_trim_sec,
        "audio_t0_unix":          t["audio"]["t0_unix"],
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
               cache_dir: str | None = None) -> dict | None:
    """
    Load and preprocess one trial from the new dataset format.

    Parameters
    ----------
    local_path       : root of dataset (contains metadata/, data/, media/)
    sid              : subject ID e.g. "S01"
    tid              : trial ID e.g. "eval_001"
    audio_layout     : list of {speaker, filename, azimuth_deg} from audio_layout.json
    attended_speaker : 1-based attended speaker index
    mode             : one of VALID_MODES (canonical or legacy alias)
    cache_dir        : optional cache directory for video/gaze/IMU

    Returns
    -------
    dict: eeg, video, gaze, imu, audio (list of 4), att_idxs
    All arrays: (n_win, WINDOW_SAMP, C). None on failure.
    """
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

    # ── EEG ───────────────────────────────────────────────────────────────────
    # Cached alongside video/gaze/IMU in the same per-(subject,trial) .npz.
    # Every EEG sample's own recorded
    # "t_sec" is converted to unix time and masked to [anchor, anchor+dur],
    # with filtering applied to the full unmasked recording BEFORE masking.
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

            # Filter/notch/reference the FULL unmasked trial recording
            # first, so sosfiltfilt/filtfilt edge transients land outside the
            # anchor-end window rather than inside it. Mask AFTER
            # filtering (and z-scoring), then resample.
            eeg_filtered = filter_reference_eeg(eeg_raw, ch_names, EEG_FS_RAW)
            eeg_proc     = _resample(eeg_filtered[mask], EEG_FS_RAW, TARGET_FS)

            if eeg_cpath:
                _save_cache(eeg_cpath, {"eeg": eeg_proc})
        except Exception as e:
            print(f"  Skipping {sid}/{tid}: EEG error: {e}")
            return None

    # ── Audio envelopes ───────────────────────────────────────────────────────
    # ONE shared reference time
    # (audio.t0_unix) is used for every speaker's waveform.
    audio_dir = root / "media" / "audio" / tid
    audio_t0  = sync["audio_t0_unix"]
    envs = []
    try:
        attendable = sorted(
            [spk for spk in audio_layout if spk.get("attendable", True)],
            key=lambda x: x["speaker"]
        )
        for spk in attendable:
            spk_n   = spk["speaker"]
            matches = sorted(audio_dir.glob(f"speaker{spk_n}_*.flac"))
            if not matches:
                raise FileNotFoundError(
                    f"No audio file found for speaker {spk_n} in {audio_dir}")
            wav, sr = sf.read(str(matches[0]), dtype="float32", always_2d=False)

            i0 = max(0, int(round((anchor_unix - audio_t0) * sr)))
            i1 = int(round((anchor_unix + trial_end_sec - audio_t0) * sr))
            wav = wav[i0:i1]

            envs.append(extract_envelope(wav, sr))
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

    # ── Pad to full window if within 1s ───────────────────────────────────────
    pad_to = ((min_len + WINDOW_SAMP - 1) // WINDOW_SAMP) * WINDOW_SAMP
    if pad_to - min_len <= TARGET_FS:
        min_len = pad_to
    n_win = min_len // WINDOW_SAMP
    if n_win == 0:
        return None

    def _win(x, C):
        if x is None:
            return None
        if len(x) < n_win * WINDOW_SAMP:
            pad = np.zeros((n_win * WINDOW_SAMP - len(x), C), dtype=np.float32)
            x   = np.concatenate([x, pad], axis=0)
        return x[:n_win * WINDOW_SAMP].reshape(n_win, WINDOW_SAMP, C)

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
                  cache_dir: str | None = None) -> dict:
    """
    Build the full pooled dataset dict for a given mode.

    Returns
    -------
    dict: eeg, video, gaze, imu, audio (list of 4 arrays),
          att_idxs, trial_ids, trial_meta_ids, trial_meta_att_idx
    """
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
                                "att_idx":  att_spk - 1})
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
    }

    n_windows = len(dataset["audio"][0])
    n_trials  = len(trial_meta)
    print(f"\nTotal ({mode}): {n_trials} trials, {n_windows} windows")
    from collections import Counter
    dist = Counter(dataset["trial_meta_att_idx"].tolist())
    print("Attended speaker dist (0-based):", dict(sorted(dist.items())))
    return dataset


# ── K-fold splitting ───────────────────────────────────────────────────────────

def get_trial_level_splits(data: dict, n_splits: int = 5, seed: int = 42):
    """Stratified K-fold at trial level, balanced on attended speaker."""
    from sklearn.model_selection import StratifiedKFold
    trial_ids = data["trial_meta_ids"]
    trial_att = data["trial_meta_att_idx"]
    skf       = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                random_state=seed)
    win_ids   = data["trial_ids"]
    for fold, (tr_t, vl_t) in enumerate(skf.split(trial_ids, trial_att)):
        yield (fold,
               np.where(np.isin(win_ids, trial_ids[tr_t]))[0],
               np.where(np.isin(win_ids, trial_ids[vl_t]))[0])


# ── PyTorch Dataset ────────────────────────────────────────────────────────────

class AADDataset(Dataset):
    """AAD Dataset with optional EEG, video, gaze, IMU.
    Speaker order randomised during training."""

    def __init__(self, data: dict, window_idx: np.ndarray,
                 train: bool = True):
        idx        = window_idx
        self.train = train
        self.eeg   = torch.from_numpy(data["eeg"][idx])   \
                     if data["eeg"]   is not None else None
        self.video = torch.from_numpy(data["video"][idx]) \
                     if data["video"] is not None else None
        self.gaze  = torch.from_numpy(data["gaze"][idx])  \
                     if data["gaze"]  is not None else None
        self.imu   = torch.from_numpy(data["imu"][idx])   \
                     if data["imu"]   is not None else None
        self.audio    = [torch.from_numpy(data["audio"][i][idx])
                         for i in range(N_SPEAKERS)]
        self.att_idxs = data["att_idxs"][idx]

    def __len__(self):
        return len(self.audio[0])

    def __getitem__(self, idx):
        att_idx = int(self.att_idxs[idx])
        perm    = torch.randperm(N_SPEAKERS) if self.train else \
                  torch.from_numpy(
                      np.random.default_rng(idx).permutation(N_SPEAKERS))
        audio        = [self.audio[perm[i]][idx] for i in range(N_SPEAKERS)]
        attended_pos = int((perm == att_idx).nonzero(as_tuple=False)[0].item())
        label        = torch.zeros(N_SPEAKERS, dtype=torch.float32)
        label[attended_pos] = 1.0
        return (
            self.eeg[idx]   if self.eeg   is not None else None,
            self.video[idx] if self.video is not None else None,
            self.gaze[idx]  if self.gaze  is not None else None,
            self.imu[idx]   if self.imu   is not None else None,
            audio,
            label,
        )


def collate_fn(batch):
    def _stack(i):
        return torch.stack([b[i] for b in batch]) \
               if batch[0][i] is not None else None
    return (
        _stack(0),
        _stack(1),
        _stack(2),
        _stack(3),
        [torch.stack([b[4][i] for b in batch]) for i in range(N_SPEAKERS)],
        torch.stack([b[5] for b in batch]),
    )