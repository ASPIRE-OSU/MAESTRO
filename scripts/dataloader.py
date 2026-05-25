"""
dataloader.py
-------------
Loads, synchronises, preprocesses and windows EEG + audio + video + gaze + IMU
for 4-speaker AAD. Built on top of the new HuggingFace dataset format.

New dataset format
------------------
  <local_path>/
    metadata/
      trials.csv
      trials_per_subject.csv
      bad_channels.csv
      eeg_channels.json
      audio_layout.json
      audio_manifest.json
    data/
      eeg/subject=S01/trial=eval_001.parquet     (t_sec, sample_idx, ch_Fp1, ...)
      gaze/subject=S01/trial=eval_001.parquet
      imu/subject=S01/trial=eval_001.parquet
    media/
      audio/<trial_id>/<speaker_file>.flac
      video/subject=S01/eval_001.mp4
      timing/subject=S01/trial=eval_001.json     (unified sync timestamps)

Sync (timing JSON)
------------------
  {
    "eeg":  { "first_sample_unix": <float>, ... },
    "gaze": { "first_sample_unix": <float>, ... },
    "audio_devices": [
      { "device_id_raw": 6, "playback_start_unix": <float>, ... },
      ...
    ]
  }

  earliest_playback = min(d["playback_start_unix"] for d in audio_devices)
  eeg_trim_sec      = earliest_playback - eeg["first_sample_unix"]
  gaze_trim_sec     = earliest_playback - gaze["first_sample_unix"]

Feature dimensions (all resampled to 64 Hz)
--------------------------------------------
  EEG   : (T, 32)
  Video : (T,  4)   mean_mag, std_mag, mean_flow_x, mean_flow_y
  Gaze  : (T,  6)   from gaze parquet columns
  IMU   : (T,  6)   from imu parquet columns
  Audio : (T,  1)   amplitude envelope per speaker

Supported modes
---------------
  eeg, gaze, imu, video, gi, eeg_gaze, eeg_video, eeg_vg, eeg_vgi
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
from scipy.signal import butter, hilbert, resample_poly, sosfiltfilt
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

VALID_MODES = (
    "eeg", "gaze", "imu", "video", "gi",
    "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi"
)

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


# ── EEG preprocessing ──────────────────────────────────────────────────────────

def preprocess_eeg(eeg_raw: np.ndarray, fs_in: int = EEG_FS_RAW) -> np.ndarray:
    """Bandpass 1-40 Hz, CAR, downsample to 64 Hz. Returns (T, 32)."""
    sos = _butter_bp(1.0, 40.0, fs_in)
    eeg = sosfiltfilt(sos, eeg_raw, axis=0).astype(np.float32)
    eeg -= eeg.mean(axis=1, keepdims=True)
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
    interpolation, matching the original gazedata.gz behavior."""
    mask = (t_sec >= t_start - 0.1) & (t_sec <= t_end + 0.1)
    ts, vals = t_sec[mask], gaze_raw[mask]
    n_out = int((t_end - t_start) * TARGET_FS)
    C = vals.shape[1] if len(vals) > 0 else gaze_raw.shape[1]
    if len(ts) < 4:
        return np.zeros((n_out, C), dtype=np.float32)

    # Interpolate each channel independently, skipping NaN samples
    grid = np.arange(t_start, t_end, 1.0 / TARGET_FS)
    out  = np.zeros((len(grid), C), dtype=np.float32)
    for c in range(C):
        valid = ~np.isnan(vals[:, c])
        if valid.sum() < 4:
            continue   # leave as zero if too few valid samples
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

    fs_imu  = 1.0 / np.median(np.diff(ts)) if len(ts) > 1 else 130.0
    n_imu   = int((t_end - t_start) * fs_imu)

    # Interpolate each channel independently, skipping NaN samples
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

# Device IDs that carry speech — device 6 carries noise only and is excluded
# from sync to match the original data collection protocol.
SPEECH_DEVICE_IDS = {3, 5}

def _load_sync(timing_path: str) -> dict:
    """Load timing JSON, return eeg_trim_sec and gaze_trim_sec.
    Only speech devices (3 and 5) are used for sync, excluding the
    noise-only device (6) to match the original sync behavior."""
    with open(timing_path) as f:
        t = json.load(f)
    speech_devices = [d for d in t["audio_devices"]
                      if d["device_id_raw"] in SPEECH_DEVICE_IDS]
    earliest = min(d["playback_start_unix"] for d in speech_devices)
    return {
        "eeg_trim_sec":  earliest - t["eeg"]["first_sample_unix"],
        "gaze_trim_sec": earliest - t["gaze"]["first_sample_unix"],
    }


# ── cache helpers ──────────────────────────────────────────────────────────────

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
    mode             : one of VALID_MODES
    cache_dir        : optional cache directory for video/gaze/IMU

    Returns
    -------
    dict: eeg, video, gaze, imu, audio (list of 4), att_idxs
    All arrays: (n_win, WINDOW_SAMP, C). None on failure.
    """
    root = Path(local_path)

    use_eeg   = mode in ("eeg",   "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi")
    use_video = mode in ("video", "eeg_video", "eeg_vg", "eeg_vgi")
    use_gaze  = mode in ("gaze",  "gi",        "eeg_gaze", "eeg_vg", "eeg_vgi")
    use_imu   = mode in ("imu",   "gi",        "eeg_vgi")

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

    eeg_trim_sec  = sync["eeg_trim_sec"]
    gaze_trim_sec = sync["gaze_trim_sec"]

    # ── EEG ───────────────────────────────────────────────────────────────────
    eeg_proc = None
    if use_eeg:
        eeg_path = (root / "data" / "eeg" /
                    f"subject={sid}" / f"trial={tid}.parquet")
        if not eeg_path.exists():
            print(f"  Skipping {sid}/{tid}: EEG parquet not found")
            return None
        try:
            df       = pd.read_parquet(eeg_path)
            ch_cols  = [c for c in df.columns if c.startswith("ch_")]
            eeg_raw  = df[ch_cols].to_numpy(dtype=np.float32)
            trim_smp = int(eeg_trim_sec * EEG_FS_RAW)
            eeg_proc = preprocess_eeg(eeg_raw[trim_smp:], EEG_FS_RAW)
        except Exception as e:
            print(f"  Skipping {sid}/{tid}: EEG error: {e}")
            return None

    # ── Audio envelopes ───────────────────────────────────────────────────────
    audio_dir = root / "media" / "audio" / tid
    envs      = []
    try:
        # Files are named: speaker{N}_dev{D}_{L|R}_spkid{ID}.flac
        # Match by speaker number prefix since spkid varies per trial
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
            min_sec = min(lengths) / TARGET_FS
            t_start = gaze_trim_sec
            t_end   = t_start + min_sec
            computed = {}

            if use_video:
                vpath = (root / "media" / "video" /
                         f"subject={sid}" / f"{tid}.mp4")
                try:
                    computed["video"] = extract_optical_flow(
                        str(vpath), t_start, t_end)
                except Exception as e:
                    print(f"  Skipping {sid}/{tid}: video error: {e}")
                    return None

            if use_gaze:
                gpath = (root / "data" / "gaze" /
                         f"subject={sid}" / f"trial={tid}.parquet")
                try:
                    gdf   = pd.read_parquet(gpath)
                    t_sec = gdf["t"].to_numpy()
                    # Pupil: right eye, falling back to left if right is NaN
                    gdf["pupil"] = gdf["R_pupil"].fillna(gdf["L_pupil"])
                    # 6 channels: gaze2d(2) + gaze3d(3) + pupil(1)
                    gcols = ["gaze2d_x", "gaze2d_y",
                             "gaze3d_x", "gaze3d_y", "gaze3d_z",
                             "pupil"]
                    computed["gaze"] = preprocess_gaze(
                        gdf[gcols].to_numpy(dtype=np.float32),
                        t_sec, t_start, t_end)
                except Exception as e:
                    print(f"  Skipping {sid}/{tid}: gaze error: {e}")
                    return None

            if use_imu:
                ipath = (root / "data" / "imu" /
                         f"subject={sid}" / f"trial={tid}.parquet")
                try:
                    idf   = pd.read_parquet(ipath)
                    t_sec = idf["t"].to_numpy()
                    # 6 channels: accelerometer(3) + gyroscope(3)
                    icols = ["ax", "ay", "az", "gx", "gy", "gz"]
                    computed["imu"] = preprocess_imu(
                        idf[icols].to_numpy(dtype=np.float32),
                        t_sec, t_start, t_end)
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

    Parameters
    ----------
    local_path  : root of the new HuggingFace dataset
    mode        : one of VALID_MODES
    subjects    : list of ints (1-16) or "all"
    trials      : "main" or "all"
    cache_dir   : optional cache for video/gaze/IMU features

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

    use_eeg   = mode in ("eeg",   "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi")
    use_video = mode in ("video", "eeg_video", "eeg_vg", "eeg_vgi")
    use_gaze  = mode in ("gaze",  "gi",        "eeg_gaze", "eeg_vg", "eeg_vgi")
    use_imu   = mode in ("imu",   "gi",        "eeg_vgi")

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