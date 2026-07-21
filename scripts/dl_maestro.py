"""
dl_maestro.py
-------------
Downloads the MAESTRO dataset from HuggingFace.
Usage
-----
  python dl_dataset.py --local_dir 'maestro'
"""

import os
import time
import argparse

from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError

REPO_ID   = "aspire-osu/maestro-eeg-dataset"
REPO_TYPE = "dataset"
N_SUBJECTS = 16

# Seconds to wait between batches and after a rate limit hit
DELAY_BETWEEN    = 5
DELAY_RATE_LIMIT = 120
MAX_RETRIES      = 5


def download_pattern(pattern: str, local_dir: str,
                     desc: str = "") -> bool:
    """
    Download files matching a pattern with retry on rate limit.
    Returns True on success, False on permanent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            snapshot_download(
                repo_id     = REPO_ID,
                repo_type   = REPO_TYPE,
                local_dir   = local_dir,
                allow_patterns = [pattern],
                max_workers = 1,
            )
            print(f"  ✓ {desc or pattern}")
            return True

        except HfHubHTTPError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                print(f"  Rate limited on attempt {attempt}/{MAX_RETRIES}. "
                      f"Waiting {DELAY_RATE_LIMIT}s...")
                time.sleep(DELAY_RATE_LIMIT)
            else:
                print(f"  HTTP error: {e}")
                return False

        except Exception as e:
            print(f"  Error on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(DELAY_BETWEEN)
            else:
                return False

    print(f"  ✗ Failed after {MAX_RETRIES} attempts: {desc or pattern}")
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--local_dir", default="maestro",
                   help="Local directory to save the dataset")
    p.add_argument("--subjects",  nargs="+", type=int,
                   default=list(range(1, N_SUBJECTS + 1)),
                   help="Subject numbers to download (default: all 1-16)")
    args = p.parse_args()

    os.makedirs(args.local_dir, exist_ok=True)
    print(f"Downloading MAESTRO to: {args.local_dir}")
    print(f"Subjects: {args.subjects}\n")

    # ── Step 1: metadata + root files (small, download first) ────────────────
    print("[ 1 / 3 ] Downloading metadata and root files...")
    for pattern, desc in [
        ("metadata/*",     "metadata/"),
        ("*.md",           "README.md"),
        ("LICENSE",        "LICENSE"),
        (".gitattributes", ".gitattributes"),
    ]:
        download_pattern(pattern, args.local_dir, desc)
        time.sleep(DELAY_BETWEEN)

    # ── Step 2: per-subject data (EEG + gaze + IMU parquet files) ─────────────
    print("\n[ 2 / 3 ] Downloading modality data (per subject)...")
    n = len(args.subjects)
    for i, s in enumerate(args.subjects, 1):
        sid = f"S{s:02d}"
        print(f"  Subject {sid}  [{i}/{n}]")

        for modality in ("eeg", "gaze", "imu"):
            pattern = f"data/{modality}/subject={sid}/*"
            ok = download_pattern(pattern, args.local_dir,
                                  f"  {modality}/{sid}")
            if not ok:
                print(f"  Warning: failed to download {modality} for {sid}")
            time.sleep(DELAY_BETWEEN)

    # ── Step 3: per-subject media (audio + video + timing) ───────────────────
    print("\n[ 3 / 3 ] Downloading media (per subject)...")

    # Audio is organised by trial, not by subject — download all at once
    print("  Downloading audio (all trials)...")
    download_pattern("media/audio/*", args.local_dir, "media/audio/")
    time.sleep(DELAY_BETWEEN)

    for i, s in enumerate(args.subjects, 1):
        sid = f"S{s:02d}"
        print(f"  Subject {sid} media  [{i}/{n}]")

        for subdir in ("video", "timing"):
            pattern = f"media/{subdir}/subject={sid}/*"
            ok = download_pattern(pattern, args.local_dir,
                                  f"  {subdir}/{sid}")
            if not ok:
                print(f"  Warning: failed to download {subdir} for {sid}")
            time.sleep(DELAY_BETWEEN)

    print("\nDownload complete.")
    print(f"Dataset saved to: {os.path.abspath(args.local_dir)}")


if __name__ == "__main__":
    main()