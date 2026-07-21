"""
model_reconstruction.py
-----------------------
Linear backward model for attended speech envelope reconstruction (T4).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import (N_EEG_CH, N_GAZE_CH, N_IMU_CH, N_VIDEO_CH,
                        VALID_MODES, mode_uses)


# ── channel count helper ────────────────────────────────────────────────────────

def n_channels_for_mode(mode: str) -> int:
    """
    Total input channel count for a given mode, derived from
    dataloader.mode_uses() so this stays in sync with every other file's
    modality-activation logic automatically.
    """
    use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
    n = 0
    if use_eeg:   n += N_EEG_CH
    if use_gaze:  n += N_GAZE_CH
    if use_imu:   n += N_IMU_CH
    if use_video: n += N_VIDEO_CH
    return n


# ── Pearson correlation ────────────────────────────────────────────────────────

def pearson_r(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """
    Pearson correlation along the time axis.

    Parameters
    ----------
    y_true : (B, T, 1)
    y_pred : (B, T, 1)

    Returns
    -------
    (B,) — per-sample Pearson r. NaN values (e.g. from constant predictions)
           are replaced with 0.0.
    """
    y_true = y_true.squeeze(-1)   # (B, T)
    y_pred = y_pred.squeeze(-1)   # (B, T)
    yt     = y_true - y_true.mean(dim=1, keepdim=True)
    yp     = y_pred - y_pred.mean(dim=1, keepdim=True)
    num    = (yt * yp).sum(dim=1)
    denom  = torch.sqrt((yt ** 2).sum(dim=1) * (yp ** 2).sum(dim=1)) + 1e-8
    r      = num / denom   # (B,)
    return torch.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)


def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Negative mean Pearson r — minimise to maximise correlation."""
    return -pearson_r(y_true, y_pred).mean()


# ── Linear backward model ──────────────────────────────────────────────────────

class LinearModel(nn.Module):
    """
    Linear backward model — single causal Conv1d across all input channels.

    Equivalent to the standard linear backward model in the AAD literature.
    Supports any modality combination by adjusting n_in_channels (use
    n_channels_for_mode() to compute this from a mode string).

    Parameters
    ----------
    integration_window : Conv1d kernel size in samples (default 32 = 0.5s @ 64 Hz)
    n_in_channels      : total input channels — see n_channels_for_mode()
    """

    def __init__(self,
                 integration_window: int = 32,
                 n_in_channels: int      = N_EEG_CH):
        super().__init__()
        self.padding = integration_window - 1
        self.conv    = nn.Conv1d(n_in_channels, 1,
                                 kernel_size=integration_window,
                                 padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, T, C)  — concatenated input modalities

        Returns
        -------
        (B, T, 1) — reconstructed envelope
        """
        x = x.transpose(1, 2)                       # (B, C, T)
        x = F.pad(x, (self.padding, 0))             # causal left-pad
        x = self.conv(x)                             # (B, 1, T)
        return x.transpose(1, 2)                     # (B, T, 1)


# ── sanity check ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from dataloader import WINDOW_SAMP

    B, T = 4, WINDOW_SAMP

    print(f"{'Mode':<22} {'C':>3}  {'Out':>12}  {'r':>7}  {'loss':>7}  {'Params':>7}")
    print("-" * 68)
    for mode in VALID_MODES:
        C     = n_channels_for_mode(mode)
        model = LinearModel(n_in_channels=C)
        x     = torch.randn(B, T, C)
        pred  = model(x)
        tgt   = torch.randn(B, T, 1)
        r     = pearson_r(tgt, pred)
        loss  = pearson_loss(tgt, pred)
        n     = sum(p.numel() for p in model.parameters())
        print(f"{mode:<22} {C:>3}  {str(tuple(pred.shape)):>12}  "
              f"{r.mean().item():>7.3f}  {loss.item():>7.3f}  {n:>7,}")