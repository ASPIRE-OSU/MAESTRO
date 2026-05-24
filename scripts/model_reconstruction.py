"""
model_reconstruction.py
-----------------------
Linear backward model for attended speech envelope reconstruction (T4).

The model accepts EEG only or concatenated multimodal input (EEG + Gaze +
IMU + Video) and produces a scalar envelope estimate at each time step via
a single causal Conv1d layer.

Input channel counts (all at 64 Hz)
-------------------------------------
  EEG only         : C = 32
  EEG + VGI        : C = 32 + 6 + 6 + 4 = 48

Usage
-----
  from model_reconstruction import LinearModel, pearson_r, pearson_loss

  model = LinearModel(n_in_channels=32)   # EEG only
  model = LinearModel(n_in_channels=48)   # EEG + VGI

  pred = model(x)                         # x: (B, T, C) → (B, T, 1)
  r    = pearson_r(target, pred)          # (B,)
  loss = pearson_loss(target, pred)       # scalar
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import N_EEG_CH


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
    (B,) — per-sample Pearson r
    """
    y_true = y_true.squeeze(-1)   # (B, T)
    y_pred = y_pred.squeeze(-1)   # (B, T)
    yt     = y_true - y_true.mean(dim=1, keepdim=True)
    yp     = y_pred - y_pred.mean(dim=1, keepdim=True)
    num    = (yt * yp).sum(dim=1)
    denom  = torch.sqrt((yt ** 2).sum(dim=1) * (yp ** 2).sum(dim=1)) + 1e-8
    return num / denom   # (B,)


def pearson_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Negative mean Pearson r — minimise to maximise correlation."""
    return -pearson_r(y_true, y_pred).mean()


# ── Linear backward model ──────────────────────────────────────────────────────

class LinearModel(nn.Module):
    """
    Linear backward model — single causal Conv1d across all input channels.

    Equivalent to the standard linear backward model in the AAD literature.
    Supports EEG-only or multimodal input by adjusting n_in_channels.

    Parameters
    ----------
    integration_window : Conv1d kernel size in samples (default 32 = 0.5s @ 64 Hz)
    n_in_channels      : total input channels (32 for EEG only, 48 for EEG+VGI)
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
    from dataloader import N_EEG_CH, N_GAZE_CH, N_IMU_CH, N_VIDEO_CH, WINDOW_SAMP

    B, T = 4, WINDOW_SAMP

    for label, C in [("EEG only", N_EEG_CH),
                     ("EEG+VGI",  N_EEG_CH + N_GAZE_CH + N_IMU_CH + N_VIDEO_CH)]:
        model = LinearModel(n_in_channels=C)
        x     = torch.randn(B, T, C)
        pred  = model(x)
        tgt   = torch.randn(B, T, 1)
        r     = pearson_r(tgt, pred)
        loss  = pearson_loss(tgt, pred)
        n     = sum(p.numel() for p in model.parameters())
        print(f"{label:<12} C={C:2d}  out={tuple(pred.shape)}  "
              f"r={r.mean().item():.3f}  loss={loss.item():.3f}  "
              f"params={n:,}")