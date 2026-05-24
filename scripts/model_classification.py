"""
model.py
--------
AAD dual-encoder model with modality-specific dilated conv encoders.

Supported modes
---------------
  eeg       : EEG encoder only
  video     : Video encoder only
  gi        : Gaze encoder + IMU encoder → concat → Linear(2D→D)
  eeg_video : EEG + Video encoders → concat → Linear(2D→D)
  eeg_vgi   : EEG + Video + Gaze + IMU → concat → Linear(4D→D)

Encoder depths (layers) per modality
--------------------------------------
  EEG   : 6 layers  — receptive field ~5.7s, full window coverage
  Audio : 6 layers  — same as EEG
  Video : 3 layers  — ~0.4s; optical flow is already a local temporal feature
  Gaze  : 4 layers  — ~1.3s; covers typical fixation duration
  IMU   : 5 layers  — ~4s; head movements span several seconds

All encoders use kernel_size=3, dilation_filters=16, causal padding.
EEG encoder has an additional 1×1 spatial mixing layer (spatial=True).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import (N_EEG_CH, N_VIDEO_CH, N_GAZE_CH, N_IMU_CH,
                        N_SPEAKERS, VALID_MODES)


class DilatedEncoder(nn.Module):
    """
    Causal dilated convolutional encoder.

    Parameters
    ----------
    in_channels      : input channels
    spatial_filters  : channels after optional 1×1 spatial layer (EEG only)
    dilation_filters : channels in each dilated conv layer (D)
    layers           : number of dilated conv layers
    kernel_size      : conv kernel; dilation at layer i = kernel_size**i
    spatial          : if True, prepend a 1×1 spatial mixing conv
    """

    def __init__(self,
                 in_channels: int,
                 spatial_filters: int  = 8,
                 dilation_filters: int = 16,
                 layers: int           = 6,
                 kernel_size: int      = 3,
                 spatial: bool         = False):
        super().__init__()
        self.spatial = spatial

        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters,
                                          kernel_size=1)
            first_in = spatial_filters
        else:
            first_in = in_channels

        self.dil_convs = nn.ModuleList()
        self.acts      = nn.ModuleList()
        ch_in = first_in
        for i in range(layers):
            dilation = kernel_size ** i
            padding  = dilation * (kernel_size - 1)
            self.dil_convs.append(
                nn.Conv1d(ch_in, dilation_filters,
                          kernel_size=kernel_size,
                          dilation=dilation,
                          padding=padding)
            )
            self.acts.append(nn.ReLU())
            ch_in = dilation_filters

        self.out_channels = dilation_filters

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) → (B, T, D)"""
        x = x.transpose(1, 2)                      # (B, C, T)
        if self.spatial:
            x = self.spatial_conv(x)
        for conv, act in zip(self.dil_convs, self.acts):
            x = conv(x)
            if conv.padding[0] > 0:
                x = x[:, :, :-(conv.padding[0])]   # causal trim
            x = act(x)
        return x.transpose(1, 2)                    # (B, T, D)


class AADModel(nn.Module):
    """
    4-speaker AAD model supporting multiple modality combinations.

    Parameters
    ----------
    mode             : one of VALID_MODES
    dilation_filters : embedding dimension D for all encoders
    spatial_filters  : EEG spatial layer width
    """

    def __init__(self,
                 mode: str             = "eeg",
                 dilation_filters: int = 16,
                 spatial_filters: int  = 8):
        super().__init__()
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
        self.mode = mode
        D = dilation_filters

        use_eeg   = mode in ("eeg",   "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi")
        use_video = mode in ("video", "eeg_video", "eeg_vg", "eeg_vgi")
        use_gaze  = mode in ("gaze",  "gi",    "eeg_gaze", "eeg_vg", "eeg_vgi")
        use_imu   = mode in ("imu",   "gi",    "eeg_vgi")

        # ── Brain/scene encoders ─────────────────────────────────────────────
        if use_eeg:
            self.eeg_encoder = DilatedEncoder(
                in_channels=N_EEG_CH, spatial_filters=spatial_filters,
                dilation_filters=D, layers=7, spatial=True
            )
        if use_video:
            self.video_encoder = DilatedEncoder(
                in_channels=N_VIDEO_CH, dilation_filters=D,
                layers=4, spatial=False
            )
        if use_gaze:
            self.gaze_encoder = DilatedEncoder(
                in_channels=N_GAZE_CH, dilation_filters=D,
                layers=6, spatial=False
            )
        if use_imu:
            self.imu_encoder = DilatedEncoder(
                in_channels=N_IMU_CH, dilation_filters=D,
                layers=6, spatial=False
            )

        # ── Fusion projection ─────────────────────────────────────────────────
        # Count how many encoders are active
        n_enc = sum([use_eeg, use_video, use_gaze, use_imu])
        if n_enc > 1:
            self.fusion = nn.Sequential(
                nn.Linear(n_enc * D, D),
                nn.ReLU(),
            )

        # ── Audio encoder (shared weights across all 4 speakers) ──────────────
        self.audio_encoder = DilatedEncoder(
            in_channels=1, dilation_filters=D,
            layers=7, spatial=False
        )

        # ── Cosine sim → scalar score per speaker ─────────────────────────────
        self.sim_proj = nn.Linear(D, 1)

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Mean cosine similarity along time axis. (B,T,D) → (B,D)"""
        return (F.normalize(a, dim=2) * F.normalize(b, dim=2)).mean(dim=1)

    def forward(self,
                eeg:   torch.Tensor,
                video: torch.Tensor,
                gaze:  torch.Tensor,
                imu:   torch.Tensor,
                audio: list) -> torch.Tensor:
        """
        Parameters
        ----------
        eeg   : (B, T, 32) or None
        video : (B, T,  4) or None
        gaze  : (B, T,  6) or None
        imu   : (B, T,  6) or None
        audio : list of N_SPEAKERS tensors, each (B, T, 1)

        Returns
        -------
        probs : (B, N_SPEAKERS)
        """
        # Collect all active embeddings
        embeddings = []
        if eeg   is not None and hasattr(self, 'eeg_encoder'):
            embeddings.append(self.eeg_encoder(eeg))
        if video is not None and hasattr(self, 'video_encoder'):
            embeddings.append(self.video_encoder(video))
        if gaze  is not None and hasattr(self, 'gaze_encoder'):
            embeddings.append(self.gaze_encoder(gaze))
        if imu   is not None and hasattr(self, 'imu_encoder'):
            embeddings.append(self.imu_encoder(imu))

        # Fuse
        if len(embeddings) == 1:
            brain_enc = embeddings[0]                        # (B, T, D)
        elif hasattr(self, 'fusion'):
            brain_enc = self.fusion(
                torch.cat(embeddings, dim=2)                 # (B, T, n*D)
            )                                                # (B, T, D)
        else:
            # Fallback: mean pooling if fusion layer missing (should not happen)
            brain_enc = torch.stack(embeddings, dim=0).mean(dim=0)

        if len(embeddings) == 0:
            raise RuntimeError(
                f"No modality embeddings computed for mode='{self.mode}'. "
                "Check that the correct modality tensors are passed to forward()."
            )

        # Cosine similarity vs each audio embedding
        logits = []
        for aud in audio:
            aud_enc = self.audio_encoder(aud)
            sim     = self._cosine_sim(brain_enc, aud_enc)   # (B, D)
            logits.append(self.sim_proj(sim))                # (B, 1)

        return F.softmax(torch.cat(logits, dim=1), dim=1)   # (B, 4)


# ── quick sanity check ────────────────────────────────────────────────────────

if __name__ == "__main__":
    B, T = 8, 320
    print(f"{'Mode':<12} {'Params':>8}  Output  Sums≈1")
    print("-" * 45)
    for mode in VALID_MODES:
        model = AADModel(mode=mode)
        eeg   = torch.randn(B, T, N_EEG_CH)   if mode in ("eeg",   "eeg_gaze", "eeg_video", "eeg_vg", "eeg_vgi") else None
        video = torch.randn(B, T, N_VIDEO_CH)  if mode in ("video", "eeg_video", "eeg_vg", "eeg_vgi")               else None
        gaze  = torch.randn(B, T, N_GAZE_CH)   if mode in ("gaze",  "gi",    "eeg_gaze", "eeg_vg", "eeg_vgi")       else None
        imu   = torch.randn(B, T, N_IMU_CH)    if mode in ("imu",   "gi",    "eeg_vgi")                                     else None
        audio = [torch.randn(B, T, 1) for _ in range(N_SPEAKERS)]
        probs = model(eeg, video, gaze, imu, audio)
        n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ok    = probs.sum(dim=1).allclose(torch.ones(B), atol=1e-5)
        print(f"{mode:<12} {n:>8,}  {tuple(probs.shape)}  {ok}")