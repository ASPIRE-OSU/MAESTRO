"""
model_classification.py
--------------------------------
4-speaker AAD model for T1 task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import (N_EEG_CH, N_VIDEO_CH, N_GAZE_CH, N_IMU_CH,
                        N_SPEAKERS, VALID_MODES, mode_uses)


class DilatedEncoder(nn.Module):
    """Causal dilated convolutional encoder. Identical to the gated variant."""

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
        """x: (B, T, C) -> (B, T, D_mod)"""
        x = x.transpose(1, 2)
        if self.spatial:
            x = self.spatial_conv(x)
        for conv, act in zip(self.dil_convs, self.acts):
            x = conv(x)
            if conv.padding[0] > 0:
                x = x[:, :, :-(conv.padding[0])]
            x = act(x)
        return x.transpose(1, 2)


class AADModel(nn.Module):
    """
    4-speaker AAD model with FIXED CONCATENATION fusion (Option 1).

    Parameters
    ----------
    mode             : one of VALID_MODES
    D_eeg            : EEG encoder embedding width
    D_gaze           : Gaze encoder embedding width
    D_imu            : IMU encoder embedding width
    D_video          : Video encoder embedding width
    D_common         : shared dimension every modality (and audio) is
                       projected into before fusion/cosine similarity
    spatial_filters  : EEG spatial layer width
    """

    def __init__(self,
                 mode: str            = "eeg",
                 D_eeg: int           = 16,
                 D_gaze: int          = 16,
                 D_imu: int           = 16,
                 D_video: int         = 16,
                 D_common: int        = 16,
                 spatial_filters: int = 8):
        super().__init__()
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
        self.mode     = mode
        self.D_common = D_common

        use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)

        if use_eeg:
            self.eeg_encoder = DilatedEncoder(
                in_channels=N_EEG_CH, spatial_filters=spatial_filters,
                dilation_filters=D_eeg, layers=7, spatial=True
            )
            self.eeg_proj = nn.Linear(D_eeg, D_common)
        if use_video:
            self.video_encoder = DilatedEncoder(
                in_channels=N_VIDEO_CH, dilation_filters=D_video,
                layers=4, spatial=False
            )
            self.video_proj = nn.Linear(D_video, D_common)
        if use_gaze:
            self.gaze_encoder = DilatedEncoder(
                in_channels=N_GAZE_CH, dilation_filters=D_gaze,
                layers=6, spatial=False
            )
            self.gaze_proj = nn.Linear(D_gaze, D_common)
        if use_imu:
            self.imu_encoder = DilatedEncoder(
                in_channels=N_IMU_CH, dilation_filters=D_imu,
                layers=6, spatial=False
            )
            self.imu_proj = nn.Linear(D_imu, D_common)

        # ── FIXED CONCATENATION FUSION ─────────────────────────────────────────
        n_enc = sum([use_eeg, use_video, use_gaze, use_imu])
        if n_enc > 1:
            self.fusion = nn.Sequential(
                nn.Linear(n_enc * D_common, D_common),
                nn.ReLU(),
            )

        self.audio_encoder = DilatedEncoder(
            in_channels=1, dilation_filters=D_common,
            layers=7, spatial=False
        )

        self.sim_proj = nn.Linear(D_common, 1)

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (F.normalize(a, dim=2) * F.normalize(b, dim=2)).mean(dim=1)

    def forward(self,
                eeg:   torch.Tensor,
                video: torch.Tensor,
                gaze:  torch.Tensor,
                imu:   torch.Tensor,
                audio: list) -> torch.Tensor:
        """
        Returns
        -------
        probs : (B, N_SPEAKERS)
        """
        embeddings = []
        if eeg   is not None and hasattr(self, 'eeg_encoder'):
            embeddings.append(self.eeg_proj(self.eeg_encoder(eeg)))
        if video is not None and hasattr(self, 'video_encoder'):
            embeddings.append(self.video_proj(self.video_encoder(video)))
        if gaze  is not None and hasattr(self, 'gaze_encoder'):
            embeddings.append(self.gaze_proj(self.gaze_encoder(gaze)))
        if imu   is not None and hasattr(self, 'imu_encoder'):
            embeddings.append(self.imu_proj(self.imu_encoder(imu)))

        if len(embeddings) == 0:
            raise RuntimeError(
                f"No modality embeddings computed for mode='{self.mode}'.")

        if len(embeddings) == 1:
            brain_enc = embeddings[0]
        elif hasattr(self, 'fusion'):
            brain_enc = self.fusion(torch.cat(embeddings, dim=2))
        else:
            brain_enc = torch.stack(embeddings, dim=0).mean(dim=0)

        logits = []
        for aud in audio:
            aud_enc = self.audio_encoder(aud)
            sim     = self._cosine_sim(brain_enc, aud_enc)
            logits.append(self.sim_proj(sim))

        return F.softmax(torch.cat(logits, dim=1), dim=1)


if __name__ == "__main__":
    B, T = 8, 320
    print(f"{'Mode':<22} {'Params':>8}  Output  Sums≈1")
    print("-" * 55)
    for mode in VALID_MODES:
        model = AADModel(mode=mode)
        use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
        eeg   = torch.randn(B, T, N_EEG_CH)   if use_eeg   else None
        video = torch.randn(B, T, N_VIDEO_CH) if use_video else None
        gaze  = torch.randn(B, T, N_GAZE_CH)  if use_gaze  else None
        imu   = torch.randn(B, T, N_IMU_CH)   if use_imu   else None
        audio = [torch.randn(B, T, 1) for _ in range(N_SPEAKERS)]
        probs = model(eeg, video, gaze, imu, audio)
        n     = sum(p.numel() for p in model.parameters() if p.requires_grad)
        ok    = probs.sum(dim=1).allclose(torch.ones(B), atol=1e-5)
        print(f"{mode:<22} {n:>8,}  {tuple(probs.shape)}  {ok}")