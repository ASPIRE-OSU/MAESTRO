"""
model_classification.py
-----------------------
Attended-talker decoder.

This is the leakage-controlled version of the model. Five defects in the
previous revision let the network reach its reported accuracy without using the
physiological recording at all; permuting the recordings across test windows
changed accuracy by 0.0009, and feeding zeros in place of the recording changed
it by nothing. The changes here, and why each is needed:

1. RECEPTIVE FIELD.  Dilations were `3**i` over 7 layers, giving a receptive
   field of 3^7 = 2187 samples = 34.2 s against a 640-sample (10 s) window. The
   deep layers convolved mostly zero padding -- on average 85 % of what the last
   layer saw was padding, which is identical in every window, so the output
   stopped depending on the input. Now `2**i` over 5 layers: 63 samples =
   0.98 s, matched to the 0-400 ms cortical response to a speech envelope.

2. NORMALISATION.  There was none. A stack of rectified layers without it drifts
   into a near-constant regime. GroupNorm now follows every convolution.

3. FINAL ACTIVATION.  A ReLU on the last layer forces embeddings to be
   non-negative, and two non-negative vectors have cosine similarity in [0,1]
   with an expectation of 1/(1+(sigma/mu)^2) -- near 1 whenever the coordinates
   vary little. Since the score IS a correlation, this compressed the very
   quantity being measured. Measured similarity between different windows was
   0.9995. The last layer is now linear.

4. DIRECTION.  The encoder was causal (past-only), but the response to audio at
   time t appears in EEG at t+100..300 ms, i.e. in the future relative to t. The
   encoder is now centred. Explicit directional modes remain available for the
   lag-band control (see `direction`).

5. SCORING.  The old head computed  mean_t[ normalize(b) * normalize(a_k) ]  and
   passed it through a linear layer with bias. If the encoder emits a constant
   vector c at every time step, that reduces exactly to

       logit_k = <w (*) c_hat, mean_t(a_hat_k)> + beta

   -- a linear classifier on the audio alone, with precisely the capacity needed
   to read the acoustic differences between target and masker recordings. The
   architecture therefore CONTAINED a working audio-only decoder, and gradient
   descent found it because it is the easier optimum. `CouplingHead` centres both
   signals over time before correlating, so a time-constant embedding yields
   exactly zero against every candidate, all scores tie and accuracy is pinned at
   chance. The degenerate solution is not penalised; it is unreachable.

Behavioural modalities (gaze, head IMU, scene video) previously went through the
same envelope-matching head as EEG. They bear no temporal relationship to a
speech envelope, so that is not a well-posed task for them and they collapsed
too. They now receive `SpatialHead`, which predicts the attended loudspeaker and
takes NO audio input, and is therefore structurally incapable of using an
acoustic shortcut.

The previous revision is preserved verbatim as `model_classification_legacy.py`
so the ablation remains runnable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataloader import (N_EEG_CH, N_VIDEO_CH, N_GAZE_CH, N_IMU_CH,
                        N_SPEAKERS, VALID_MODES, mode_uses)


# ── gradient reversal, for the audio-only adversary ───────────────────────────

class _GradientReversal(torch.autograd.Function):
    """Identity forwards; multiplies the gradient by -lambda backwards."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam: float = 1.0):
    return _GradientReversal.apply(x, lam)


# ── encoder ───────────────────────────────────────────────────────────────────

class DilatedEncoder(nn.Module):
    """Dilated temporal convolution stack.

    Parameters
    ----------
    in_channels      : input channel count
    spatial_filters  : width of the 1x1 spatial layer (EEG only)
    dilation_filters : embedding width D
    layers           : number of dilated layers
    kernel_size      : convolution kernel size
    spatial          : prepend a 1x1 convolution over channels (EEG only)
    direction        : "centred" (default) sees +-RF/2 around t
                       "past"    sees [t-RF, t]
                       "future"  sees [t, t+RF]
                       The directional modes exist for the lag-band control: a
                       genuine evoked response can only occupy positive lags,
                       whereas an artifact from stimulus playback bleeding into
                       the recording sits at zero lag and is symmetric in time.
    dropout          : dropout applied after each hidden layer
    """

    def __init__(self,
                 in_channels: int,
                 spatial_filters: int = 8,
                 dilation_filters: int = 16,
                 layers: int = 5,
                 kernel_size: int = 3,
                 spatial: bool = False,
                 direction: str = "centred",
                 dropout: float = 0.1):
        super().__init__()
        assert direction in ("centred", "past", "future")
        self.spatial = spatial
        self.direction = direction
        self.n_layers = layers

        if spatial:
            self.spatial_conv = nn.Conv1d(in_channels, spatial_filters,
                                          kernel_size=1)
            first_in = spatial_filters
        else:
            first_in = in_channels

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.trims = []
        ch_in = first_in
        for i in range(layers):
            dilation = 2 ** i
            full = dilation * (kernel_size - 1)
            pad = full if direction in ("past", "future") else full // 2
            self.convs.append(
                nn.Conv1d(ch_in, dilation_filters, kernel_size,
                          dilation=dilation, padding=pad)
            )
            self.norms.append(
                nn.GroupNorm(min(4, dilation_filters), dilation_filters)
            )
            self.trims.append(pad if direction in ("past", "future") else 0)
            ch_in = dilation_filters

        self.drop = nn.Dropout(dropout)
        self.out_channels = dilation_filters
        self.receptive_field = 1 + sum(2 ** i * (kernel_size - 1)
                                       for i in range(layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, C) -> (B, T, D)"""
        x = x.transpose(1, 2)
        if self.spatial:
            x = self.spatial_conv(x)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            x = conv(x)
            if self.trims[i] > 0:
                x = (x[:, :, :-self.trims[i]] if self.direction == "past"
                     else x[:, :, self.trims[i]:])
            x = norm(x)
            if i < self.n_layers - 1:      # no activation on the final layer
                x = self.drop(F.relu(x))
        return x.transpose(1, 2)


# ── scoring heads ─────────────────────────────────────────────────────────────

class CouplingHead(nn.Module):
    """Time-centred correlation between a recording and each candidate.

        c_d(k)  = Pearson_t( b_d , a_{k,d} )
        s_k     = tau * <w, c(k)>                    (w has NO bias term)
        score_k = s_k - mean_j s_j                   (candidate-centring)

    Guarantee: if the recording embedding is constant over time, subtracting its
    temporal mean gives exactly zero, so every correlation is zero, every score
    is zero, all candidates tie and accuracy is pinned at 1/K. There is no
    constant-embedding solution for the optimiser to find.

    Two further properties follow from the same algebra. The correlation is
    unchanged by rescaling a candidate, corr(b, alpha*a + beta) = corr(b, a) for
    alpha > 0, so a candidate cannot be favoured by being louder. And
    candidate-centring cancels anything added equally to all candidates, which
    is also why `w` carries no bias: a bias shifts every score identically and
    can never change which candidate wins.
    """

    def __init__(self, d_common: int, init_tau: float = 0.07):
        super().__init__()
        self.w = nn.Linear(d_common, 1, bias=False)
        self.log_tau = nn.Parameter(torch.tensor(math.log(1.0 / init_tau)))

    @staticmethod
    def corr(b: torch.Tensor, a: torch.Tensor, eps: float = 1e-6):
        """Per-dimension Pearson correlation over time. (B,T,D) -> (B,D)."""
        b = b - b.mean(1, keepdim=True)          # <- removes the DC component
        a = a - a.mean(1, keepdim=True)
        b = b / (b.norm(dim=1, keepdim=True) + eps)
        a = a / (a.norm(dim=1, keepdim=True) + eps)
        return (b * a).sum(1)

    def score_from_corr(self, c: torch.Tensor) -> torch.Tensor:
        return self.w(c).squeeze(-1) * self.log_tau.exp()

    def forward(self, brain_enc, audio_encs) -> torch.Tensor:
        s = torch.stack([self.score_from_corr(self.corr(brain_enc, a))
                         for a in audio_encs], dim=1)          # (B, K)
        return s - s.mean(1, keepdim=True)


class SpatialHead(nn.Module):
    """Predicts the attended loudspeaker index from a modality embedding.

    Takes NO audio input, so it cannot use an acoustic shortcut even in
    principle. This also makes its permutation null exactly 1/K: under a shuffle
    the branch sees an unrelated window's recording, and with balanced labels the
    chance of a match is 1/K regardless of how biased the classifier is.
    Whatever it scores above chance is therefore attributable to the recording.
    """

    def __init__(self, d_in: int, n_spk: int = N_SPEAKERS,
                 hidden: int = 32, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_in, hidden), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(hidden, n_spk),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """(B, T, D) -> (B, n_spk)"""
        return self.net(torch.cat([emb.mean(1), emb.std(1)], dim=-1))


class AudioOnlyAdversary(nn.Module):
    """Tries to name the correct candidate from the audio embeddings alone.

    Its own parameters train normally; the audio encoder receives the reversed
    gradient, so any residual acoustic cue this head can read is actively
    unlearned by the encoder.
    """

    def __init__(self, d_common: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2 * d_common, hidden), nn.ReLU(), nn.Linear(hidden, 1),
        )

    def forward(self, audio_encs, lam: float = 1.0) -> torch.Tensor:
        f = torch.stack([torch.cat([a.mean(1), a.std(1)], dim=-1)
                         for a in audio_encs], dim=1)          # (B, K, 2D)
        return self.net(grad_reverse(f, lam)).squeeze(-1)       # (B, K)


# ── model ─────────────────────────────────────────────────────────────────────

MODALITY_CHANNELS = {
    "eeg":   N_EEG_CH,
    "gaze":  N_GAZE_CH,
    "imu":   N_IMU_CH,
    "video": N_VIDEO_CH,
}


class AADModel(nn.Module):
    """Attended-talker decoder with two branches.

    COUPLING BRANCH (EEG only)  - does the recording's time course match this
        candidate's time course?  EEG is the only modality with a temporal
        relationship to a speech envelope.

    ORIENTATION BRANCH (all modalities) - which loudspeaker was this listener
        oriented toward?  Receives no audio, so it cannot shortcut.

    The two are combined at the score level: orientation logits are over
    loudspeaker index and are mapped into candidate-slot order through the
    window's permutation before being added to the coupling scores.

    Parameters
    ----------
    mode             : one of VALID_MODES
    D_*              : per-modality embedding widths
    D_common         : shared width every modality and the audio project into
    spatial_filters  : EEG spatial layer width
    layers           : dilated layers per encoder
    direction        : encoder receptive-field direction (see DilatedEncoder)
    modality_dropout : probability of withholding each modality during training;
                       without it the strongest branch absorbs the gradient and
                       the others are never trained
    adversary        : enable the audio-only adversary
    """

    def __init__(self,
                 mode: str = "eeg",
                 D_eeg: int = 16,
                 D_gaze: int = 16,
                 D_imu: int = 16,
                 D_video: int = 16,
                 D_common: int = 16,
                 spatial_filters: int = 8,
                 layers: int = 5,
                 direction: str = "centred",
                 dropout: float = 0.1,
                 modality_dropout: float = 0.3,
                 adversary: bool = True,
                 n_classes: int = N_SPEAKERS):
        super().__init__()
        assert mode in VALID_MODES, f"mode must be one of {VALID_MODES}"
        self.mode = mode
        self.D_common = D_common
        self.modality_dropout = modality_dropout
        # Number of classes the ORIENTATION branch predicts, and hence the
        # number of candidate slots.  4 for T1 (loudspeaker index); 2 for the
        # binary spatial tasks T2/T3, where the classes are the two grouped
        # references (left/right, inner/outer).
        self.n_classes = n_classes

        use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
        widths = {"eeg": D_eeg, "gaze": D_gaze, "imu": D_imu, "video": D_video}
        self.modalities = tuple(
            m for m, use in (("eeg", use_eeg), ("gaze", use_gaze),
                             ("imu", use_imu), ("video", use_video)) if use
        )

        self.encoders = nn.ModuleDict()
        self.projs = nn.ModuleDict()
        for m in self.modalities:
            self.encoders[m] = DilatedEncoder(
                in_channels=MODALITY_CHANNELS[m],
                spatial_filters=spatial_filters,
                dilation_filters=widths[m],
                layers=layers,
                spatial=(m == "eeg"),
                direction=direction,
                dropout=dropout,
            )
            self.projs[m] = nn.Linear(widths[m], D_common)

        # coupling branch: EEG only
        self.couple_mod = "eeg" if "eeg" in self.modalities else None
        if self.couple_mod is not None:
            self.audio_encoder = DilatedEncoder(
                in_channels=1, dilation_filters=D_common, layers=layers,
                spatial=False, direction=direction, dropout=dropout,
            )
            self.head = CouplingHead(D_common)
            self.adversary = AudioOnlyAdversary(D_common) if adversary else None
        else:
            self.adversary = None

        # orientation branch: every modality present
        self.spatial_heads = nn.ModuleDict(
            {m: SpatialHead(D_common, n_spk=n_classes) for m in self.modalities}
        )
        if len(self.modalities) > 1:
            self.spatial_fuse = SpatialHead(D_common * len(self.modalities),
                                            n_spk=n_classes)

    # ── pieces ────────────────────────────────────────────────────────────────

    def encode(self, inputs: dict) -> dict:
        """{modality: (B,T,C)} -> {modality: (B,T,D_common)}"""
        out = {}
        for m in self.modalities:
            x = inputs.get(m)
            if x is not None:
                out[m] = self.projs[m](self.encoders[m](x))
        return out

    def encode_audio(self, audio):
        return [self.audio_encoder(a) for a in audio]

    def forward(self,
                eeg=None, video=None, gaze=None, imu=None,
                audio=None, spk_of_slot=None,
                brain_override=None, adv_lam: float = 1.0) -> dict:
        """
        Parameters
        ----------
        eeg/video/gaze/imu : (B, T, C) or None
        audio              : list of K tensors (B, T, 1); None for orientation-only
        spk_of_slot        : (B, K) long -- loudspeaker index occupying each slot,
                             used to map orientation logits into slot order
        brain_override     : substitute embedding, used by the permutation null

        Returns
        -------
        dict with `logits` (B, K) and the intermediates the loss needs.
        """
        embs = self.encode({"eeg": eeg, "video": video,
                            "gaze": gaze, "imu": imu})

        if self.training and self.modality_dropout > 0 and len(embs) > 1:
            keep = [m for m in embs if torch.rand(()) > self.modality_dropout]
            if not keep:                      # never drop every modality
                keep = [self.couple_mod or self.modalities[0]]
            embs = {m: embs[m] for m in keep}

        out = {"embs": embs, "logits": None, "couple_logits": None,
               "spk_logits": {}, "audio_encs": None, "brain_enc": None}

        # coupling branch
        brain_enc = embs.get(self.couple_mod) if self.couple_mod else None
        if brain_override is not None:
            brain_enc = brain_override
        if brain_enc is not None and audio is not None:
            audio_encs = self.encode_audio(audio)
            out["brain_enc"] = brain_enc
            out["audio_encs"] = audio_encs
            out["couple_logits"] = self.head(brain_enc, audio_encs)
            if self.adversary is not None:
                out["adv_logits"] = self.adversary(audio_encs, adv_lam)

        # orientation branch
        for m in self.modalities:
            if m in embs:
                out["spk_logits"][m] = self.spatial_heads[m](embs[m])
        if len(self.modalities) > 1 and embs:
            ref = next(iter(embs.values()))
            cat = torch.cat([embs[m] if m in embs else torch.zeros_like(ref)
                             for m in self.modalities], dim=-1)
            out["spk_logits"]["fused"] = self.spatial_fuse(cat)

        logits = out["couple_logits"]
        if out["spk_logits"] and spk_of_slot is not None:
            key = ("fused" if "fused" in out["spk_logits"]
                   else next(iter(out["spk_logits"])))
            slot = torch.gather(out["spk_logits"][key], 1, spk_of_slot)
            logits = slot if logits is None else logits + slot
        out["logits"] = logits
        return out


if __name__ == "__main__":
    B, T = 8, 640
    print(f"{'Mode':<22} {'Params':>8}  {'RF':>6}  Output")
    print("-" * 52)
    for mode in VALID_MODES:
        model = AADModel(mode=mode)
        use_eeg, use_gaze, use_imu, use_video = mode_uses(mode)
        kw = dict(
            eeg=torch.randn(B, T, N_EEG_CH) if use_eeg else None,
            video=torch.randn(B, T, N_VIDEO_CH) if use_video else None,
            gaze=torch.randn(B, T, N_GAZE_CH) if use_gaze else None,
            imu=torch.randn(B, T, N_IMU_CH) if use_imu else None,
            audio=[torch.randn(B, T, 1) for _ in range(N_SPEAKERS)],
            spk_of_slot=torch.stack([torch.randperm(N_SPEAKERS)
                                     for _ in range(B)]),
        )
        out = model(**kw)
        n = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rf = model.encoders[model.modalities[0]].receptive_field
        print(f"{mode:<22} {n:>8,}  {rf:>6}  {tuple(out['logits'].shape)}")
