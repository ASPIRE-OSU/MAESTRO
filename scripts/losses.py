"""
losses.py
---------
Training objective for the attended-talker decoder.

The previous revision trained on label-smoothed cross-entropy over the four-way
decision and nothing else.  That loss is a function of the scores only; it has
no opinion about HOW the scores were produced, and the model could produce them
without the physiological recording (see the `model_classification` docstring).
The degenerate solution therefore cost nothing and was the easier optimum, so
gradient descent found it.  Checkpoint selection on validation accuracy then
locked it in, because accuracy is exactly what the shortcut maximises.

The fix is to make the control we use to DETECT a recording-independent decoder
part of what the model is optimised for.  Four terms are added:

  contrastive     which recording goes with which stimulus.  A collapsed encoder
                  makes the similarity matrix rank-one, which pins this term at
                  its chance value -- collapse is its WORST case, not a minimum.

  null hinges     the real recording must outscore a permuted one and a zero one
                  by a margin.  This is literally the evaluation control, as a
                  loss; the quantity it maximises is the reported contribution.

  anti-collapse   penalises the per-dimension temporal variance falling toward
                  zero, which is the exact quantity that vanishes on collapse,
                  plus an off-diagonal covariance term so the variance
                  requirement cannot be met with D copies of one signal.

  adversarial     an audio-only head, with its gradient reversed into the audio
                  encoder, so residual acoustic cues are actively unlearned.
"""

import torch
import torch.nn.functional as F


DEFAULTS = dict(
    smoothing=0.1,      # label smoothing on the task term
    w_aux=0.3,          # per-modality orientation auxiliary
    w_clip=1.0,         # contrastive
    w_null=0.5,         # permutation + zero hinges
    w_vic=0.1,          # anti-collapse
    w_adv=0.3,          # audio-only adversary
    margin=0.5,         # hinge margin
)


def pairwise_scores(head, brain_enc, audio_enc):
    """All-pairs coupling score. (B,T,D), (B,T,D) -> (B,B)."""
    b = brain_enc - brain_enc.mean(1, keepdim=True)
    a = audio_enc - audio_enc.mean(1, keepdim=True)
    b = b / (b.norm(dim=1, keepdim=True) + 1e-6)
    a = a / (a.norm(dim=1, keepdim=True) + 1e-6)
    C = torch.einsum("itd,jtd->ijd", b, a)                       # (B,B,D)
    return head.score_from_corr(C)


def contrastive(head, brain_enc, audio_pos, subject):
    """InfoNCE along the recording axis, restricted to same-listener pairs.

    S[i, j] scores window i's recording against window j's correct candidate;
    the target for row i is column i.

    Why collapse cannot minimise this.  If every window yields the same
    embedding b, then S[i, j] = s(b, a_j) =: c_j does not depend on i, so every
    row of S is the same vector c.  The first term becomes
    logsumexp(c) - mean(c), and by Jensen  mean_j exp(c_j) >= exp(mean(c)),
    hence logsumexp(c) >= log B + mean(c) and the term is >= log B -- the
    chance-level loss.  For the transposed term each row is constant, its
    softmax is uniform, and it equals log B exactly.

    Why same-listener only.  Raw preprocessed EEG identifies the listener with
    0.90 accuracy (16-way, chance 0.0625).  Across listeners the score matrix is
    separable by identity alone, giving a second degenerate solution in which
    the encoder represents WHO rather than WHAT.
    """
    B = brain_enc.shape[0]
    S = pairwise_scores(head, brain_enc, audio_pos)
    same = subject[:, None] == subject[None, :]
    S = S.masked_fill(~same, float("-inf"))
    tgt = torch.arange(B, device=brain_enc.device)
    usable = same.sum(1) >= 2
    if usable.sum() == 0:
        return brain_enc.sum() * 0.0
    return 0.5 * (F.cross_entropy(S[usable], tgt[usable])
                  + F.cross_entropy(S.t()[usable], tgt[usable]))


def null_hinges(head, brain_enc, audio_pos, margin=0.5):
    """The real recording must beat a permuted one and a zero one.

    At the collapsed solution the permuted score equals the real score (the
    score does not depend on whose recording it received) and both equal the
    zero-recording score, so each hinge sits at softplus(margin) with a non-zero
    derivative: the first gradient step already pushes the real score up.
    """
    B = brain_enc.shape[0]
    roll = torch.roll(torch.arange(B, device=brain_enc.device), 1)
    s_real = head.score_from_corr(head.corr(brain_enc, audio_pos))
    s_perm = head.score_from_corr(head.corr(brain_enc[roll], audio_pos))
    s_zero = head.score_from_corr(head.corr(torch.zeros_like(brain_enc),
                                            audio_pos))
    return (F.softplus(s_perm - s_real + margin).mean()
            + F.softplus(s_zero - s_real + margin).mean())


def anti_collapse(brain_enc, gamma=0.5, l_var=1.0, l_cov=0.04):
    """Hinge on the per-dimension temporal standard deviation, plus a
    redundancy penalty between dimensions."""
    zc = brain_enc - brain_enc.mean(1, keepdim=True)              # (B,T,D)
    var = F.relu(gamma - zc.std(1)).mean()
    z = zc.reshape(-1, brain_enc.shape[-1])
    z = z - z.mean(0)
    cov = (z.t() @ z) / max(1, z.shape[0] - 1)
    D = z.shape[1]
    off = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / D
    return l_var * var + l_cov * off


def total_loss(out, label, subject, head, cfg=None, spk_label=None):
    """Assemble the objective.  Returns (loss, per-term dict)."""
    cfg = {**DEFAULTS, **(cfg or {})}
    parts = {}

    loss = F.cross_entropy(out["logits"], label,
                           label_smoothing=cfg["smoothing"])
    parts["task"] = float(loss.detach())

    # per-modality auxiliary: without it the strongest branch absorbs the
    # gradient once it classifies correctly and the others never train
    spk_logits = out.get("spk_logits") or {}
    branches = [v for k, v in spk_logits.items() if k != "fused"]
    if branches and spk_label is not None and cfg["w_aux"] > 0:
        aux = sum(F.cross_entropy(v, spk_label) for v in branches) / len(branches)
        loss = loss + cfg["w_aux"] * aux
        parts["aux"] = float(aux.detach())

    brain_enc, audio_encs = out.get("brain_enc"), out.get("audio_encs")
    if brain_enc is not None and audio_encs is not None:
        audio_pos = torch.stack(audio_encs, 1)[torch.arange(len(label)), label]
        if cfg["w_clip"] > 0:
            v = contrastive(head, brain_enc, audio_pos, subject)
            loss = loss + cfg["w_clip"] * v
            parts["clip"] = float(v.detach())
        if cfg["w_null"] > 0:
            v = null_hinges(head, brain_enc, audio_pos, cfg["margin"])
            loss = loss + cfg["w_null"] * v
            parts["null"] = float(v.detach())
        if cfg["w_vic"] > 0:
            v = anti_collapse(brain_enc)
            loss = loss + cfg["w_vic"] * v
            parts["vic"] = float(v.detach())
        if "adv_logits" in out and cfg["w_adv"] > 0:
            v = F.cross_entropy(out["adv_logits"], label)
            loss = loss + cfg["w_adv"] * v
            parts["adv"] = float(v.detach())

    return loss, parts
