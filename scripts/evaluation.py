"""
evaluation.py
-------------
The permutation battery.  Every reported accuracy must be accompanied by these
numbers; an accuracy on its own is not interpretable for this task.

THE PERMUTATION NULL.  Permute the physiological recordings across test windows
while each window keeps its OWN candidates and its OWN label.  A decoder that
uses the recording loses accuracy; one that reads the candidates alone does not.
The reported quantity is

    contribution = accuracy - mean accuracy under permutation

This must be a FEATURE permutation, not a label permutation.  Permuting labels
would also break the candidate-to-label relationship, so an audio-only decoder
would fall to chance too and the test would report "significant" for a model
with no physiological content at all.  Formally we test
`score independent of recording, given candidates and label`, not
`candidates and recording jointly independent of label`.

SUPPORTING CONTROLS.
  zeros ablation      zeros through the encoder in place of the recording.
  decision-flip rate  fraction of windows whose choice changes under permutation;
                      approximately zero for an inert decoder.
  collapse            mean pairwise cosine of the TIME-CENTRED, unit-normalised
                      embeddings -- the quantity the coupling score consumes.
                      1.0 means every window produces the same temporal pattern.
  stratified nulls    permuting only within a stratum makes that stratum's
                      information useless.  "position" rules out a slow drift
                      shared by recording and envelope; "trial" is stricter --
                      all windows of a trial share a listener, an attended
                      talker and a stimulus set, so a decoder that merely
                      recognised WHICH TRIAL it was viewing would survive a
                      global permutation but not a within-trial one.
"""

import numpy as np
import torch
import torch.nn.functional as F


class Evaluator:
    """Encodes the test set once, then re-scores under arbitrary permutations
    without re-encoding."""

    def __init__(self, model, loader, device, chunk: int = 512, strata=None):
        self.model, self.device, self.chunk = model, device, chunk
        self.strata = {} if strata is None else {
            k: np.asarray(v) for k, v in strata.items()}
        model.eval()

        embs = {m: [] for m in model.modalities}
        auds, labels, perms, spks, zero_e = [], [], [], [], {}
        with torch.no_grad():
            for eeg, video, gaze, imu, audio, lab, spk_of_slot, att, subj in loader:
                inputs = {"eeg": eeg, "video": video, "gaze": gaze, "imu": imu}
                inputs = {k: (v.to(device) if v is not None else None)
                          for k, v in inputs.items()}
                e = model.encode(inputs)
                for m, v in e.items():
                    embs[m].append(v.cpu())
                if model.couple_mod is not None:
                    a = torch.stack([model.audio_encoder(x.to(device))
                                     for x in audio], 1)          # (B,K,T,D)
                    auds.append(a.cpu())
                labels.append(lab); perms.append(spk_of_slot); spks.append(att)
                if not zero_e:
                    z = model.encode({k: (torch.zeros_like(v)
                                          if v is not None else None)
                                      for k, v in inputs.items()})
                    zero_e = {m: v[:1].cpu() for m, v in z.items()}

        self.embs = {m: torch.cat(v) for m, v in embs.items() if v}
        self.aud = torch.cat(auds) if auds else None
        self.labels = torch.cat(labels)
        self.perms = torch.cat(perms)
        self.spks = torch.cat(spks)
        self.zero_e = zero_e
        self.N = len(self.labels)
        self.K = (self.aud.shape[1] if self.aud is not None
                  else self.perms.shape[1])

        # Keep the cached encoder outputs resident on the accelerator.  Every
        # permutation re-reads the whole test set, so leaving these on the host
        # makes the permutation loop transfer-bound rather than compute-bound:
        # at 3520 test windows a single permutation moves ~360 MB, and 10000 of
        # them move 3.6 TB per fold.  The tensors themselves are small enough to
        # stay resident, and `.to()` is a no-op once they are, so `logits()`
        # needs no change.  Falls back to the host if the device is short of
        # memory, which only costs speed.
        try:
            self.embs = {m: v.to(device) for m, v in self.embs.items()}
            if self.aud is not None:
                self.aud = self.aud.to(device)
            self.perms = self.perms.to(device)
            self.labels = self.labels.to(device)
            self.zero_e = {m: v.to(device) for m, v in (zero_e or {}).items()}
            self.resident = True
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self.resident = False

    @torch.no_grad()
    def logits(self, perm=None, zero: bool = False):
        m0 = self.model
        idx = torch.arange(self.N) if perm is None else torch.as_tensor(perm)
        out = []
        for s in range(0, self.N, self.chunk):
            sl = slice(s, min(s + self.chunk, self.N))
            n = sl.stop - sl.start
            if zero:
                e = {m: self.zero_e[m].expand(n, -1, -1).to(self.device)
                     for m in self.embs}
            else:
                e = {m: self.embs[m][idx[sl]].to(self.device) for m in self.embs}
            lg = None
            if m0.couple_mod is not None and self.aud is not None:
                a = [self.aud[sl, k].to(self.device)
                     for k in range(self.aud.shape[1])]
                lg = m0.head(e[m0.couple_mod], a)
            if m0.spatial_heads:
                sp = {m: m0.spatial_heads[m](e[m])
                      for m in m0.modalities if m in e}
                if len(m0.modalities) > 1:
                    sp["fused"] = m0.spatial_fuse(
                        torch.cat([e[m] for m in m0.modalities], -1))
                key = "fused" if "fused" in sp else next(iter(sp))
                slot = torch.gather(sp[key], 1, self.perms[sl].to(self.device))
                lg = slot if lg is None else lg + slot
            out.append(lg)
        return torch.cat(out)

    def accuracy(self, lg):
        return float((lg.argmax(1) == self.labels).float().mean())

    def _permutation(self, rng, stratum=None):
        if stratum is None or stratum not in self.strata:
            return rng.permutation(self.N)
        st = self.strata[stratum]
        p = np.arange(self.N)
        for g in np.unique(st):
            m = np.where(st == g)[0]
            p[m] = m[rng.permutation(len(m))]
        return p

    def by_group(self, groups, n_shuffle: int = 20, seed: int = 1000) -> dict:
        """Accuracy, permutation null and contribution WITHIN each group of test
        windows -- used for the SNR-stratified analysis.

        `groups` is an (N,) array of labels over the test windows in loader
        order.  The null is computed per window: over `n_shuffle` permutations,
        the fraction of times the window is still decoded correctly when it is
        given ANOTHER window's recording while keeping its own candidates and
        its own label.  Averaging that within a group gives the group's
        audio-only floor, so a group's contribution is not confounded by the
        group's own class balance or candidate difficulty.

        This matters for SNR in particular: the acoustic mark that identifies
        the attended talker is itself a function of the mixture, so the floor
        need not be flat across SNR bins.  Reporting accuracy per bin without
        the per-bin null cannot distinguish "decoding improves with SNR" from
        "the shortcut gets easier with SNR".
        """
        groups = np.asarray(groups)
        assert len(groups) == self.N, (
            f"groups has {len(groups)} entries for {self.N} test windows")

        real_ok = (self.logits().argmax(1) == self.labels).cpu().numpy().astype(float)
        null_ok = np.zeros(self.N, dtype=float)
        for k in range(n_shuffle):
            rng = np.random.default_rng(seed + k)
            lg = self.logits(perm=self._permutation(rng))
            null_ok += (lg.argmax(1) == self.labels).cpu().numpy()
        null_ok /= max(n_shuffle, 1)

        out = {}
        for g in np.unique(groups):
            m = groups == g
            acc, nul = float(real_ok[m].mean()), float(null_ok[m].mean())
            out[int(g)] = dict(n=int(m.sum()), accuracy=acc, null_mean=nul,
                               contribution=acc - nul)
        return out

    def battery(self, n_shuffle: int = 20, seed: int = 1000) -> dict:
        real_lg = self.logits()
        real = self.accuracy(real_lg)

        nulls, flips = [], []
        strat = {k: [] for k in self.strata}
        for k in range(n_shuffle):
            rng = np.random.default_rng(seed + k)
            lg = self.logits(perm=self._permutation(rng))
            nulls.append(self.accuracy(lg))
            flips.append(float((lg.argmax(1) != real_lg.argmax(1))
                               .float().mean()))
            for nm in self.strata:
                srng = np.random.default_rng(5000 + k)
                strat[nm].append(self.accuracy(
                    self.logits(perm=self._permutation(srng, nm))))
        nulls = np.array(nulls)

        # collapse, on the coupling modality (or the first present)
        m = self.model.couple_mod or self.model.modalities[0]
        E = self.embs[m]
        sel = torch.randperm(len(E))[:512]
        Ec = E[sel] - E[sel].mean(1, keepdim=True)
        Ec = Ec / (Ec.norm(dim=1, keepdim=True) + 1e-6)
        V = F.normalize(Ec.reshape(len(sel), -1), dim=-1)
        n = len(V)
        collapse = float(((V @ V.t()).sum() - n) / (n * (n - 1)))

        res = dict(
            accuracy=real,
            null_mean=float(nulls.mean()), null_std=float(nulls.std()),
            contribution=float(real - nulls.mean()),
            p_permutation=float((np.sum(nulls >= real) + 1) / (n_shuffle + 1)),
            zeros_accuracy=self.accuracy(self.logits(zero=True)),
            flip_rate=float(np.mean(flips)),
            collapse=collapse,
            chance=1.0 / self.K,
            n_windows=self.N,
        )
        for nm, v in strat.items():
            res[f"null_{nm}"] = float(np.mean(v))
            res[f"contribution_{nm}"] = float(real - np.mean(v))
        return res
