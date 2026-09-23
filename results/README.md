# Benchmark results

The complete evaluation outputs behind every number, table, and figure in the
paper, produced by the code on this branch (after the 20 s window-count fix and
the T2/T3/SNR leakage fix). Everything here is either a raw evaluation output
(JSON) or a file regenerated deterministically from those JSONs by the scripts
below. The superseded pre-fix results live in `../results_legacy/` and should
not be used.

## Layout

| Directory | Contents | Produced by |
|---|---|---|
| `res_{split}_w{W}_h{H}_qmatch/` | T1 (four-class attended source): one `results_{config}_{split}.json` per modality configuration, with per-fold accuracy, permuted-accuracy null, and contribution | `scripts/train_aad.py` |
| `spat_{split}_w{W}_h{H}_qmatch/` | T2 (hemisphere) and T3 (eccentricity), same format | `scripts/train_hemisphere.py` / `scripts/train_eccentricity.py` (via `scripts/spatial_tasks.py`) |
| `significance/` | 10,000-permutation per-fold nulls and p-values, one `sig_{task}_{config}_{split}_w{W}.json` per cell | `scripts/recompute_significance.py` |
| `snr/` | SNR-stratified T1 evaluation with per-bin audio-only floors | `scripts/analyze_snr.py` |
| `smoke_within_w10.0_h5.0_{qmatch,raw}/` | The matched-vs-raw candidate-construction training pair (EEG+Gaze, 10 s, within-subject, both binary tasks) | training scripts with the construction flag toggled |
| `ablate_nohinge_{split}_w{W}_h{H}_qmatch/` | The 20-cell retrain with both hinge terms removed from the objective | training scripts, no-hinge variant |
| `paper_tables/` | The released per-fold CSVs (`folds_t1.csv`, `folds_t2t3.csv`, `snr_bins.csv`), `stats.json`, and the LaTeX tables pasted into the paper | `scripts/make_paper_tables.py` |
| `paper_figures/` | The two Section V figures (`fig_folds.png`, `fig_snr.png` — the latter is `snr_fixed.png` in the paper) | `scripts/make_paper_figures.py` |

Fold indexing: under LOSO the fold index is the held-out participant (0-based,
participants 1–16); within-subject it is the 5 content-disjoint CV folds.

## Regenerating the tables and figures

From the repository root (needs numpy, scipy, matplotlib):

```bash
python scripts/make_paper_tables.py  results --out results/paper_tables
python scripts/make_paper_figures.py results --out results/paper_figures
```

Both are deterministic: regenerated CSVs and `stats.json` are byte-identical to
the committed ones, and the PNGs reproduce bit-for-bit under the same
matplotlib version.

## Checkpoints

Please visit : [Checkpoints](https://github.com/ASPIRE-OSU/MAESTRO/releases/tag/weights-v1) 