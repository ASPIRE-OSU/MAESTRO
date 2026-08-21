# Legacy results — pre-fix model

Everything in this directory was produced by the PREVIOUS revision of the model
(`scripts/model_classification_legacy.py`, `scripts/train_aad_legacy.py`).

**These checkpoints will not load against the current `model_classification.py`**:
the encoder has different layer counts and adds normalisation layers, so the
state-dict keys and shapes differ.

**The accuracies in these JSON files are not attributable to the physiological
recording.** Permuting the recordings across test windows changes them by at
most 0.0035 (mean 0.0004) across all 40 published configurations, and
substituting zeros for the recording changes them by nothing. They are retained
only so the before/after ablation can be reproduced.

See the module docstring of `scripts/model_classification.py` for the five
defects and their remedies.
