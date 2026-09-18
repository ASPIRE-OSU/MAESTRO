"""
train_eccentricity.py — T3 (Attended Eccentricity): is the attended talker at an
inner ({S2,S3}, +/-22.5 deg) or an outer ({S1,S4}, +/-67.5 deg) position?
Chance 0.5.

The two references are the group means of the four co-present talker envelopes,
a_inner = (a_2 + a_3)/2 and a_outer = (a_1 + a_4)/2, distribution-matched so that
neither can be identified from its amplitude statistics alone.

Runs under either official split protocol:
  --split_setting loso   : listener generalisation (16 folds)
  --split_setting within : content generalisation, pooled across listeners (5 folds)

See spatial_tasks.py for what changed relative to the previous revision and why
every reported number now comes with a permutation null and a contribution.
"""

from spatial_tasks import main

if __name__ == "__main__":
    main("eccentricity", __doc__)
