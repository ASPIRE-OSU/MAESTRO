"""
train_hemisphere.py — T2 (Attended Hemisphere): is the attended talker in the
left or the right hemifield?  Chance 0.5.

The two references are the group means of the four co-present talker envelopes,
a_left = (a_1 + a_2)/2 and a_right = (a_3 + a_4)/2, distribution-matched so that
neither can be identified from its amplitude statistics alone.

Runs under either official split protocol:
  --split_setting loso   : listener generalisation (16 folds)
  --split_setting within : content generalisation, pooled across listeners (5 folds)

See spatial_tasks.py for what changed relative to the previous revision and why
every reported number now comes with a permutation null and a contribution.
"""

from spatial_tasks import main

if __name__ == "__main__":
    main("hemisphere", __doc__)
