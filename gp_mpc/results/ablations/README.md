# Ablations

Closed-loop checks behind the design choices (inducing points, no speed bounds, rail). They were run on
the medium scenario's round-1 data while the method was being
developed, with 6 N exploration noise and 25% wide starts (the final
pipeline uses 10 N and 50%).

| File | What was varied | Result |
|---|---|---|
| `inducing_selection_and_prior_mean.json` | Inducing points: farthest-point / mixed / random. Prior mean: zero / data mean. 3 seeds, 3 episodes each. Subset-of-data GP | Farthest-point: 0 falls in 18. Mixed: 4 in 18. Random: 8 in 18. The prior mean made little difference |
| `envelope_and_rail.json` | GP: sparse (DTC) / subset of data. Soft speed bounds at the data maximum / 99.5th percentile / none. Rail constraint on or off. 3 seeds, 4 episodes each | No bounds + rail: 0 falls in 12 for both GPs. Bounds at the 99.5th percentile: 9 and 10 in 12 |
| `strong_with_envelope_log.txt` | The strong scenario run with the speed bounds still on | Kept for reference |

`robust_check.py` in its current form reproduces the second table. The
first came from an earlier version of the same script that swept the
prior mean and the selection rule.
