# CNN-EXP-049A-v1: count-aware regional pilot

## Decision being tested

The existing A-v0 result is retained as a binary baseline.  It is not used to
train B because its 16-channel latent projection was disconnected from the
supervised loss.

A-v1 makes the latent vector a mandatory bottleneck and predicts one regional
Poisson log-rate from the strand-specific sum of nucleotide counts in each R16.

```text
multiscale fused 64
        ↓
trained latent 16
        ↓
regional log-rate 1
```

The registered loss is:

```text
masked Poisson NLL + 0.1 × active-vs-empty pairwise ranking
```

Windows are sampled from the natural allowed-region distribution
(`positive_window_fraction=0`) rather than with the A-v0 positive-window
oversampling.

## Scope

Only fold 0 is trained.  Held-out contigs are fixed by the original fold seed:

- `NC_064045.1`
- `NC_064047.1`
- `NC_064048.1`

Registered checkpoints are 5K, 10K and 20K.  Every checkpoint is evaluated on
exactly the same held-out regions.  Validation contigs are never read for model
selection.

## Metrics

- natural regional AP;
- AP lift over active-region prevalence;
- dense-rank Spearman between regional score and summed count, active R16 only;
- empty-region suppression at 99.5% active recall.

The A-v0 fold-0 cache is reevaluated with the same implementation and sigmoid
link.  A-v1 uses the Poisson nonzero probability `1-exp(-exp(log_rate))`; this is
monotonic in log-rate, so it does not change unquantized ranking.

## Pre-registered decision rule

Let `AP0` and `S0` be A-v0 fold-0 AP and positive-count Spearman.

1. Eligible A-v1 checkpoints must retain at least `99%` of `AP0`.
2. Among eligible checkpoints, select the one with maximum positive-count
   Spearman; break an exact tie by higher AP, then fewer steps.
3. Proceed to full three-fold A-v1 only if selected Spearman is at least
   `S0 + 0.02` and empty-region suppression is no more than 5 percentage points
   below A-v0.
4. Otherwise stop this objective or redesign it; do not inspect natural
   validation to rescue the decision.

The rule keeps genome-wide discrimination primary while requiring a material
gain in the count ordering needed for EPIC Spearman.

## Cloud artifacts

The pilot uses a separate root and never overwrites `exp049_run.zip`:

```text
/tmp/exp049a_rate_pilot
/kaggle/working/exp049a_rate_pilot.zip
```

Run the `EXP049 A-v1 COUNT-AWARE PILOT` section of
`CNN_EXP049_CLOUD_TRAIN.ipynb`.  Export the pilot after every interrupted or
completed session and Quick Save with outputs enabled.
