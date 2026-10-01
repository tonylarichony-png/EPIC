# CNN-EXP-049B: paired end-to-end cascade pilot

## Question

Does online regional conditioning improve the final genome-wide nucleotide
ranking, when Network A and Network B are allowed to co-adapt?

The separate R16 metrics of Network A are diagnostics.  They are not a gate for
this experiment and cannot establish the value of the complete cascade.

## Paired arms

Both arms train for 20,000 steps from the same seeded Network-B initialisation
and consume the same deterministic profile-sampling trajectory.

| Arm | Regional input | Trainable parameters |
|---|---|---|
| control | 17 zero channels | Network B |
| joint | online A-v1 log-rate + 16 latent channels | Network A and Network B |

The joint arm starts Network A from the evaluated A-v1 fold-0 20K checkpoint.
It then trains on all nine train contigs.  The three validation contigs remain
unseen by both networks.

Network A sees an R16-aligned 8192-bp window centred on each sampled B target.
Its condition is gathered by absolute genomic R16 coordinate and broadcast to
the 2052-bp B input.  The minus strand is returned to genomic orientation before
the lookup and reversed only when it is paired with reverse-complement B input.

## Optimisation

- Network B learning rate: `1e-3`;
- Network A learning rate in the joint arm: `1e-4`;
- batch: two genomic tiles, evaluated on both strands;
- FiLM after blocks `2,4,6,8`;
- nucleotide objective: the same presence, within-R16, count-head, residual,
  and optional EXP047 risk weights in both arms;
- joint-only auxiliary: `0.1 × (Poisson NLL + 0.1 × regional ranking)`.

The zero-conditioned control deliberately keeps the same FiLM and residual-head
parameterisation as the joint arm.  This isolates information from A rather
than extra modules in B.

## Evaluation and decision

Each completed arm receives one full natural validation scan.  The primary
metric is overall nucleotide AP at five-decimal score precision.  Per-contig AP
must agree with the overall direction.  EPIC Spearman of the same final score
is secondary and cannot rescue an AP loss.

- advance to a paired 50K continuation when joint minus control AP is at least
  `+0.005` overall and is non-negative on every validation contig;
- stop this cascade when overall delta is non-positive or at least two contigs
  are negative;
- otherwise record the 20K result as inconclusive rather than tuning on the
  observed validation values.

The validation split has been inspected in earlier experiments, so this is an
exploratory architecture decision, not an unbiased estimate of leaderboard
performance.

## Observed 20K result

| Arm | Overall nucleotide AP | EPIC Spearman |
|---|---:|---:|
| zero-condition control | 0.046371994 | 0.120001431 |
| joint A→FiLM→B | **0.092451843** | **0.128554933** |

The joint AP gain was `+0.046079849` and every validation contig improved:
`+0.049521233`, `+0.039955227`, and `+0.046067501`.  The registered continuation
rule passed decisively.

## Registered 20K→50K continuation

Both 20K arms continue for another 30K sampled batches.  Model weights, Adam
moments, AMP scaler, and sampler state are restored.  The exhausted OneCycle
schedule is replaced by a new cosine phase:

```text
B: 3e-4 → 3e-5
A: 3e-5 → 3e-6 (joint arm only)
```

The 20K checkpoints remain immutable; continuation is written to
`continuation_50k.pt`.  Final evaluation is performed only after both arms reach
50K.  We will retain architectural support only if joint remains at least
`+0.005 AP` above its matched control and is non-negative on every contig.  The
50K joint AP is also compared with its own 20K AP and with EXP047-B, but those
comparisons are not matched causal estimates.

Because the decision to continue used the 20K validation result, the 50K scan
is an exploratory learning-curve measurement on an already inspected split.

## Observed 50K result

| Arm | Overall nucleotide AP | EPIC Spearman | ΔAP vs 20K |
|---|---:|---:|---:|
| zero-condition control | 0.073778159 | 0.133377696 | +0.027406165 |
| joint A→FiLM→B | **0.104208798** | 0.130878357 | +0.011756955 |

At 50K, joint remains above its matched control by `+0.030430639 AP`, with a
positive delta on every validation contig.  The architectural-support rule is
therefore satisfied.  However, the joint result is `−0.000003128 AP` relative
to EXP047-B (`0.104211926`): it reproduces the existing AP level rather than
creating a new champion.  The joint-control gap also shrank from `0.046080` at
20K to `0.030431` at 50K because the control improved faster.

Full tables, per-contig comparison, learning curves, and oracle caveats:
[[docs/CNN_EXP045_049_RESULTS_REVIEW_2026-10-01]].

## Artifacts

The pilot uses a new run root and does not modify A-v1 or the legacy EXP049 run:

```text
/tmp/exp049b_joint_pilot
/kaggle/working/exp049b_joint_pilot.zip
```
