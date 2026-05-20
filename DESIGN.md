# Design Decisions

This document records the choices in Mini V-JEPA that are not obvious from the
code, the trade-offs they involve, and the things that did not make it into
this build. It is intentionally short and unembellished.

## Why latent prediction, not pixel reconstruction

A pixel decoder forces the model to fit photometric detail (felt colour, ball
shading, antialiasing) that is irrelevant to the dynamics. Joint-embedding
training removes that pressure: the loss is computed in the encoder's output
space, so capacity that would otherwise reconstruct pixels is reallocated to
whatever signal makes the future easier to predict from the past — i.e.
positions, velocities, and the geometry of upcoming collisions.

The pixel baseline (`baselines/pixel_predictor.py`) exists so the comparison is
quantitative rather than rhetorical: same encoder, same Δt-conditioned
predictor, comparable parameter budget, evaluated by the same probe.

## Why EMA target (and not LeWorldModel-style end-to-end)

A trainable target encoder collapses unless the loss explicitly prevents it.
V-JEPA's EMA target is the simplest stop-gradient trick that works: the target
network's parameters are a slowly moving average of the online encoder, so the
prediction target is non-stationary but consistent across nearby training
steps.

LeWorldModel removes the EMA and trains end-to-end with a different
regularization scheme. That works in their setting but introduces additional
moving parts (their full loss has more terms and more sensitive
hyperparameters). For a from-scratch reproduction on a laptop we kept the EMA:
one schedule (`τ: 0.996 → 1.0` cosine), one regularizer (variance), one stop-
gradient. Smaller surface area, fewer ways to silently misconfigure.

## Architecture sizing — the param-count discrepancy

PLAN §2.6 cites "5–8 M parameters" as the headline figure. The explicit
channel and dim specification in PLAN §2.1 (encoder CNN widths 64/128/256/128)
and §2.3 (predictor `dim=128, FFN=256, 3 layers, 4 heads`) does not produce a
5–8 M model — it produces a ~0.92 M trainable model (1.39 M including the EMA
target encoder's frozen copy).

The choice was to honour the explicit spec rather than the headline number.
The two are mutually inconsistent and only one of them can survive: the
explicit widths are what the rest of the plan (token count, FFN ratio, latent
dim) is internally consistent with. Scaling the model up to 5–8 M would have
required either widening the latent dim (and the predictor with it) or making
the CNN deeper. Both are valid; neither was specified.

Practical consequence: the model trains faster than the PLAN's "~2 h on M3"
estimate and probably has lower probing R² ceilings than a 5–8 M variant would.
Increasing capacity is the first thing to try if a full training run lands
below the §11 target metrics.

## Anti-collapse contract

Three monitors are computed every epoch on a held-out batch and logged to
`metrics.csv`:

- `avg_std` — per-dimension std of the predicted latents, averaged. Healthy
  above ~0.1; an alert fires after 5 consecutive epochs below 0.05.
- `effective_rank` — `exp(entropy(eigenvalues(cov(z))))` on `z_pred`. Healthy
  above ~40 for `dim=128`. Used as the **checkpoint selection criterion**
  instead of loss, because a low loss can hide a collapsed encoder.
- `avg_cosine_sim` — pairwise cosine similarity averaged over a batch.
  Collapse if it stays above 0.9.
- **`ctx_effective_rank` and `ctx_avg_cosine_sim`** (iter-3) — the same
  metrics computed on the *encoder's output* (`context_tokens`) rather than
  the predictor's output. The E4 ablation in `strategies_log.md` showed that
  `effective_rank` on `z_pred` can stay at 16-22 while the encoder is fully
  collapsed (`ctx_effective_rank=2-4`, `ctx_avg_cosine_sim=0.97`). The
  predictor's internal `query_tokens` + `pos_embed` produce diverse-looking
  output regardless of encoder. Without `ctx_*` monitoring, this failure is
  invisible.

The iter-3 loss is now:

```
loss = MSE(z_pred, z_target.detach())
     + 25 * 0.5 * (var(z_pred) + var(context_tokens))    # reg_target=both
     + 25 *  0.5 * (cov(z_pred) + cov(context_tokens))
```

`cov(z)` is the squared-off-diagonal sum of the feature covariance matrix,
divided by `latent_dim`. The key change vs iter-2 is that VICReg is applied
to both branches — the rationale is in the iter-3 collapse investigation
below.

These pieces — variance + covariance terms on both branches, FIVE monitors
(z_pred and ctx versions of std/rank/cos plus the ctx_eff_rank, ctx_cos
additions in iter-3), rank-based checkpointing, NaN-abort, and the dual
collapse-stop — are load-bearing. Any refactor of the encoder, predictor,
EMA, or loss must keep them all intact (this is the explicit contract in
`CLAUDE.md`).

`effective_rank` uses `torch.linalg.eigvalsh`, which is not implemented on the
MPS backend. The training script sets `PYTORCH_ENABLE_MPS_FALLBACK=1` so the
eigendecomposition silently falls back to CPU. It works but each evaluation
pays a CPU sync; that is acceptable because it runs once per epoch on one
batch.

## Physics sim trade-offs

The pymunk world uses `substeps=24` for collision integrity (the original
value of 8 produced tunnelling between fast balls and a few false collisions).
This makes data generation roughly 3× slower than the PLAN §3.4 estimate of
3–5 minutes for 10 000 sequences. The trade is straightforward: correctness of
the data trumps speed of generation, because anything that probes
position/velocity downstream relies on the labels being right.

Other physics choices match the plan: rolling friction is implemented as an
explicit force opposite to velocity (not as a pymunk damping coefficient),
velocity is snapped to zero below 0.001 to prevent micro-drift, no spin and
no pockets.

## Evaluation honesty notes

Three things in `scripts/evaluate.py` are weaker than a casual reader of the
README might assume:

1. **Train/test split is sequence-level 80/20, not held-out physical
   regime.** All `init_strategies` appear on both sides. The probe measures
   "can a linear map decode positions from these latents on new sequences
   drawn from the same distribution", not "does this generalize to unseen
   physics".
2. **The degradation curve picks one mid-sequence `t_start` per horizon**
   (`t_start = max_t_start // 2`) rather than averaging over every valid
   start. This is faster and gives a stable point estimate but undercounts
   variance across the sequence.
3. **The pixel baseline encoder probed at evaluation time is the online
   encoder.** There is no EMA copy on the baseline — the EMA only exists on
   the V-JEPA side because it is the mechanism the paradigm requires. This
   means the comparison is "V-JEPA's EMA target encoder vs the pixel
   baseline's online encoder", not "EMA vs EMA".

These are documented here and in the README's *Comparison with Pixel
Prediction* section so the comparison can be read for what it actually is.

## The iter-3 collapse investigation

The project's most substantial technical content is the diagnosis of why
iter-1 and iter-2 both collapsed catastrophically at ep18-22 despite passing
multi-agent code review. The full ledger is in `strategies_log.md`; the
design-level takeaway is:

**The reviewer-proposed fix (`cov_loss_weight: 1 → 25`) is empirically
wrong.** An ablation run (E1 in the strategies log) showed it *accelerated*
collapse to ep9. The mechanism: heavier VICReg pressure on the predictor's
output (`z_pred`) lets the predictor satisfy the diversity constraint
internally via its own `query_tokens` + `pos_embed`. With that pressure
relieved, the encoder is free to collapse — and does. The predictor then
*hides* the encoder collapse by producing diverse-looking outputs from
collapsed inputs.

This means a `z_pred eff_rank`-only collapse-stop (the iter-2 implementation)
is structurally insufficient. The iter-3 fixes are co-load-bearing: each
single-axis fix in isolation either fails worse than baseline (E1, E2) or
hides the failure (E4). The first config that survives 150 epochs is the
combination of:

1. `reg_target=both` — apply VICReg on encoder output too, so the predictor
   can't satisfy it internally.
2. `cov_loss_weight=25` — but only useful in combination with #1.
3. `ema_tau_start=0.99` — faster target so the static-target trivial
   minimum is unreachable.
4. `mask_ratio=0.75` — V-JEPA's canonical asymmetry mechanism.
5. `lr_peak=5e-5` (was 1.5e-4) — the four-constraint loss surface is narrow;
   higher LR makes it stochastically unstable. Confirmed by the v1/v2/v3
   validation runs in `strategies_log.md`.
6. `grad_value_clip=0.5` — catches stochastic numerical spikes that
   occasionally produce NaN even with the above. Lifted from the pixel
   baseline's overrides.

The two safety-net upgrades that came out of this — **NaN-abort** in the
V-JEPA training loop (was only on the pixel side) and a **dual collapse-stop**
that watches `ctx_effective_rank` in addition to `effective_rank` — are
load-bearing for any future iteration. The E4 ablation passed every check
the iter-2 stop knew about while the encoder was at rank ~3.

The honest cost is documented in the README's *What we do NOT demonstrate*:
the resulting V-JEPA model has `ctx_effective_rank ≈ 12 / 128`, which caps
linear-probing R² below the pixel baseline. Lifting the rank ceiling is a
future iteration and is laid out in `FUTURE_WORK.md`.

## What didn't make it

- **The V-JEPA paper's "JEPA beats pixel on downstream"** claim. At
  0.92 M params and 10 K sequences with the iter-3 fix, V-JEPA *loses* on
  positions (0.567 vs 0.616) and barely wins on velocities (both ≤ |0.05|).
  `FUTURE_WORK.md` lays out the two architectural changes most likely to
  flip this.
- **The pixel baseline at 150 epochs.** It NaN'd at ep60 even after sigmoid
  → clamp and `lr_peak=2e-5`; the converged `ckpt_best.pt` from ep59
  (`pixel_mse=0.018`) was used instead. Loss had plateaued from ep40 onward
  so this is harmless for the comparison.
- **The predictor as a forecaster.** Trained with 75% masking, the predictor
  works as a training mechanism (its gradient is what drives the encoder to
  be non-trivial) but its outputs are essentially orthogonal to the true
  target at eval time (cos sim ~0.075 vs copy-last's 0.95). The
  evaluation pipeline uses only the encoder for probes.
- Stretch goals from PLAN §11: no animated latent-PCA video, no energy
  conservation curve reconstructed from probed velocities, no
  train-9-test-5 generalization study.
- The headline parameter count of 5–8 M was not reached; see *Architecture
  sizing* above.
