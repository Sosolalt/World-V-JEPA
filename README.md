# Mini V-JEPA: A 2D Billiard World Model in Latent Space

> A from-scratch V-JEPA that learns 2D billiard dynamics in latent space.
> At 0.92 M trainable parameters on a single MacBook M3, the encoder beats a
> parameter-matched pixel-prediction baseline on every linear probe in our suite
> by +13pp (positions), +20pp (corrected velocity), +19pp (OCP), and reaches 84%
> pairwise accuracy on a Meta-style intuitive-physics counterfactual screen.

![Architecture](assets/architecture_v2_a_wide.png)

Mini V-JEPA is a 0.92 M-parameter Joint Embedding Predictive Architecture
trained on a single MacBook Pro M3 to predict the dynamics of nine balls on a 2D
billiard table. The model never reconstructs a single pixel. It encodes four
context frames into a latent representation and predicts the latent representation
of a future frame, supervised against an EMA copy of its own encoder. Despite the
small budget — three orders of magnitude below the original V-JEPA — the
resulting encoder produces measurably better physical readouts than a
parameter-matched pixel-prediction baseline on every downstream linear probe
(positions, velocities, held-out initial-condition regime, object-contact
prediction) and assigns visibly higher surprise to physics-violating sequences in
a Meta-style Violation-of-Expectation (VoE) screen adapted from *Intuitive
Physics Emerges from V-JEPA* ([arXiv 2502.11831](https://arxiv.org/abs/2502.11831)).

---

## Key Results

V-JEPA V2 — trained on 10K billiard sequences (M3, 0.92 M trainable params) with
canonical pixel-patch tubelet masking, L1 loss, and the iter-3 anti-collapse
stack. All numbers come from [scripts/evaluate.py](scripts/evaluate.py) with the
V2 evaluation harness (online encoder, attentive probe, mean-pool + RidgeCV,
velocity-via-difference, N=2000 sequence-level paired bootstrap with 1000
resamples, eval mask ratio 0.0).

| Probe | V-JEPA V2 | Pixel (from-scratch) | Pixel (frozen V1 enc.) |
|---|---|---|---|
| Per-frame position R² | **0.74** [0.71, 0.77] ± 0.012 | 0.61 [0.58, 0.64] ± 0.014 | 0.59 [0.55, 0.63] ± 0.015 |
| Velocity-difference R² † | **0.36** [0.32, 0.40] ± 0.018 | 0.04 [0.01, 0.07] ± 0.011 | 0.03 [0.00, 0.06] ± 0.012 |
| OCP AUC (K=10) | **0.72** [0.69, 0.75] ± 0.013 | 0.53 [0.50, 0.56] ± 0.011 | 0.54 [0.51, 0.57] ± 0.012 |
| Held-out-regime pos R² | **0.61** ± 0.016 | 0.48 ± 0.017 | 0.46 ± 0.018 |
| VoE pairwise accuracy | **84%** [80, 87] | 56% [52, 60] | — |

CIs are 95% sequence-level paired bootstrap (1000 resamples, N=2000). The `± x`
column is across-seed standard deviation of the probe (5 seeds, encoder fixed) —
a separate noise source not folded into the eval-bootstrap CI. Per-sequence
paired-bootstrap deltas (V-JEPA − pixel from-scratch): positions **+0.13 [+0.09,
+0.17]**, velocity-difference **+0.32 [+0.28, +0.36]**, OCP AUC **+0.19 [+0.15,
+0.23]**, all disjoint from zero.

**† Velocity headline is an upper bound.** The +32pp gap is partly an artifact
of the pixel baseline's deliberately temporally-smooth representation (its `Δz`
is near-zero by construction). After variance-equalizing the pixel-baseline `Δz`
per dim before RidgeCV — a crude bias correction — the corrected gap is
**+0.20 [+0.17, +0.23]** (paired-bootstrap CI). A pixel baseline with an
explicit next-frame temporal-contrast head would close more. We did not train
one; the corrected estimate is the honest headline.

The frozen-V1-encoder column is the readout diagnostic explained in
[Baseline framing](#baseline-framing--why-two-pixel-predictors). Limitations are
listed in [What we do NOT demonstrate](#what-we-do-not-demonstrate).

---

## Why JEPA, Not Generative?

A single 64×64 billiard frame carries position but not motion. Predicting the
future requires integrating several frames into a velocity estimate, propagating
that estimate through elastic collisions and light rolling friction, and reasoning
about whether two balls will or will not occupy overlapping space some Δt later. A
generative video model can be trained on this task, but its loss is evaluated in
pixel space: gradient capacity is consumed by felt texture, ball shading,
antialiasing artifacts and cushion-border interpolation — none of which carry
predictive signal about the underlying dynamics. The decoder rewards photometric
fidelity, not physical fidelity, and the encoder bends to whatever the decoder
needs.

V-JEPA (Bardes et al., 2024; V-JEPA 2, Meta 2025) reframes the objective. The
supervision target is the latent representation produced by an EMA copy of the
encoder itself — `z_target = sg(f_θ̄(x_{t+Δt}))`. Pixel-level detail is
*discarded by the encoder before the loss sees it*, so the only path to lower loss
is for the encoder + predictor to encode quantities the future actually depends
on: positions, velocities, geometry of contact. The bet is that this inductive
bias allocates the model's capacity to physical state rather than to rendering,
and that the downstream encoder will linearly decode those quantities better than
a same-capacity pixel-trained encoder. It is precisely the argument LeCun makes in
*A Path Towards Autonomous Machine Intelligence* (2022): world models should live
in abstract representational space, not in observation space.

The well-documented failure mode is representation collapse — the encoder maps
every frame to a constant vector and the predictor trivially matches it. Two
iterations of this project collapsed at epoch 18-22
([strategies_log.md](strategies_log.md)). The V1 fix combined four mechanisms —
VICReg on both branches (`reg_target=both`), scale-matched covariance weight
(`cov_weight=25`), faster EMA (`τ_start=0.99`) and 75% mask replacement on context
*tokens* — and survived 150-epoch runs without collapse. The V2 architecture in
this README replaces V1's token-output masking with canonical V-JEPA
**pre-encoder** pixel-patch tubelet masking, switches MSE to L1, and confines the
loss to masked positions. C-JEPA
([arXiv 2410.19560](https://arxiv.org/abs/2410.19560)) supplies the
VICReg-on-target rationale; the *Intuitive Physics Emerges from V-JEPA* protocol
supplies the counterfactual evaluation.

The hypothesis — at fixed parameter count, a latent-prediction objective produces
an encoder whose features linearly decode physical state better than a
pixel-prediction objective, *and* whose latent dynamics support counterfactual
physical reasoning that pixel decoders cannot fake — is the entire content of
this project. The result holds empirically on every probe except one
horizon-1 rollout edge case explained in [Results](#predictor-and-autoregressive-rollout).

---

## Architecture

![V2 architecture](assets/architecture_v2_a_wide.png)

Mini V-JEPA V2 follows the canonical V-JEPA recipe end-to-end: mask the input in
pixel space, encode the unmasked context, ask a small predictor to fill in the
masked positions in latent space, and supervise against an EMA target encoder
applied to the un-masked input.

**Tubelet masking (pre-encoder).** A context clip is a tensor of shape
`(B, T=4, 3, 64, 64)`. We partition it into `2×8×8` spatio-temporal tubelets,
yielding `(T/2)·(H/8)·(W/8) = 2·8·8 = 128` tubelets per clip. Multi-block
sampling draws four short blocks at 15% spatial coverage and one long block at
50%, with per-block aspect ratio `U(0.75, 1.5)`; the union covers ~70-90% of
tubelets after overlap. The union defines the *masked* set; the complement is the
*visible* set passed to the encoder. This replaces V1's post-encoder
token-replacement scheme, which was the wrong asymmetry direction and is removed
in [mini_vjepa/vjepa.py](mini_vjepa/vjepa.py).

**Context encoder.** Topologically unchanged from iter-3: three stride-2 conv
blocks (`Conv2d → GroupNorm → GELU`, channels 3 → 64 → 128 → 256), a
`Conv2d(256, 128, 1)` channel-squeeze projection, `LayerNorm`, and one
`SpatialSelfAttention` layer (4 heads). Each context frame is processed by the
conv tower independently to produce an 8×8 spatial grid of 128-dim tokens (64
tokens per frame × 4 frames = 256 tokens per clip). **Where the tubelet mask
applies.** Convolutions are local, so masking-before-conv yields no asymmetry
pressure on the conv stages themselves. We instead apply the mask at the
*post-tokenization, pre-attention* stage: each `2×8×8` tubelet maps to 2
adjacent frames × 1 spatial position = 2 of the 256 tokens. Masked tokens are
dropped before the `SpatialSelfAttention` layer (V1 ran self-attention on all
256 tokens, then replaced 75% of the *output*); visible tokens enter both the
self-attention layer and the predictor. The asymmetry pressure therefore
acts on the only cross-token mixing stages we have (the encoder's
self-attention and the predictor), which is the part of the encoder that can
use it. (Per-conv masking would require a transformer-based encoder per V-JEPA
paper; that is on the [Phase-4 stretch list](#what-we-do-not-demonstrate).)

**Temporal aggregation and predictor.** Visible tokens are concatenated across
frames into a single context sequence. The predictor (3 layers / 4 heads,
`dim=128`, `ffn=256`) receives this sequence *plus* one query token per masked
position. Each masked-position query is constructed as
`temporal_pos[t] + mask_token`, where `mask_token` is a learned 128-dim parameter
and `temporal_pos` is a learned per-frame embedding. In V1, `temporal_pos`
and `mask_token` lived on the V-JEPA module wrapper; V2 moves them into the
predictor so the encoder's representation budget never carries mask-position
information. The predictor outputs one 128-dim vector per masked
position.

**EMA target encoder.** A frozen copy of the context encoder is applied to the
*complete* (un-masked) clip. Targets are the un-masked tubelet embeddings sliced
at the masked positions and detached. The target encoder is updated by
`θ̄ ← τ·θ̄ + (1−τ)·θ` with `τ` on a cosine schedule from 0.99 to 1.0.

**Loss.** True L1 on masked positions only:

```
L_pred = mean_{i ∈ M} | z_pred_i − sg(z_target_i) |_1
L     = L_pred
      + 25 · 0.5 · (var(z_pred) + var(z_ctx))
      + 25 · 0.5 · (cov(z_pred) + cov(z_ctx))
```

where `M` is the masked index set, `sg` is stop-gradient, and `z_ctx` is the
online encoder's output on the visible set. `var` and `cov` are the VICReg
terms from [mini_vjepa/losses.py](mini_vjepa/losses.py). Smooth L1 is *not* used:
at `β=1.0` it degenerates to MSE for our residual range.

**Multi-step rollout (Phase 3 training).** Each step we predict `z_pred(t+1)`,
append it to the visible context, and predict `z_pred(t+2)`, `z_pred(t+3)`. The
reported loss is the average L1 across the three forward steps. This is what
teaches the predictor to be a forecaster rather than a one-shot conditional-mean
estimator.

Parameter count is **0.92 M trainable** (1.39 M including the EMA target). The
Phase-4 wider-trunk + projector lift was not taken — Phase-3 cleared every gate
without it.

### Why these specific changes

**Pixel-patch tubelet masking *before* the encoder.** V1 masked at the
token-output stage: the encoder saw the full clip, and the mask token was
scattered onto its outputs before the predictor ran. This provides *no asymmetry
pressure on the encoder itself* — the encoder is trained on identical inputs at
every step, so it has no reason to allocate capacity to context-completion.
V-JEPA, I-JEPA (Assran et al. 2023), V-JEPA 2 and the toy reference
[keon/jepa](https://github.com/keon/jepa) all mask in input space; the encoder
is forced to produce representations that *enable* completion from partial input.

**True L1 over MSE.** At our 128-dim latent scale the residual magnitudes sit
in the regime where PyTorch `smooth_l1` with default `β=1.0` is numerically
identical to MSE. MSE has a known failure mode for continuous high-mask-ratio
predictors: it minimizes by collapsing to the conditional mean, producing low
loss with near-zero per-sample cosine similarity — exactly the V1 pathology
(cos-sim 0.18 after the eval-mask-leak fix, vs copy-last 0.95). L1 has flatter
gradients away from zero and tolerates the multi-modal target distribution that
masked prediction induces. V-JEPA 2 ships with L1 for this reason.

**Mask token in the predictor, not the encoder.** Putting the mask token on the
encoder side leaks "this position is masked" information into the encoder's
representation budget and lets the predictor's `query_tokens` silently satisfy
the diversity regularizer internally — the iter-3 E4 failure mode the V1
`ctx_effective_rank` monitor was added to catch. The canonical V-JEPA / I-JEPA
design places the mask token strictly at the predictor's input alongside the
position embedding for the masked location.

### What was preserved from V1

The full iter-3 anti-collapse machinery is intact, **except** for the
`mask_ratio=0.75` token-replacement knob (subsumed by the new pre-encoder
tubelet masking):

- VICReg variance + covariance on **both** branches (`reg_target=both`,
  `var_weight=25`, `cov_weight=25`).
- EMA target encoder with cosine `τ` schedule 0.99 → 1.0.
- Dual collapse-stop firing on `effective_rank < 5` *or* `ctx_effective_rank < 5`
  for 8 consecutive epochs.
- NaN-abort guard in the training loop.
- Five monitors logged per epoch: `avg_std`, `effective_rank`, `avg_cosine_sim`
  on `z_pred` plus `ctx_effective_rank` and `ctx_avg_cosine_sim` on encoder
  output. Per Phase 0, `ctx_effective_rank` is now averaged over 10 evenly-spaced
  batches per epoch instead of a single batch.
- Rank-based checkpoint selection (not loss).
- `lr_peak=5e-5`, `grad_value_clip=0.5`, AdamW with cosine LR schedule.

---

## Training & Anti-Collapse Monitoring

V2 keeps every anti-collapse knob earned in the iter-3 ablation (see
[strategies_log.md](strategies_log.md) §3) and changes exactly two things on the
training side: the **masking primitive** and the **loss function**. Pixel-patch
tubelet masking is applied *before* the encoder; the reconstruction loss is true
**L1** over the predictor's output at masked positions only —
`torch.abs(z_pred[masked] - z_target[masked].detach()).mean()` — not Smooth L1.
Phase 3 then extends the loss to a 3-step rollout: `z_pred(t+1)` is concatenated
back into the context window and `z_pred(t+2)`, `z_pred(t+3)` are supervised
against `EMA(frame_{t+k})`, averaging the per-step L1.

The VICReg regularization is untouched (`reg_target=both`, `cov_weight=25`,
`var_weight=25`, applied to *both* the predictor output and the encoder's
context tokens). The optimizer is AdamW with split weight decay (encoder 0.05,
predictor 0.0), peak LR `5e-5` over a 10-epoch linear warmup followed by cosine
decay to `1e-5` ([scripts/train.py](scripts/train.py)), and a hard
`grad_value_clip=0.5` on top of `grad_clip=1.0`. The EMA target encoder follows
a cosine τ schedule `0.99 → 1.0` across 150 epochs
([mini_vjepa/ema.py](mini_vjepa/ema.py)).

Five monitors gate every epoch and land in `runs/<dir>/metrics.csv`:

- `effective_rank` — `exp(entropy(softmax(eigvalsh(Cov(z_pred)))))`, the
  dimensional rank of the predictor's output distribution. Out of `latent_dim=128`.
- `ctx_effective_rank` — same statistic on the *encoder's* context tokens,
  computed per-frame (over the 8×8 spatial grid). Out of
  `min(spatial_tokens=64, latent_dim=128) = 64` per-frame ceiling. V1's
  `z_pred`-only check passed cleanly while the encoder was fully collapsed (E4
  in `strategies_log.md`); this monitor is load-bearing. V2 fixes the
  single-batch noise that drove V1's rank=12 diagnosis by averaging over **10
  evenly-spaced batches per epoch** and logging mean ± std.
- `avg_std` — per-dimension feature standard deviation on `z_pred`; catches
  scale-direction collapse.
- `avg_cosine_sim` — pairwise cosine similarity across the batch on `z_pred`;
  catches direction collapse with intact per-dim variance.
- `ctx_avg_cosine_sim` — same statistic on the encoder's context tokens; the
  encoder-side analogue that catches the E4 "predictor fakes diversity"
  failure mode.

Two safety nets stay armed throughout training:

- **NaN-abort** returns exit code 2 on the first non-finite loss rather than
  continuing to "train" with NaN parameters.
- **Dual collapse-stop** fires when *either* `effective_rank` *or*
  `ctx_effective_rank` stays below `5.0` for 8 consecutive epochs.

V2 adds a third gate, the **per-mask-ratio cos-sim sweep**, run at the 30-epoch
checkpoint of every retrain: predictor cos-sim is measured at eval
`mask_ratio ∈ {0.0, 0.25, 0.5, 0.75, 0.9}` and must degrade monotonically. A
flat curve signals that the predictor memorized one mask distribution rather
than learning the underlying conditional — invisible to all four scalar
monitors. The Phase 2 retrain swept `0.58 / 0.51 / 0.42 / 0.31 / 0.18` at ep149
and the Phase 3 retrain reproduced the monotone profile within ±0.02.

Across both V2 full runs, neither safety net fired and the 30-epoch gate cleared
on the first attempt: `ctx_effective_rank ≥ 8` sustained ep10-29 (mean over 10
batches), `z_pred eff_rank ≥ 30` by ep20, predictor cos-sim at `mask_ratio=0.0`
of 0.31 by ep29.

### Training curves

![Training curves](assets/results/v2/training_curves.png)

Per-epoch `effective_rank`, `ctx_effective_rank`, `avg_std`, `avg_cosine_sim`,
and the L1 loss for the full 150-epoch Phase 3 run. The `ctx_*` curves
(top-right) are mean ± std over 10 batches per epoch — the noise band makes it
visible that `ctx_effective_rank` settled at **19 ± 2 out of 64** from ep30
onward (30% per-frame utilization; V1 reported 12 from a single-batch estimate,
re-measured on the V1 checkpoint over 10 batches yielded 14 ± 3). The `z_pred
effective_rank` curve climbs to **82 / 128** by ep30 and holds above 70 through
ep149 (V1: 67). `avg_std` per dim stayed in [0.42, 0.61] across ep20-149 (V1:
[0.31, 0.55]) and `avg_cosine_sim` stayed below 0.18 (V1 floor: 0.22). Both
rank curves remained above the `collapse_eff_rank_threshold=5.0` line for the
entire 150 epochs. The per-mask-ratio sweep panel (bottom) shows the monotone
`0.58 / 0.51 / 0.42 / 0.31 / 0.18` profile measured at ep149. Phase 3's
multi-step rollout objective is visible as the step in the loss curve at ep0 of
the second retrain; rollout MAE at horizons h=2-10 improves by ~3× over the
Phase-2-only checkpoint.

---

## Dataset & Physics

The simulator is a pymunk-based 2D billiard table in normalized coordinates: a
1.0 × 0.5 rectangle with four cushion segments, no pockets
([simulation/physics.py](simulation/physics.py)). Nine balls of radius 0.03 and
mass 0.170 kg interact through perfectly-elastic-by-default ball–ball
collisions (`elasticity=0.92`), elastic cushion bounces (`elasticity=0.75`),
and an explicit `rolling_friction=0.005` term applied per ball per step.
Integration runs at 120 Hz physics with **`substeps=24`** per frame — three
times the original PLAN §3.4 budget, raised after early sequences showed
cushion-tunneling at high incident speeds. Sub-threshold velocities (`< 0.001`
table-units/s) snap to zero so resting balls stay at rest and reflection laws
remain numerically clean across long sequences.

Initial conditions are drawn from a five-strategy mix
([configs/default.yaml](configs/default.yaml),
[simulation/generator.py](simulation/generator.py)):

| Strategy | Train weight | Geometry |
|---|---|---|
| `break` | 0.30 | Triangular rack at apex (0.7 w, 0.5 h) struck by a cue ball at 3–6 m/s |
| `midgame_cluster` | 0.20 | 3–5 ball cluster with one ball moving at 1.5–3.5 m/s |
| `midgame_spread` | 0.20 | Balls spread across the table, 1–3 moving at 1.0–3.0 m/s |
| `two_ball` | 0.15 | Isolated two-ball collision pattern |
| `random_velocities` | **0 (held-out)** | Random positions, 40% of balls moving at 0.8–3.0 m/s |

The default training set (`data/default.npz`) is 10 000 sequences of 32 frames at
64×64 RGB, sampled every `FRAME_STRIDE=4` physics ticks, giving 30 effective
FPS. Frames are stored as `uint8` to keep the dataset under 2.5 GB in RAM;
positions and velocities are kept as paired `float32` ground truth for probing.
Per-sequence init-strategy labels are saved alongside the frames.

![Billiard simulation](assets/simulation.gif)

**Held-out-regime test split (V2).** The probe suite treats `random_velocities`
as a **test-only** distribution. V2 generates a training dataset with this
strategy's weight zeroed (`configs/default_v2.yaml`, derived from
`configs/default.yaml` with `init_strategies.random_velocities: 0`); the encoder
never sees the regime during training and is evaluated on it cold. This is the
source of the 0.61 position R² held-out number.

**VoE counterfactual generator (V2, Phase 8b).**
`simulation/counterfactual.py` produces 500 matched-pair sequences: a physical
rollout and a counterfactual in which a single ball–ball collision is
artificially skipped (the pymunk contact callback is intercepted on a specific
frame range, so the two balls pass through each other once and the rest of the
trajectory continues physically). Pairs are matched on initial conditions and on
every frame before the divergence point, so any model-side difference between
the two scores is attributable to the counterfactual event alone. The protocol
mirrors *Intuitive Physics Emerges from V-JEPA*
([arXiv 2502.11831](https://arxiv.org/abs/2502.11831)) §3 adapted to 2D
billiards. The stress dataset (15 balls, `configs/stress.yaml`) is generated
but not yet trained on.

---

## Results

All metrics come from [scripts/evaluate.py](scripts/evaluate.py) on the V2
V-JEPA Phase-3 checkpoint and the two pixel baselines, evaluated on
N=2000 held-out sequences. CIs are 95% sequence-level paired bootstrap with
**1000 resamples**; the resampling unit is the test sequence, the probe is fit
once on the train split and held fixed across resamples (the resample varies
only what the probe is evaluated on). The headline encoder is the **online**
context encoder per the V-JEPA / I-JEPA / DINOv2 / BYOL / MoCo convention; the
EMA target is reported only as an ablation.

### Linear probing

![Probe grid](assets/results/v2/probe_grid.png)

| Probe | V-JEPA V2 (online + attentive) | Pixel (from-scratch) | Pixel (frozen V1 enc.) |
|---|---|---|---|
| Per-frame position R² | **0.74** [0.71, 0.77] | 0.61 [0.58, 0.64] | 0.59 [0.55, 0.63] |
| Velocity-difference R² | **0.36** [0.32, 0.40] | 0.04 [0.01, 0.07] | 0.03 [0.00, 0.06] |
| OCP AUC (K=10) | **0.72** [0.69, 0.75] | 0.53 [0.50, 0.56] | 0.54 [0.51, 0.57] |
| Held-out regime (`random_velocities`) pos R² | **0.61** | 0.48 | 0.46 |

**Position probe.** Attentive probe — 1 cross-attention head with 8 learned
query tokens, followed by a linear head — trained with AdamW (lr=1e-3, wd=0.01)
for 200 steps with batch 128 on the train split of the 2000 held-out sequences
(1600/400 train/test inside that split); the encoder remains frozen throughout
and the V-JEPA train split for the encoder is fully disjoint. Reported number is
mean R² across 5 probe seeds. The attentive head is the V-JEPA paper standard;
the +13pp lift includes the +4-17pp typical attentive-over-Ridge bonus.

**Velocity probe.** Difference probe `Δz = pool_S(z_t) − pool_S(z_{t-1})`
followed by RidgeCV (α ∈ logspace(-3, 6, 19)) on StandardScaler-normalized
features. The finite-difference construction is the physically correct decoder.
We report a known caveat: pixel-baseline encoders trained to reconstruct future
frames produce deliberately temporally-smooth features, so their `Δz` is
near-zero by construction. The +32pp lift is therefore partly measuring how
aggressively the pixel encoder low-passes its representation, and not a pure
"V-JEPA encodes velocity better" effect. A pixel baseline with an explicit
next-frame temporal-contrast head would close some of this gap; we did not
train one.

**OCP.** Logistic regression on mean-pooled latents predicting whether balls
*(i, j)* come within `2·radius + ε` in the next `K=10` frames. Positives are
the natural class (24% prevalence on our test set); we report AUC because of
the imbalance, with PR-AUC = 0.59 alongside. Pixel baselines at 0.53–0.54 sit
barely above the class prior — consistent with the hypothesis that next-frame
reconstruction does not require modeling collision geometry.

### Baseline framing — why two pixel predictors

We report two pixel baselines, both sharing V-JEPA's parameter budget and
Δt-conditioned predictor:

1. **From-scratch pixel** ([baselines/pixel_predictor.py](baselines/pixel_predictor.py)):
   the encoder is trained jointly with a transposed-conv decoder on pixel-MSE.
   This is the V-JEPA-paper-standard comparison — same architecture family,
   different objective.
2. **Frozen V1 encoder + pixel decoder** (the readout-isolation diagnostic): we
   load V1's V-JEPA-trained encoder, freeze it, and train only a decoder +
   predictor on pixel-MSE. This does **not** isolate "objective ablation" (the
   V1 encoder was already shaped by VICReg + masking); it isolates whether the
   *encoder's representational space* is intrinsically readable by a pixel
   objective. The 0.59 readout shows that V1's encoder *does* contain enough
   positional information for a pixel decoder to recover, but the from-scratch
   pixel encoder lands in the same band (0.61) — so the
   from-scratch–pixel-vs-V-JEPA gap is not driven by the architecture, and a
   pixel encoder cannot find this positional structure from its own training
   signal alone.

### Predictor and autoregressive rollout

![Per-mask sweep](assets/results/v2/per_mask_sweep.png)
![Rollout MAE](assets/results/v2/rollout_mae.png)

With the inference-time mask leak patched (`--eval-mask-ratio 0.0` default), the
V2 predictor cosine similarity at h=1 is **0.58**, versus V1's reported 0.075
(mask-leak measurement) and V1's true 0.18 (patched re-measurement; see
[reproducer](#what-this-proves-vs-v1) below). The per-mask sweep at evaluation
— `m ∈ {0, .25, .5, .75, .9}` → cos-sim `0.58 / 0.51 / 0.42 / 0.31 / 0.18` —
is strictly monotone, the per-mask gate from [PLAN_V2.md](PLAN_V2.md) §3 Phase 2.

The rollout protocol: starting from a context window at `t_start`, the predictor
emits `z_pred(t+1)`; we append it to context, predict `z_pred(t+2)`, and so on
through `t+H`. We decode each predicted latent to ball positions via the
per-model RidgeCV position probe (trained on each model's *own* latents on the
1600-sequence probe train split) and report MAE in normalized table coordinates
against ground-truth pymunk positions at horizon h. Each model is decoded by
its own probe — the rollout table compares apples to apples.

| Horizon | V-JEPA V2 | Pixel from-scratch | Copy-last-position |
|---|---|---|---|
| h=1 | 0.024 [0.022, 0.026] | 0.035 [0.032, 0.038] | 0.022 [0.020, 0.024] |
| h=2 | 0.034 [0.031, 0.037] | 0.058 [0.054, 0.062] | 0.041 [0.038, 0.044] |
| h=5 | 0.058 [0.054, 0.062] | 0.082 [0.077, 0.087] | 0.078 [0.073, 0.083] |
| h=10 | 0.068 [0.063, 0.073] | 0.102 [0.096, 0.108] | 0.103 [0.097, 0.109] |

MAE is in normalized table-units (1.0 × 0.5 table). 95% sequence-bootstrap CIs
in brackets. Copy-last-position (assume balls remain stationary at the last
context frame) is the relevant trivial baseline; its values track the per-horizon
average ball displacement (~0.025 table-units per stored frame for the
mid-game-speed regimes; lower at h=1 because ~40% of balls are stationary in
the test window). V-JEPA narrowly loses to copy-last at h=1 (decoded-latent
probe noise marginally exceeds one frame of average motion) but beats it from
h=2 onward and beats the pixel-baseline rollout at every horizon. **Probe-noise
sensitivity check**: applying V-JEPA's position probe to pixel-baseline latents
(a shared-probe sanity row) yields h=1 MAE 0.041 and h=10 MAE 0.108 —
pixel-baseline still loses at all horizons, so the per-model-probe choice in
the main table is not what drives the result.

### OCP — what probes can't fake

![Latent PCA trajectory](assets/results/v2/latent_pca_trajectory.png)
![t-SNE colored by speed](assets/results/v2/tsne_by_speed.png)

OCP (Object Contact Prediction, Physion-family) asks a logistic regression on
frozen encoder features: *"will balls i and j come within 2·radius + ε of each
other in the next K=10 frames?"* This is a forward physical reasoning task —
it cannot be solved by a probe that only recovers static positions, because it
requires the encoder to internalize trajectories and collision geometry.
V-JEPA reaches AUC **0.72**; both pixel baselines sit at 0.53–0.54, matching
the class prior. The PCA trajectory plot shows V-JEPA latents traversing
structured manifolds in the leading principal subspace; t-SNE colored by mean
ball speed shows a clean speed gradient that V1 lacked.

### Encoder geometry

`ctx_effective_rank` over 10 batches/epoch (Phase 0 fix to F2): **19 / 64**
(30% per-frame utilization; V1 reported 12 from single-batch noise; V1
re-measured over 10 batches on the same checkpoint: 14 ± 3). `z_pred` effective
rank: **82 / 128** (V1: 67). Trainable parameters: 0.92 M (1.39 M total
including EMA target).

---

## VoE Counterfactual Screen

Linear probing measures what a frozen encoder *can be read out as*. It cannot
measure whether the encoder's latent dynamics respect the physics it appears
to encode (*Intuitive Physics Emerges from V-JEPA*,
[arXiv 2502.11831](https://arxiv.org/abs/2502.11831);
[PLAN_V2.md](PLAN_V2.md) §1 S6). We adapt Meta's Violation-of-Expectation
protocol to 2D billiards.

Each VoE *trial* is a matched pair of 32-frame sequences sharing identical
initial conditions: a **physical** sequence (collision A↔B resolved by pymunk)
and a **counterfactual** sequence in which the A↔B contact is silently skipped
on a designated frame, so A and B pass through one another exactly once and
continue ballistically (`simulation/counterfactual.py`). 500 pairs are sampled
with collision frame, ball identities, and incidence angle controlled.

**Scoring (mirrors arXiv 2502.11831 §3 verbatim).** For each sequence we
encode the full trajectory with each model's online encoder, then for each
post-event frame we compute the model's own *surprise* as the L1 distance
between its predictor's forward output and its own EMA target on that frame.
Per-trial *cumulative surprise* is the sum of surprise across the post-event
window. **The headline metric is pairwise accuracy**: the fraction of trials
on which the counterfactual sequence's surprise *exceeds* the physical
sequence's surprise. A model that has internalized collision dynamics should
score above 50%; a model that merely decodes static positions cannot
distinguish the pair until they overlap and is near chance. Each model is
scored on its own predictor/target — no cross-model ground truth is used —
so the comparison is symmetric.

| Model | Pairwise accuracy | Mean Δ surprise (CF − physical) |
|---|---|---|
| **V-JEPA V2 (ours)** | **84%** [80, 87] | +0.31 [+0.27, +0.34] |
| Pixel baseline (from-scratch) | 56% [52, 60] | +0.04 [+0.01, +0.07] |

V-JEPA's 84% is in the 66-86% band Meta reports on the original real-video
benchmark across context types. The pixel baseline at 56% is barely above
chance — its decoder is happy to render two balls passing through each other
and its encoder has no incentive to flag the impossibility. V-JEPA's latent
objective forces the encoder + predictor to model the *event*, not the
rendering.

![VoE rank distribution](assets/results/v2/voe_distribution.png)

---

## What this proves vs V1

V1's headline claim — "pixel baseline narrowly beats V-JEPA on positions
(0.616 vs 0.567)" — was a compound of six distinct evaluation-pipeline bugs,
each catalogued in [docs/eval_failure_modes.md](docs/eval_failure_modes.md)
(F1–F12). V2 fixes each before retraining, then re-runs the same eval harness
on both checkpoints.

**(a) Inference-time mask leak (F1).** V1's `scripts/evaluate.py` reproduced
the 75% training-time masking on every cos-sim and rollout measurement. The
patched V1 predictor's true cos-sim at h=1 was **0.18**, not the reported
0.075. V2's 0.58 is a ~3× lift over correctly-measured V1.
**Reproducer:** `git checkout v1-eval-patched && python scripts/evaluate.py
--ckpt runs/20260519_195827_iter3_full_150/ckpt_best.pt --eval-mask-ratio 0.0
--max-eval-sequences 2000`.

**(b) `ctx_effective_rank` single-batch measurement (F2).** V1's "rank 12 /
128" diagnosis was measured on a single batch per epoch. Re-measured over 10
batches with mean ± std, true V1 rank was **14 ± 3 / 64**; V2 lifts this to
**19 / 64**. The /64 denominator (per-frame mathematical ceiling of
`min(spatial_tokens=64, latent_dim=128)`) replaces V1's misleading /128
framing. The architectural pivot to pre-encoder tubelet masking was driven by
the *correctly-framed* gap (14 vs the 32+ rank typical of healthy V-JEPA
encoders), not the V1 inflation.

**(c) Pixel-baseline self-reference (F3).** V1's pixel baseline re-instantiated
`ContextEncoder` from `mini_vjepa.encoder` — "V-JEPA loses to pixel" was
comparing two encoders of *identical capacity* trained on different
objectives. The V2 frozen-V1-encoder readout-isolation baseline shows the
encoder's representational space is reachable by either objective (both pixel
variants land at 0.59-0.61); the gap to V-JEPA's 0.74 therefore reflects
*what V-JEPA chooses to encode*, not encoder family.

**(d) Under-regularized Ridge at 32K features (F4, F9).** V1's
context-window probe concatenated `(4, 64, 128)` → 32 768 features and fit
Ridge with α ∈ logspace(-3, 3, 13). Sklearn emitted
`LinAlgWarning: rcond=8.5e-8`; the α grid was silently truncated, producing
the spurious **−0.59 velocity R²**. V2 uses mean-pool + StandardScaler +
RidgeCV(logspace(-3, 6, 19)).

**(e) Per-frame velocity probe was physically ill-posed (F9).** A single
64×64 frame contains no motion. V2 replaces the per-frame probe with the
difference probe `Δz = pool_S(z_t) − pool_S(z_{t-1})`, matching the physical
definition; V-JEPA jumps from 0.022 to **0.36** velocity R². (Caveat about
pixel-encoder smoothness bias noted under [Velocity probe](#linear-probing).)

**(f) Wrong headline encoder (S2).** V1 headlined the EMA target, but
V-JEPA / I-JEPA / DINOv2 / BYOL / MoCo all probe the **online** encoder. V2
reports online as headline, EMA as ablation row.

---

## What we do NOT demonstrate

We mirror V1's honesty section because the same standard applies.

- **The 15-ball stress dataset is not trained on.** All numbers in this README
  are on the 9-ball default distribution. Scaling-to-harder-regimes is still a
  stretch goal.
- **The predictor loses to copy-last-position at h=1** (MAE 0.024 vs 0.022 in
  the rollout table). Decoded-latent rollout carries irreducible probe noise;
  copy-last-position is near-exact at one frame for the ~40% of balls that are
  stationary in the test window. The predictor's win starts at h=2 and grows
  with horizon.
- **The VoE protocol covers one counterfactual type only** (collision-skip).
  Meta's original paper covers six. Generalizing to teleportation, object
  permanence, energy conservation, and continuity counterfactuals is owed and
  on the Phase-9 stretch list.
- **The frozen-V1-encoder diagnostic is a readout-isolation test, not an
  objective ablation.** Freezing a JEPA-trained encoder onto a pixel decoder
  measures *whether the encoder's representation is decodable by a pixel head*
  — not what would happen if V1's *exact* training procedure were swapped to
  pixel-MSE. The latter is the from-scratch pixel baseline, and we report both.
- **Probe capacity is not MDL-controlled.** We pin the attentive probe to a
  single configuration (1 head, 8 query tokens, 200 AdamW steps) and report
  across 5 seeds. A more rigorous study would sweep probe capacity per
  Belinkov (2022).
- **Velocity-probe lift is partially biased toward V-JEPA** by the
  pixel-baseline's temporal smoothness (see Linear probing caveat above). The
  true objective-driven gap on velocity is smaller than +32pp; after
  variance-equalizing the pixel baseline's `Δz` per dim before RidgeCV — a
  crude bias correction that removes the "near-zero `Δz` variance" advantage —
  the corrected gap is +0.20 [+0.17, +0.23]. A properly-controlled fix would
  re-train a pixel baseline with an explicit next-frame temporal-contrast head;
  we did not.
- **`ctx_effective_rank` is 19 / 64 = 30% utilization** — short of the 32+
  typical of full-scale V-JEPA. Phase 4's wider-trunk + projector lift is the
  next intervention if a future version aims for parity. We did not take it
  because Phase 3 already cleared the load-bearing OCP and VoE gates.
- **Compute budget on a single M3 caps generalization claims.** All results are
  on 10K sequences, 150 epochs, float32 only.

---

## Reproducibility & Compute

- Hardware: single MacBook Pro M3 (unified memory). No external GPU.
- Precision: float32 only. No autocast, no bf16, no fp16.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` set in `scripts/train.py` so
  `torch.linalg.eigvalsh` (used by `effective_rank`) transparently falls back
  to CPU; every monitor evaluation pays one CPU sync.
- Wall time: ~110 min per full 150-epoch run (V1: ~90 min; the +20 min is
  dominated by the 10-batch monitor sampling and the multi-step rollout in
  Phase 3).
- Compute budget consumed: **3 of 8** allotted full 150-epoch runs (Phase 2
  retrain, Phase 3 retrain, one re-validation after the Phase 0 patches
  landed).
- Memory accounting: `ps`-reported RSS undercounts MPS unified memory by ~6×.
  RAM headroom at N=2000 eval sequences was verified with macOS `sample <pid>`
  ("Physical footprint" header) before each full run.
- Reproduction: `python scripts/train.py --config configs/default_v2.yaml
  --data data/default.npz --tag v2_phase3 --rollout-steps 3` deterministically
  reproduces the Phase 3 checkpoint at seed 42.

---

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Generate the default V2 dataset (random_velocities zeroed for held-out test)
python scripts/generate_data.py --config configs/default_v2.yaml

# 2. 30-epoch validation gate. Confirms the V2 anti-collapse stack survives:
#    ctx_effective_rank ≥ 8 sustained ep10-29 (10-batch mean),
#    z_pred eff_rank ≥ 30 by ep20, predictor cos-sim @ mask_ratio=0.0 ≥ 0.25
#    by ep29, monotone per-mask sweep.
python scripts/train.py \
    --config configs/default_v2.yaml \
    --data data/default.npz \
    --epochs 30 --lr-schedule-epochs 150 \
    --tag v2_validation

# 3. Full Phase 2 V-JEPA run (~110 min on M3)
python scripts/train.py \
    --config configs/default_v2.yaml \
    --data data/default.npz \
    --tag v2_phase2

# 4. Phase 3 multi-step rollout retrain (resumes from Phase 2 ckpt, ~110 min)
python scripts/train.py \
    --config configs/default_v2.yaml \
    --data data/default.npz \
    --resume runs/<phase2>/ckpt_best.pt \
    --rollout-steps 3 \
    --tag v2_phase3

# 5. Pixel baselines (~90 min each). Two variants:
python scripts/train.py --baseline pixel --config configs/default_v2.yaml \
    --data data/default.npz --tag pixel_from_scratch
python scripts/train.py --baseline pixel --init-from-vjepa \
    runs/<v1>/ckpt_best.pt --freeze-encoder \
    --config configs/default_v2.yaml --data data/default.npz \
    --tag pixel_frozen_v1_encoder

# 6. Evaluate (online encoder + attentive probe + difference-velocity + OCP +
#    autoregressive rollout + sequence-level bootstrap CI)
python scripts/evaluate.py \
    --ckpt runs/<v2_phase3>/ckpt_best.pt \
    --data data/default.npz \
    --config configs/default_v2.yaml \
    --pixel-ckpt runs/<pixel_from_scratch>/ckpt_best.pt \
    --pixel-frozen-ckpt runs/<pixel_frozen_v1_encoder>/ckpt_best.pt \
    --eval-mask-ratio 0.0 \
    --max-eval-sequences 2000 \
    --bootstrap-resamples 1000 \
    --out assets/results/v2/

# 7. VoE counterfactual screen (~10 min)
python scripts/voe_eval.py \
    --ckpt runs/<v2_phase3>/ckpt_best.pt \
    --pixel-ckpt runs/<pixel_from_scratch>/ckpt_best.pt \
    --n-pairs 500 \
    --out assets/results/v2/voe_distribution.png
```

For a fast smoke check, replace step 3 with `--debug` (2 epochs on
`data/sample/sample.npz`). The architecture diagram is at
`assets/architecture_v2_a_wide.png`.

---

## References

| Paper | Role |
|---|---|
| [V-JEPA (Bardes et al., 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | Architecture reference; canonical pre-encoder masking |
| [V-JEPA 2 (Meta, 2025)](https://arxiv.org/abs/2506.09985) | L1 loss, world-model planning at scale |
| [I-JEPA (Assran et al., 2023)](https://arxiv.org/abs/2301.08243) | Image predecessor; multi-block masking |
| [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) | Philosophical foundation |
| [Intuitive Physics Emerges from V-JEPA (Garrido et al., 2025)](https://arxiv.org/abs/2502.11831) | VoE protocol — primary evaluation precedent |
| [C-JEPA (2024)](https://arxiv.org/abs/2410.19560) | VICReg-on-target + invariance at small scale |
| [VICReg (Bardes et al., 2021)](https://arxiv.org/abs/2105.04906) | Anti-collapse regularization |
| [Kobak et al. (JMLR 2020)](https://jmlr.org/papers/v21/19-844.html) | Ridge probing methodology |
| [Belinkov (CL 2022)](https://direct.mit.edu/coli/article/48/1/207/107571/) | Probing-classifier pitfalls |
| [Physion (Bear et al., 2021)](https://arxiv.org/abs/2106.08261) | OCP benchmark family |
| [keon/jepa](https://github.com/keon/jepa) | Toy V-JEPA on Moving MNIST — canonical masking reference |

---

## License

MIT. See [LICENSE](LICENSE).
