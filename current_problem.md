# Current Problem — Training Iteration 2 Failure

**Date:** 2026-05-19
**Run dirs:** `runs/20260519_120319/` (V-JEPA), `runs/baseline_pixel/20260519_122621/` (pixel)
**Log:** `logs/full_run.log`
**Status:** Both training runs auto-aborted by the safety nets. Neither produced a usable model.

---

## TL;DR

The safety nets added after Iteration 1 worked exactly as designed (caught both failures early, saved ~70 min of wasted compute). The actual fixes to the loss / LR / EMA did **not** prevent the underlying failures:

- **V-JEPA still collapsed dimensionally** despite the new VICReg covariance term. The cov term fired (grew to 14× contribution by epoch 27) but the gradient was insufficient to overpower the MSE pull toward the trivial solution.
- **Pixel baseline still NaN'd**, but later (epoch 60 vs epoch 40 previously). Lower LR + value clip helped but didn't solve.

The first-round independent reviewers gave clean **GO** verdicts. They were wrong. Their numerical estimates of cov_reg behavior did not match the real-run numbers by an order of magnitude.

---

## V-JEPA failure trajectory

Full sequence of events from the metrics CSV:

| epoch | loss | mse | 25·var | 1·cov | avg_std | **eff_rank** | cos_sim | notes |
|---|---|---|---|---|---|---|---|---|
| 0 | 22.66 | 1.89 | 20.77 | 0.0 | 0.169 | 20.85 | 0.971 | random init |
| 9 | (warmup) | — | — | — | — | — | — | — |
| 19 | 16.83 | 1.73 | 13.99 | 1.11 | 0.713 | **15.27** | 0.498 | warmup complete, healthy |
| 20 | 16.24 | 1.73 | 11.79 | 2.72 | 0.536 | 7.90 | 0.703 | rank dropping |
| 21 | 20.15 | 2.97 | 15.14 | 2.04 | 0.466 | **4.22** | 0.763 | below threshold (5.0) |
| 22 | 21.81 | 1.09 | 18.54 | 2.17 | 0.404 | **1.64** | 0.798 | fully collapsed |
| 23 | 24.83 | 1.08 | 23.73 | 0.015 | 0.047 | 1.67 | 0.997 | low-std also triggered |
| 24 | 20.28 | 1.31 | 12.96 | 6.01 | 0.473 | 2.14 | 0.747 | cov term growing |
| 25 | 20.80 | 1.30 | 13.32 | 6.18 | 0.430 | 2.19 | 0.789 | WARNING streak=5 |
| 26 | 22.25 | 1.21 | 15.03 | 6.01 | 0.375 | 1.86 | 0.835 | WARNING streak=6 |
| 27 | 26.41 | 1.14 | 11.06 | **14.21** | 0.558 | 1.61 | 0.630 | cov maxes out |
| 28 | 26.84 | 1.36 | 11.72 | 13.76 | 0.468 | 1.88 | 0.744 | STOP fired |

### Wall time
- V-JEPA aborted at epoch 28 of 150: ~17 min wall (saved ~75 min vs old behavior).

### What this proves
- The collapse-detect-and-stop safety net **worked perfectly**. Stopped at the right moment.
- The covariance regularizer at `cov_weight=1.0` is **not strong enough** to prevent collapse on this setup. It fires (cov_reg grew from 0 → 14.2) but the gradient signal does not overpower MSE.
- The model reviewer's hypothetical "cov dominates 10× at eff_rank≈4" was off by an order of magnitude. Real values: cov=2.04 at eff_rank=4.22 (cov contribution ~10% of total loss, not 10×).
- avg_std remained "healthy" (0.4–0.7) through most of the collapse, confirming again that `avg_std < 0.05` is the wrong alarm for dimensional collapse.

### Pre-collapse symptoms (epoch 19→22)
- `cos_sim` rose 0.50 → 0.80 (token-pair similarity increasing → dimensions becoming correlated)
- `eff_rank` fell 15.27 → 1.64 in **3 epochs**
- `mse` actually decreased (1.73 → 1.09) — the model was "fitting" by finding the trivial solution
- `25·var` stayed in 12–19 range — variance reg was active but local

---

## Pixel baseline failure trajectory

| epoch | loss / pixel_mse | lr | notes |
|---|---|---|---|
| 0 | 0.0906 | 5.03e-6 | warmup start |
| 10 | 0.0387 | 5.00e-5 | warmup complete (lr_peak from override) |
| 20 | 0.0253 | 4.94e-5 | converging well |
| 29 | 0.0210 | 4.80e-5 | last logged healthy |
| ... | (gap to epoch 60) | | |
| 60 | NaN | (one bad batch) | abort exit code 2 |

### Wall time
- Pixel baseline aborted at epoch 60 of 150: ~45 min wall (saved ~75 min vs old behavior).

### What this proves
- The new `torch.isfinite(loss)` abort check **worked perfectly**. No more 100+ epochs of training NaN.
- `lr_peak=5e-5` + `grad_value_clip=0.5` extended the healthy regime from epoch 40 → 60 (50% improvement), but didn't eliminate the divergence. The decoder is still unstable at this LR.
- LR at the NaN moment was ~3.5e-5 (already in cosine decay), so a tiny LR alone won't fix it.

---

## Comparison to Iteration 1 (the original failure)

| Metric | Iteration 1 | Iteration 2 (this run) |
|---|---|---|
| V-JEPA epochs run | 120 | 28 |
| V-JEPA fate | crashed at epoch 120 (eigvalsh) | clean abort at epoch 28 (collapse-stop) |
| V-JEPA min eff_rank | 1.0 | 1.6 |
| Pixel baseline epochs run | 150 (110 of them NaN) | 60 |
| Pixel baseline fate | continued for 110 NaN epochs | clean abort at first NaN |
| Wasted compute | ~3h | ~1h |
| Usable models | none (best ckpt = random init) | none (no best ckpt — collapsed before warmup-end gating activated, which is correct) |

The safety improvements (collapse-stop, NaN-abort, eff-rank-gated best, eigvalsh try/except) all worked. The model improvements (cov term, higher var weight, baseline LR override) all helped but were not sufficient.

---

## Root cause analysis

### V-JEPA: why cov_weight=1.0 was insufficient

VICReg paper uses `(sim=25, var=25, cov=1)` on ImageNet embeddings of dim 8192. On 8192 dims, the off-diagonal cov matrix has ~67M entries, so summing their squares produces a naturally large number — `cov=1` is fine because cov_reg is already in the tens or hundreds before weighting.

On our setup:
- Latent dim = 128 → 16K off-diagonal entries
- z_pred shape = `(B, 64, 128)` flattened to `(B·64, 128)` ≈ `(4096, 128)`
- cov values stayed in `O(1)` range throughout training (peak 14.2)

So the cov gradient was always dominated by var_reg (weight 25) in absolute terms. The "fix" was directionally right but magnitude-wrong.

### V-JEPA: why MSE pulled to trivial solution

The MSE target is `model.target_encode(target_frame)` — the EMA target encoder's view of a future frame. With `ema_tau_start=0.996`, the target moves very slowly. So the predictor learns to match a near-stationary target. The "easiest" target is a constant vector → both online and target collapse together.

### Pixel baseline: why decoder still NaN'd

Suspect: the final `Sigmoid` in `PixelDecoder` (line 47-ish of `baselines/pixel_predictor.py`). When pre-sigmoid logits drift to extreme values (e.g. > 50 in absolute value), the sigmoid saturates, MSE gradient through saturated sigmoid is exactly zero for that element but huge for the few un-saturated pixels — eventually one batch produces `inf` activations during forward pass.

Possible alternatives:
- Replace sigmoid with `torch.clamp(x, 0, 1)` (no gradient saturation issue)
- Predict logits and use `binary_cross_entropy_with_logits` instead of MSE
- Even lower LR (1e-5 peak)
- Weight decay on decoder
- Smaller decoder

---

## Independent reviewer track record

The independent-reviewer methodology has caught structural bugs (per-batch monitor waste, missing NaN-abort, ckpt_best selecting random init) but has **twice now failed to predict collapse dynamics**.

Round 1 reviewers (before first failed run): Both reviewers gave "GO with caveats". The caveats they raised did not include "var_loss_weight=1.0 will not prevent collapse." The collapse happened.

Round 2 reviewers (after fixes, before this failed run): Both reviewers gave clean "GO". The model reviewer explicitly defended `cov_weight=1.0` with a table showing cov should dominate by 10× at eff_rank≈4. The real-run number was cov=2 vs 25·var=15 — i.e., cov was ~10% of var, not 10× of var.

**Pattern:** the reviewers do excellent code-level audits (math, shapes, gradient flow, edge cases) but their numerical predictions about training dynamics are unreliable. They cannot substitute for running an actual short training and watching the metrics.

**Implication for next iteration:** before launching another 150-epoch run, do a 30-epoch debug run on the default dataset with the proposed fixes to confirm eff_rank is stable. ~7 min of compute to validate ~3 hours.

---

## Proposed fixes (Iteration 3) — original intuitions

The fixes I proposed when I first wrote this doc were:
1. `cov_loss_weight: 1.0 → 25.0` (match `var_loss_weight` scale).
2. `ema_tau_start: 0.996 → 0.99`.
3. `collapse_eff_rank_threshold: 5.0 → 10.0`.
4. Maybe lower `lr_peak: 1.5e-4 → 5e-5`.

**These intuitions were partially right but partially wrong.** I ran them as ablations and they failed in informative ways. See `strategies_log.md` for the full ledger; the summary follows.

---

## What the iter-3 ablation actually found (2026-05-19, ~3 h of MPS time)

I ran a structured single-axis ablation on `default.npz` with `--epochs 30 --lr-schedule-epochs 150` (preserves the iter-2 LR profile but caps wall time):

| Experiment | Change | Outcome |
|---|---|---|
| **E0_def** | iter-2 config as-is | Confirmed catastrophic collapse at ep18 (min ctx_eff_rank ~ undefined, z_pred=1.8). |
| **E1** | `cov_loss_weight=25` only | **COLLAPSED FASTER** (ep9). The fix I'd proposed in this doc actually accelerated collapse. |
| **E2** | `reg_target=context` only | **COLLAPSED EARLIER** (ep6). Reg on encoder output fights spatial autocorrelation of CNN features. |
| **E4** | `mask_ratio=0.75` only | **DECEPTIVELY HEALTHY**: z_pred eff_rank stayed at 16-22 the entire 30 epochs, but **ctx_eff_rank=2-4 with ctx_cos_sim=0.97 — encoder fully collapsed**. The predictor's `mask_token + pos_embed + query_tokens` were faking diversity. Existing collapse-stop did not fire because it only watches z_pred. |
| **E7** | `reg_target=both` + `cov_loss_weight=25` + `ema_tau_start=0.99` + `mask_ratio=0.75` | **PARTIAL SUCCESS**: first config that did not collapse. ctx_eff_rank=5-11 throughout. z_pred peaks at 80. But late-stage z_pred volatility (ep27-29: 79→7→14). |
| **E_arch** | E7 + encoder spatial pos_embed | **NaN at ep20.** No rank improvement over E7 before that. Encoder permutation-equivariance is NOT the bottleneck. |
| **E_lowlr** | E7 + `lr_peak=5e-5` | **BEST**: z_pred floor 23 (vs E7's 7), peaks at 82, ctx_eff_rank 5-13 throughout. Stable. |

**Best single epoch across all experiments:** E_lowlr ep18 — z_pred eff_rank=82, ctx_eff_rank=12.0 simultaneously. This is the only point we've seen where both metrics are simultaneously healthy.

---

## Root cause (revised)

The collapse on `default.npz` has **three** load-bearing causes that compound at scale, **none of which iter-2 reviewers identified**:

1. **The predictor has independent diversity machinery.** Its `query_tokens` (64, 128) + `pos_embed` (321, 128) + `mask_token` (128) can produce diverse z_pred outputs regardless of context. Any var/cov reg applied only to z_pred is satisfied internally by the predictor; the encoder is freed of pressure and collapses. This is why E1 (cov×25 on z_pred) *accelerated* collapse — heavier z_pred reg = more "diversity work" the predictor handles internally = less pressure on the encoder.

2. **`cov_loss_weight=1.0` was undersized in two senses.** The reviewer's table was 10× off, and the iter-2 hypothesis (just bump cov to 25) was insufficient because of cause #1 above. Cov weight alone doesn't matter — it has to be applied where it constrains the encoder.

3. **The current collapse-stop is half-blind.** It checks z_pred eff_rank only. E4 showed the failure mode where z_pred eff_rank stays high (16-22) while the encoder is fully collapsed (ctx_eff_rank=2-4). Without the `ctx_*` metric I added for this investigation, this would silently pass every check and ship a useless model.

The minimal sufficient fix needs all of:
- `reg_target=both` (pressures both encoder and predictor; reg can't be satisfied internally on one side).
- `cov_loss_weight=25` (cov gradient strong enough to actually push features apart in 128 dims).
- `ema_tau_start=0.99` (target moves fast enough that the predictor can't lock onto a static collapsed target).
- `mask_ratio=0.75` (V-JEPA-style asymmetry; predictor depends on visible context not just its own params).
- `lr_peak=5e-5` (the four-constraint loss surface is too narrow at 1.5e-4 — E7 showed late instability that disappears at 5e-5).

Three of the four small "fixes" I originally proposed do NOT work in isolation. They are *jointly* sufficient, individually insufficient.

---

## Concrete iter-3 prescription (applied as part of this investigation)

### Already applied to `configs/default.yaml`
```yaml
training:
  lr_peak: 5.0e-5           # was 1.5e-4
  ema_tau_start: 0.99       # was 0.996
  cov_loss_weight: 25.0     # was 1.0
  reg_target: both          # new key (default = "z_pred" if absent)
  mask_ratio: 0.75          # new key (default = 0.0 if absent)
  loss_fn: mse              # new key (default = "mse" if absent)
  grad_value_clip: 0.5      # new key (added after v1 stochastic NaN at ep20)
```

Note on `grad_value_clip`: the first iter-3 validation run (v1) hit a non-finite loss at ep20 with the
`lr=5e-5 + reg=both + cov=25 + ema=0.99 + mask=0.75` config (the NaN-abort caught it cleanly).
A second run (v2, same config) didn't NaN but was rank-poor (z_pred dipped to 15). A third run (v3)
with `grad_value_clip=0.5` added produced the cleanest training of the whole investigation —
z_pred eff_rank locked at 79-82 from ep18 onward, ctx_eff_rank monotonically growing 5→9. The
value clip is the same defense the pixel baseline already uses; it costs nothing during normal
training and converts stochastic NaNs into bounded updates.

### Already applied to `scripts/train.py`
- **NaN-abort** in the V-JEPA loop (`if not torch.isfinite(loss): return 2`). Previously only the pixel baseline had this; E_arch hit a NaN and "trained" for 3 more useless epochs before I killed it.
- **Dual collapse-stop**: trigger on `ctx_effective_rank < threshold` OR `z_pred effective_rank < threshold`. E4 silently passed the old check.
- **`ctx_*` metrics** in `metrics.csv` and the per-epoch log line. Without these, the predictor-hides-encoder failure mode is invisible.
- New CLI knobs: `--override key.path=value` (lets us run ablations without spawning yaml files), `--lr-schedule-epochs N` (lets us preview the first N epochs of a 150-epoch schedule), `--tag` (run-dir labelling).

### Already applied to `mini_vjepa/{vjepa,losses,encoder}.py`
- `compute_loss` accepts `context_tokens` and `reg_target ∈ {z_pred, context, both}`. Default `z_pred` preserves iter-2 behavior so any existing tests still pass.
- `apply_context_mask` + learned `mask_token` parameter for the V-JEPA asymmetry.
- Optional `smooth_l1_loss` (V-JEPA paper uses smooth L1; not enabled by default since the experiments didn't isolate it as load-bearing).
- Optional `use_spatial_pos_embed` on the encoder (default off — E_arch showed no benefit and a NaN-stability cost; leave off).

### Validation gate before re-running 150 epochs

Run this and check the criteria *before* committing to a 3 h training run:
```bash
.venv/bin/python scripts/train.py \
    --config configs/default.yaml \
    --data data/default.npz \
    --epochs 30 \
    --lr-schedule-epochs 150 \
    --tag iter3_validation
```

**Pass criteria** (must hold for all of ep10..ep29):
- `effective_rank ≥ 7` (z_pred)
- `ctx_effective_rank ≥ 5`
- No NaN
- No collapse-stop fires

If any criterion fails, do NOT proceed to the full run. Investigate.

---

## What's still not solved (caveats on the "fix")

E7/E_lowlr keep `ctx_eff_rank ≤ 13` on its best epochs. For a 128-dim feature space, well-trained ViT/CNN encoders reach `eff_rank > 30`. We're a factor of 2-3× below that, which **will cap downstream probing R²**. Iteration 3 will train without collapsing but the representation is rank-limited.

Likely next-iteration work to lift the ceiling (NOT validated in this round — listed in order of expected impact / cost ratio):
1. Increase `latent_dim` from 128 to 256 or 384 so VICReg is closer to its calibrated regime.
2. Add a projector head (~1024-dim) for VICReg, with the 128-dim backbone used for probing.
3. Reduce predictor's intrinsic diversity (zero-init `query_tokens`, smaller `pos_embed` magnitude).
4. Switch to V-JEPA's true pixel-level masking (online encoder sees masked input, target sees full).
5. Use SmoothL1 instead of MSE (paper does; minor expected effect).

---

## Pixel baseline (unchanged from original proposal)

The sigmoid/NaN diagnosis from the original section still stands; not investigated further in this iteration since the V-JEPA collapse was the blocker. Apply the proposed pixel-decoder change:
- Replace `torch.sigmoid(h)` with `torch.clamp(h, 0, 1)` in `PixelDecoder.forward`, or
- Switch to `binary_cross_entropy_with_logits` and skip the sigmoid entirely.

---

## Files touched by this investigation

| File | Change |
|---|---|
| `configs/default.yaml` | Apply the five-knob iter-3 fix above + new key documentation. |
| `mini_vjepa/vjepa.py` | `compute_loss` extended with `reg_target` + `context_tokens`; added `mask_token` and `apply_context_mask`. |
| `mini_vjepa/encoder.py` | Optional `use_spatial_pos_embed` (off by default). |
| `mini_vjepa/losses.py` | Added `smooth_l1_loss` (unused unless `loss_fn=smooth_l1`). |
| `scripts/train.py` | `--override`, `--lr-schedule-epochs`, `--tag`; NaN-abort; dual collapse-stop; `ctx_*` metrics. |
| `scripts/compare_runs.py` | New script — side-by-side comparison of metrics.csv with z_pred + ctx ranks. |
| `scripts/run_iter3_ablation.sh` | New script — sequential runner for the iter-3 ablation experiments. |
| `strategies_log.md` | New doc — full experimental ledger of iter-3 investigation. |
| `current_problem.md` | (this file) revised with the findings.
