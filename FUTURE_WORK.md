# Future Work

Iter-3 shipped V1: V-JEPA trains 150 epochs cleanly on `data/default.npz` and
produces all 5 evaluation figures + `metrics.json`. The model does not
collapse, but the encoder is rank-limited (`ctx_effective_rank ≈ 9-12` out of
128 max), which caps downstream linear-probing R² and lets the pixel baseline
narrowly win on position probing.

The two unfinished investigations below are the obvious next steps. Both are
deliberately deferred from V1 because either would have required another
multi-iteration training+evaluation loop.

---

## Option B — Larger latent dimension (~4-6 h compute, small code change)

**Hypothesis:** `ctx_effective_rank ≈ 12` is structurally capped by the fact
that VICReg's covariance term has degenerate axis-aligned minima at low
feature dimension. VICReg paper used `d=8192`; we use `d=128`. Scaling the
latent dim up to 256 or 384 should relax the cap because:

- More dims = more "room" for the decorrelation constraint to spread variance.
- The cov-reg's rank-K-axis-aligned degenerate-min argument applies at any
  `K < d`, but the relative *cost* of staying axis-aligned grows with `d`
  (more dims that have to be made deliberately silent).
- Real V-JEPA / I-JEPA encoders use 384-1280 dim and don't show this ceiling.

**Implementation:**
- `configs/default.yaml`: `model.latent_dim: 128 → 256` (or 384).
- That single yaml change propagates through `mini_vjepa/encoder.py` (proj
  output), `predictor.py` (dim), `vjepa.py` (`mask_token`, `temporal_pos`
  shapes), `ema.py` (target encoder follows automatically).
- Parameter count roughly doubles. Per-epoch wall time on M3 will grow ~30-40 %.

**Validation gate:** the 30-epoch probe with `--lr-schedule-epochs 150` should
hold `effective_rank ≥ 30` (current ceiling was ~12 at d=128). Anything below
~25 means dim alone is insufficient and Option C is required.

**Risk:** the pixel baseline also benefits from higher latent dim (its
decoder can carry more info). If both improve proportionally, the relative
comparison doesn't change. The V-JEPA advantage either materializes or it
doesn't — at d=256 we'll know which.

**Compute budget:** ~3 h V-JEPA + ~1.5 h pixel baseline + ~30 min eval = 5 h.

---

## Option C — Add a projector head for VICReg (~1-2 days)

**Hypothesis:** even at d=128 the encoder backbone can produce a rich
representation, but VICReg pressure on the raw backbone output forces the
backbone to spend capacity satisfying the decorrelation constraint. V-JEPA
paper, SimCLR, BYOL, and DINO all use a projector head (a small MLP that
maps backbone features into a higher-dim "projection space") and apply the
SSL loss only in the projection space. The backbone is then free to learn
whatever it likes — including being lower-rank in the projection space — as
long as the projection is well-decorrelated. Downstream tasks (linear
probing) use the *backbone*, not the projector.

**Implementation:**
- New `Projector` module: `Linear(128, 512) → GELU → Linear(512, 1024)`.
- `vjepa.py`: a `proj_head` that's applied to both `z_pred` and `z_target`
  before the MSE / var / cov terms. Stop-grad on target side runs *before*
  projection.
- `train.py`: ensure the new params go into the predictor weight-decay group.
- `evaluate.py`: probes still use the backbone's `target_encoder` output,
  NOT the projected version. The whole point of the projector is to keep
  the backbone clean for downstream use.

**Validation:** same 30-epoch probe + 150-epoch full run protocol.

**Risk:** more knobs to tune (projector width, depth, init). The MLP shapes
above are a starting guess from the V-JEPA paper; they might need scaling.

**Compute budget:** ~half a day for the code change + ~5 h training + eval.

---

## Smaller residual cleanups (~hours each)

These are real but lower-impact items that came up during iter-3:

1. **Predictor reduces its own intrinsic diversity.** `Predictor.query_tokens`
   are initialized `std=0.02` and learned. `Predictor.pos_embed` is the same.
   At inference these provide enough diversity machinery that a fully-collapsed
   encoder still produces a "diverse-looking" `z_pred` (this is the E4 failure
   mode in `strategies_log.md`). Zero-initializing `query_tokens` would force
   the predictor to depend more strictly on the encoder's context. Worth one
   ablation: 30-epoch run with `query_tokens` zero-init.

2. **Pixel-level masking instead of token-level mask_token replacement.** V-JEPA
   paper masks pixel patches in the input image *before* the encoder sees them.
   Our `apply_context_mask` replaces encoder *output* tokens. The latter is
   easier to implement but provides less asymmetry pressure: the encoder still
   processes a full image, just one of its output tokens gets replaced.
   Switching to input-pixel masking would require a `--mask-pixels` mode in
   `dataset.py` or a new pre-processor.

3. **Pixel baseline numerical fragility.** Even after sigmoid→clamp and
   `lr_peak=2e-5`, the pixel decoder still NaN'd at ep60 on one of the iter-3
   runs (`ckpt_best.pt` was good, NaN-abort caught it cleanly). The decoder
   architecture itself is the suspect — three transposed-conv blocks with no
   bottleneck normalisation between them. Adding GroupNorm after each transposed
   conv (currently only the post-GELU is normed) or using BCE-with-logits in
   place of MSE-after-clamp would likely make it fully stable. Low priority
   because the baseline already converged.

4. **`evaluate.py` linear probe is memory-heavy.** Currently capped at 2000
   train sequences to fit in 32 GB RAM. If the dataset grows or the latent dim
   grows (Option B), the cap may need to shrink further. Switching to
   incremental ridge or to `sklearn.linear_model.SGDRegressor` would lift the
   memory ceiling at a small accuracy cost.

5. **Stress dataset (15 balls) was never trained.** PLAN §3.4 specifies a
   `data/stress.npz` generalization probe. Iter-3 only trained on `default.npz`.
   A 30-epoch run on stress (or even loading stress data and probing a
   default-trained model on it) would demonstrate generalization claims.

---

## What we explicitly are NOT going to do

- Adjust `var_loss_weight` or `cov_loss_weight` further. Iter-3 ablation
  showed these are not the bottleneck; weight-tweaking is no longer
  productive.
- Switch to `smooth_l1` loss. Implemented but unused; expected gain is small.
- Add encoder spatial positional embeddings. E_arch in iter-3 ablation showed
  no rank improvement and a new NaN stability issue.

---

## How to pick between B, C, and ship-as-is

Order of operations if there's appetite for one more iteration:
1. **Option B first** — single yaml change, ~5 h compute. If V-JEPA pos R²
   climbs to ≥ 0.70 with d=256, ship that as V2 and stop.
2. **Option C only if B is insufficient** — more code, but if the rank
   ceiling persists despite more dim, the projector is the right architectural
   answer.

If neither option is attempted, V1 already tells an honest, technically
interesting story: the iter-3 investigation in `strategies_log.md` and the
documented gap between the original V-JEPA narrative and what 10K-sequence
budget actually delivers is the artifact.
