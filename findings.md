# Mini V-JEPA — Cumulative Findings

**Period covered:** 2026-05-17 → 2026-05-19 (extended with iter-3 investigation 2026-05-19)
**Project state:** All infrastructure built and reviewed; training has failed twice; iter-3 ablation investigation (2026-05-19) has identified a four-knob fix that prevents collapse on default.npz. Full 150-epoch validation run not yet performed.

This file captures everything learned across the multi-agent build, the independent review passes, and the two failed training attempts. It complements `current_problem.md` (which focuses on the latest failure) by recording the broader project picture: what works, what's known-good, what's not, and which lessons should outlive any single fix attempt.

For the iter-3 ablation that diagnosed the collapse mechanism and identified the fix, see [strategies_log.md](strategies_log.md).

---

## 1. Project structure (built and verified)

### Modules (~3000 LOC, 36 passing tests)

| Layer | Files | Status |
|---|---|---|
| Physics simulator | `simulation/physics.py`, `simulation/renderer.py`, `simulation/generator.py` | **Working.** Energy/momentum/reflection verified. Determinism byte-perfect. |
| Data pipeline | `mini_vjepa/dataset.py`, `scripts/generate_data.py`, `configs/*.yaml` | **Working.** NPZ format `(N,32,64,64,3) uint8` + positions/velocities `(N,32,n_balls,2) float32`. 10000-seq default dataset (~58s gen, 123 MB on disk). |
| Model architecture | `mini_vjepa/{encoder,predictor,ema,losses,masking,vjepa}.py` | **Built.** Shapes correct. EMA frozen. Stop-grad verified. Param count 920K trainable, 1.39M total. |
| Training loop | `scripts/train.py` | **Built.** Safety nets work (see §5). Training itself collapses (see §6). |
| Pixel baseline | `baselines/pixel_predictor.py` | **Built.** Reuses ContextEncoder + Predictor verbatim. Decoder NaN's during training (see §6). |
| Evaluation | `scripts/evaluate.py`, `notebooks/evaluation.ipynb` | **Built and tested.** All five PNG figures + `metrics.json` produced from any checkpoint. Pipeline correct; cannot demonstrate JEPA win without a trained JEPA model. |
| Docs | `README.md`, `DESIGN.md`, `assets/architecture.png`, `assets/simulation.gif` | **Done** but with placeholder numbers (no trained model). |

### Test coverage

`tests/`:
- `test_physics.py` — 4 tests (energy, momentum, reflection, snap-to-zero)
- `test_dataset.py` — 17 tests (NPZ contract, shapes, dtypes, indexing, determinism)
- `test_evaluation.py` — 14 tests (probe leakage, parity, latent-space copy-last, determinism)
- `test_training_smoke.py` — 1 test (2-epoch debug run completes)

**36/36 passing** as of 2026-05-19. No CI configured.

---

## 2. Known-good properties (verified by independent probes)

These have been confirmed by multiple independent reviewer agents and direct probes; they are stable parts of the system.

### Physics
- Energy drift 0.00% over 1000 frictionless ticks
- Reflection law exact at 120 Hz (26.57° in/out)
- Friction isotropic across 7 angles
- Determinism: sha1-identical NPZ across runs with same seed
- No initial-frame ball overlaps (all 5 strategies, 3 seeds × 3 n_balls)
- Runtime ball-pair overlap < 1e-3 (~3% of radius) thanks to `substeps=24`

### Data pipeline
- Contract matches PLAN §3.4 exactly
- Indexing alignment: `positions[t]` and `velocities[t]` precede `world.step` → no temporal drift
- Off-by-one clean: `t_start ∈ [0,16]`, `Δt ∈ [1,12]`, max index `31 < 32`
- RGB convention consistent end-to-end (felt-band pixels show `G > R, G > B`)
- Compression 70× (10000-seq default = 123 MB on disk, not the PLAN's 2 GB estimate)

### Model architecture
- All shape contracts hold for B∈{1,2,8}: encoder `(B,3,64,64) → (B,64,128)`; encode_context `(B,4,...) → (B,256,128)`; predict `(B,64,128)`
- GroupNorm only — no BatchNorm, no Dropout, float32 throughout
- Target encoder stop-gradient: 22 params, all `requires_grad=False`; after `loss.backward()` all target grads are `None`
- EMA update math correct: τ=0 → target=online; τ=1 → target unchanged; in-place under `no_grad`
- Δt embedding wired (predictions differ for different Δt values)
- Variance regularization gradient flows to both encoder and predictor
- Covariance regularization (added Iteration 2) gradient also flows to both
- MPS forward+backward works with `PYTORCH_ENABLE_MPS_FALLBACK=1`
- Pixel baseline reuses (does not fork) `ContextEncoder` and `Predictor` — verified by class identity, param counts match exactly (encoder 471,168; predictor 449,152)

### Training loop (safety side)
- Optimizer partition exhaustive (every trainable param in exactly one group)
- LR schedule mathematically correct: linear warmup → cosine to `lr_final` at last step
- EMA τ schedule monotonic non-decreasing in [0.996, 1.0]
- Loop ordering correct: `backward → clip_grad_norm → step → scheduler.step → ema_update`
- Δt sampled in [1, 12] (never 0)
- MPS env var set before `import torch`
- `torch.mps.empty_cache()` every 20 epochs
- After fixes: collapse-stop fires at exactly `streak == patience`, no off-by-one
- After fixes: NaN abort exits with code 2 on first non-finite loss
- After fixes: per-batch monitor compute eliminated (verified: `effective_rank` called once per epoch, not 156×)

### Evaluation
- Sequence-level train/test split — no frame-level leakage (sklearn random-target probe gives R² < 0.2)
- Linear probe uses sklearn `Ridge(alpha=1.0)`, honest `r2_score`
- Copy-last baseline computed in latent space (cosine = 1.0 on duplicated frames)
- Pixel-baseline comparison uses *reused* encoder weights, not retrained
- t-SNE / PCA use the target encoder (EMA) representation
- Determinism: same seed → bit-exact equal `overall_r2` across runs
- All 5 expected PNGs + `metrics.json` produced reliably

---

## 3. Multi-agent independent-review track record

Across the project we ran two rounds of independent reviewers (5 specialized + 2 confirmation passes). Track record:

### Caught (high value)

1. **Per-batch monitor recomputation in `compute_loss`** (~30% perf waste on MPS) — caught by both Iteration-2 reviewers.
2. **`ckpt_best.pt` selecting random init** (because `best_eff_rank=-inf` and eff_rank monotonically decreased after epoch 0) — caught by Iteration-1 reviewers and root-caused via the trajectory data.
3. **MPS fallback env var set after `import torch` in baseline** — caught by Iteration-1 training reviewer.
4. **Pixel baseline NaN loop wasting 110 epochs** — caught by Iteration-1 training reviewer.
5. **Sequence-level split is contiguous, not shuffled** — caught by eval reviewer.
6. **Notebook hardcoded timestamped run paths** — caught by eval reviewer.
7. **Eigvalsh exception class** — caught by Iteration-1 model reviewer (correctly identified `torch._C._LinAlgError`).
8. **Stale imports after dropping monitor calls** — caught by Iteration-2 model reviewer.
9. **Param-budget gap** (920K trainable vs PLAN's 5–8M) — caught by Iteration-1 model reviewer, attributed to PLAN inconsistency.
10. **5-epoch `avg_std<0.05` warning is the wrong sentinel for dimensional collapse** — caught after the first failure when the alarm never fired despite eff_rank=1.

### Missed (and why it matters)

1. **`var_loss_weight=1.0` is far too low for this setup** — Iteration-1 reviewers did not flag this. The model collapsed on the next run. (Hindsight: the PLAN itself specified weight=1.0; reviewers were checking spec compliance rather than dynamics.)
2. **`cov_loss_weight=1.0` is also too low for this setup** — Iteration-2 reviewers explicitly defended this with a numerical argument that turned out to be 10× off. The model collapsed again on the next run.
3. **Sigmoid in pixel decoder gradient saturation** — never flagged. Pixel baseline NaN'd in both runs.
4. **Initial-placement overlap up to 5.25e-2 of diameter** — caught only after the first physics-review iteration uncovered it; the initial review missed it. (Eventually fixed.)
5. **Visual ball radius 2 px vs PLAN's "3-4 px"** — same; second-iteration discovery.

### Reviewer methodology pattern

The reviewers reliably catch:
- Code-level bugs (shape mismatches, exception classes, missing keys, ordering errors)
- Spec compliance (contracts honored, hyperparameters match PLAN)
- Static analysis (stale imports, dead code, suspect exception handling)
- Property tests (stop-gradient, idempotence, edge cases of τ schedules)

The reviewers unreliably predict:
- **Training dynamics** (whether a loss term will dominate in practice)
- **Numerical magnitudes** at runtime (their hypotheticals were off by orders of magnitude in both rounds)
- **Generative model stability** (decoder sigmoid pathology never raised)

**Implication:** the review-gate-per-feature model from CLAUDE.md is excellent at preventing bugs but does not validate that training will converge. Validation of dynamics requires **actually running training** for tens of epochs and watching the metrics — this is now built into the proposed Iteration-3 workflow.

---

## 4. Configuration baseline (current state)

### `configs/default.yaml` (most relevant fields)

```yaml
data:
  n_balls: 9
  n_sequences: 10000
  sequence_length: 32
  physics_fps: 120
  init_strategies: {break:0.30, midgame_cluster:0.20, midgame_spread:0.20, two_ball:0.15, random_velocities:0.15}

model:
  latent_dim: 128
  spatial_tokens: 64
  context_frames: 4
  predictor_layers: 3
  predictor_heads: 4
  predictor_ffn: 256
  delta_t_min: 1
  delta_t_max: 12

training:
  batch_size: 64
  epochs: 150
  lr_peak: 1.5e-4
  lr_final: 1.0e-5
  warmup_epochs: 10
  weight_decay_encoder: 0.05
  weight_decay_predictor: 0.0
  grad_clip: 1.0
  ema_tau_start: 0.996
  ema_tau_end: 1.0
  var_loss_weight: 25.0       # was 1.0 originally
  cov_loss_weight: 1.0        # added Iteration 2
  collapse_eff_rank_threshold: 5.0
  collapse_patience_epochs: 8
  checkpoint_every: 25
  max_checkpoints: 4
  seed: 42

baseline_overrides:           # added Iteration 2
  lr_peak: 5.0e-5             # was 1.5e-4 (caused decoder explosion)
  grad_clip: 1.0
  grad_value_clip: 0.5
```

### Dataset on disk

- `data/default.npz` — 123 MB, 10000 sequences, 9 balls, seed 42 (already generated, deterministic)
- `data/simple.npz` — 363 KB, 64 sequences, 3 balls (for debug)
- `data/sample/sample.npz` — 91 KB, 16 sequences, 3 balls (in git, for tests)

---

## 5. Safety nets (all verified working in real run)

These fired correctly during Iteration 2 and saved compute. They are not the bottleneck.

| Safety net | Where | Verified by |
|---|---|---|
| Collapse-stop on `eff_rank < threshold` for `patience` epochs | `scripts/train.py` | Iteration-2 real run aborted at epoch 28 exactly when streak hit 8 |
| NaN-abort on non-finite loss | `baselines/pixel_predictor.py` | Iteration-2 real run aborted at epoch 60 batch 2 (exit code 2) |
| `eigvalsh` try/except returning NaN | `mini_vjepa/losses.py` | Probed with degenerate cov matrices; no crash |
| `ckpt_best.pt` gated by warmup AND `eff_rank ≥ threshold` AND not-NaN | `scripts/train.py` | No best-ckpt written during Iteration 2 (correct: model collapsed) |
| Per-batch monitor compute eliminated | `mini_vjepa/vjepa.py` | Instrumented counter: 2 calls over 2 debug epochs (not 312) |
| ETA logging per epoch | `scripts/train.py`, `baselines/pixel_predictor.py` | Visible in logs from epoch 0 |

---

## 6. What is NOT working (the actual remaining problems)

### V-JEPA: dimensional collapse

**Symptom:** `effective_rank` falls from 15 → 1.6 between epochs 19 and 22 of training, then sticks at 1.6–2.2 until aborted.

**Loss decomposition at the moment of collapse** (epoch 22):
- mse = 1.09 (model fits target well — by encoding everything to the same point)
- 25·var = 18.54 (variance reg active but local)
- 1·cov = 2.17 (cov reg fires but small in absolute terms)
- avg_std = 0.404 (looks healthy — this is why the original `avg_std<0.05` alarm was useless)
- cos_sim = 0.798 (token-pair similarity high — confirms all dims encode same signal)

**Why the fix didn't fix it:** the cov_reg gradient at `cov_weight=1.0` and our setup's scale (`cov ∈ O(1)`) is dominated by the MSE gradient pulling toward `z_pred = z_target = constant`. VICReg authors got away with `cov=1.0` on ImageNet because their dim was 8192 (cov sum scales O(d²)); on our 128-dim setup the term needs ~25× more weight to balance.

**Trajectory of cov_reg through collapse:**
- epoch 19: cov=1.11 (healthy)
- epoch 22: cov=2.17 (collapsed but cov gradient too small)
- epoch 27: cov=14.21 (cov spiking but too late — model already in collapse basin)

### Pixel baseline: NaN explosion

**Symptom:** Loss decreases healthily from 0.090 → 0.021 over 60 epochs, then one batch produces non-finite loss; abort.

**Improvement over Iteration 1:** 60 healthy epochs vs 40 previously. Lower LR (5e-5 vs 1.5e-4) and value clipping (0.5) helped but did not eliminate.

**Suspected cause:** the final `nn.Sigmoid()` in `PixelDecoder.forward()`. When pre-sigmoid activations drift to extreme values, sigmoid saturates → near-zero gradient for those pixels but huge gradient for un-saturated ones → unstable update → eventually one forward pass produces `inf`.

**Alternatives not yet tried:**
- Replace `Sigmoid` with `torch.clamp(x, 0, 1)` (no saturation pathology)
- Predict logits and use `binary_cross_entropy_with_logits` (no separate sigmoid needed)
- Lower LR further (2e-5 or 1e-5)

---

## 7. Compute log

| Date | Run | Duration | Outcome |
|---|---|---|---|
| 2026-05-18 | Iteration 1 V-JEPA | ~80 min | Collapsed by epoch 19; crashed at epoch 120 (eigvalsh) |
| 2026-05-18 | Iteration 1 pixel | ~105 min | NaN at epoch 40; trained 110 wasted NaN epochs |
| 2026-05-19 | Iteration 2 V-JEPA | ~17 min | Collapsed by epoch 22; auto-aborted at epoch 28 |
| 2026-05-19 | Iteration 2 pixel | ~45 min | NaN at epoch 60; clean abort exit code 2 |

**Total MPS-hours spent:** ~4 hours. **Usable trained models produced:** zero V-JEPA, zero pixel.

---

## 8. Process lessons

1. **Review gates catch code bugs but not training dynamics.** Two rounds of independent reviewers passed; both runs collapsed for reasons no reviewer flagged. Going forward, treat reviewer "GO" as a code-correctness pass, not a training-readiness pass.
2. **A 30-epoch debug run on the full default dataset (~7 min) is the cheapest dynamics probe.** This catches collapse in the regime it actually happens, without the 150-epoch wait. Should be the gate before any full run going forward.
3. **The two safety nets (collapse-stop + NaN-abort) are now load-bearing.** They saved ~75 min on this run by killing both processes early. Keep them; do not weaken them.
4. **PLAN-specified hyperparameters are not magic.** PLAN said `var_loss_weight=1.0`; that was wrong. Reviewers comparing against PLAN cannot catch this. Empirical validation is required for dynamics-sensitive hyperparameters.
5. **`avg_std` alone is the wrong collapse sentinel.** `effective_rank` is necessary (and sufficient) for dimensional collapse. The PLAN's `avg_std<0.05` alarm is preserved but redundant — the eff_rank alarm is the load-bearing one.
6. **Documentation of negative results matters.** `current_problem.md` + this file would have taken 10 minutes to write on day 1 and saved hours of "did we already try this?" loops.

---

## 9. What's left to do (in priority order)

1. **Stop V-JEPA collapsing.** Iteration-3 candidates: bump `cov_loss_weight` to 25.0; drop `ema_tau_start` to 0.99; possibly lower `lr_peak` to 5e-5. Validate on a 30-epoch debug run before committing 150.
2. **Stop pixel baseline NaN'ing.** Replace decoder sigmoid with clamp, or predict logits + BCE.
3. **Re-launch when (1) and (2) are both validated.** Full 150 epochs each, ~3h total.
4. **Run `scripts/evaluate.py` on the real checkpoints.** Produces the 5 PNGs and `metrics.json` for the README.
5. **Update README headline numbers** with real R² values and degradation curve.
6. **Update DESIGN.md** with the "two failed runs and what we learned" narrative — this is more interesting than the original DESIGN content.

Stretch (only after the above lands):
7. PLAN §6.4 ablations (no EMA, no var-reg, larger latent)
8. Stress dataset (15 balls) generalization test
9. CI configuration
