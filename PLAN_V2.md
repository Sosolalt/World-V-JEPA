# Mini V-JEPA — V2 Improvement Plan (revision 2)

> Synthesis of two consultation rounds (10 specialists + 5 reviewers, two iterations).
> Supersedes PLAN_V2.md revision 1 and [`FUTURE_WORK.md`](FUTURE_WORK.md) options B and C.
> Companion: [`PLAN_Mini_V-JEPA.md`](PLAN_Mini_V-JEPA.md) (original), [`strategies_log.md`](strategies_log.md) (iter-3 ablation ledger).

---

## 0. What changed since rev 1

Rev 1's Phase 1 ran today on the V1 checkpoint. The numbers:

| Probe | V-JEPA | Pixel | Δ |
|---|---|---|---|
| Per-frame pos R² — EMA target (V1 headline) | 0.577 | 0.585 | pixel +0.008 (~0.3σ at N=300) |
| Per-frame pos R² — online encoder | 0.555 | 0.585 | pixel +0.030 |
| 4-frame context-window pos R² | 0.423 | 0.386 | V-JEPA +0.037 |
| 4-frame context-window vel R² | **−0.59** | **−1.93** | both broken |
| Rollout MAE V-JEPA predictor (all horizons) | **3.95** | — | 40× worse than copy-last (0.09), 400× worse than identity (0.009) |
| Held-out regime (`random_velocities`) pos | 0.328 | — | -14pp vs in-dist |

**Two interpretations were debated; consultation resolved both:**

1. **EMA staleness was REFUTED** — target slightly beats online, both ≈ 0.57. Consistent with Polyak averaging (S2). The V-JEPA paper itself probes the *online* encoder, not the EMA target — V1 had the wrong headline.
2. **Negative R² on context-window probe is a METHODOLOGY BUG, not an encoder failure** (S1, S5, S7). Three compounding causes: under-regularized Ridge at 32K features (`α=1.0 + lsqr` on a rank-12 representation produces a `LinAlgWarning: rcond=8.5e-8`), un-standardized features with native variance heterogeneity, and concat-of-tokens probing for velocity when V-JEPA's encoder is **not designed to encode velocity per-token** — velocity lives in the *temporal derivative* `pool_S(z_t) − pool_S(z_{t-1})` (S7).

**Three NEW failure modes surfaced in review that block the plan if unfixed:**

- **Inference-time mask leak.** `scripts/evaluate.py:546, 669` call `model.apply_context_mask(ctx_tokens, mask_ratio)` where `mask_ratio` is read from `cfg.training.mask_ratio` = 0.75. Every cos-sim and rollout number we have reported was measured with the training-time mask still active. The predictor's claimed uselessness (cos-sim 0.075) is partially the predictor + partially this artifact. Must be patched **before** Phase 1-bis re-runs.
- **`ctx_effective_rank` is computed on ONE batch per epoch** (`scripts/train.py:341-364` uses the last batch only). The rank=12 diagnosis driving all architecture work rests on a single-batch measurement (Reviewer C).
- **Pixel baseline shares the V-JEPA encoder architecture** via `ContextEncoder` reuse in `baselines/pixel_predictor.py:84`. The 0.585 vs 0.577 comparison is two encoders of identical capacity trained on different objectives. Adding a **frozen-V1-encoder** pixel-decoder baseline gives an independent diagnostic (Reviewer C).

---

## 1. Five-round consultation findings (compressed)

### Round 1 — 10 specialists (literature pass)

| # | Specialist | Headline finding |
|---|---|---|
| S1 | ctx-window probe failure | Negative R² is textbook methodology bug at low-rank features. Fix: mean-pool + StandardScaler + RidgeCV(`logspace(-3, 6, 19)`) + sequence-level bootstrap. Citations: Kobak 2020 JMLR, Alain & Bengio 2016, Belinkov 2022. |
| S2 | EMA vs online | V-JEPA/I-JEPA/DINOv2-3/BYOL/MoCo all probe the **online** encoder. EMA-vs-online +0.022 Δ is Polyak averaging, not staleness. Attentive probe adds +4-17pp over Ridge. |
| S3 | predictor as forecaster | Cos-sim 0.075 + rollout 40× worse than copy-last is the **textbook conditional-mean collapse** for MSE-trained continuous predictors at high mask ratio. Existence proof at our scale: **LeWorldModel (15M params, single GPU)**. Highest-leverage single change: **pixel-patch tubelet masking BEFORE encoder** (cross-paper consensus). |
| S4 | encoder rank lift | The 1×1 channel squeeze (Conv2d(256,128,1)) IS the mathematical rank ceiling (Geometry of Projection Heads Theorem 5.1). **MEC** (`-logdet(I + μZᵀZ/n)`, 15 LOC drop-in for VICReg cov term) expected +20-40 rank lift. KoLeo as free safety net. SIGReg M=256 not M=1024 at our 10K-sample scale. |
| S5 | linear probing pitfalls | At N=300 test sequences the V-JEPA-vs-Pixel 0.008 gap is ~0.3σ — **statistically indistinguishable**. Need N≥2500 or sequence-level paired bootstrap. Add attentive probe + k-NN cross-check. |
| S6 | physics evals beyond probes | Add **OCP** (binary "will balls A & B collide in next K frames", ~100 LOC, can't be faked by probes) + **VoE counterfactuals** (we have pymunk!) + autoregressive rollout MSE. Meta's "Intuitive Physics Emerges from V-JEPA" uses VoE exclusively, not probes. |
| S7 | velocity decoding | V-JEPA encoder NOT designed to encode velocity per-token. Velocity = `W · (pool_S(z_t) − pool_S(z_{t-1}))`. **VICReg-on-encoder actively suppresses high-frequency dims** where velocity would live. 3-line fix to test. |
| S8 | GitHub priors | No sub-1M JEPA exists publicly. Strong precedents: **lucas-maes/le-wm** (SIGReg-only, 15M, planning works), **facebookresearch/jepa-wms** (Dec 2025, AdaLN-Zero + multi-step rollout), **keon/jepa** (toy V-JEPA: canonical multi-block input masking + L1 + NO VICReg). Single recipe to copy: canonical V-JEPA multi-block input masking. |
| S9 | masking deep dive | Our token-output `mask_token` replacement is the WRONG asymmetry direction — V-JEPA / I-JEPA / MAE mask BEFORE the encoder. Recipe: 2×8×8 tubelets, multi-block (4 short @15% + 1 long @50%), mask token in PREDICTOR not encoder. |
| S10 | loss functions | **L1 (V-JEPA 2) > Smooth L1 ≈ MSE** at current setup — PyTorch Smooth L1 β=1.0 makes it ≡ MSE for our residual range. Distance-weight predicted tokens by 1/(1+Δt) (V-JEPA 2.1 lesson). |

### Round 2 — 3 reviewers (consolidation)

- **A (plan integrity)**: GO with caveats — atomic update of `evaluate.py:546,669`, bump max-eval-sequences to 2000, velocity gate must use difference-probe.
- **B (pragmatist)**: GO with caveats — drop MEC/SIGReg/dim 256, Phase 5 jumps ahead of 2/3/4, hard cap ≤8 full runs, OCP AUC ≥ 0.65 is load-bearing ship gate.
- **C (skeptic)**: GO with caveats, NO-GO triggers — Phase 0 sanity, batch-level ctx_eff_rank, frozen-V1-encoder pixel baseline. Plus 6 things 10 specialists missed.

### Round 3 — 2 final reviewers (unified plan)

- **D (pragmatic validation)**: GO with caveats — eval-time `mask_ratio` must be made explicit (CLI flag), CLI default `max-eval-sequences` is 800 (mismatch with plan), specify `temporal_pos` interaction with mask token.
- **E (adversarial final)**: GO with caveats — Failure #1 (inference-time mask leak), Failure #2 (`spatial_tokens=64` hardcoded), Failure #3 (`RLIMIT_AS` silently rejected today).

**All 5 reviewers across 2 iterations converged on GO with caveats. No NO-GO.** Caveats consolidated into §3 below.

---

## 2. Diagnosis (consensus)

V1's "loss to pixel baseline" was a measurement artifact, not a capability gap:
- The 0.008 R² gap is ~0.3σ at N=300 (S5 + Reviewer D).
- The EMA-target probe used the wrong encoder per V-JEPA paper convention (S2).
- The pixel baseline shares the V-JEPA encoder architecture (Reviewer C).
- The velocity probe is physically ill-posed at per-frame (every reviewer).
- The ctx-window velocity probe was numerically broken (S1, S5).

V1's predictor IS broken at inference (cos-sim 0.075, rollout 40× worse than copy-last). This is the textbook conditional-mean collapse for MSE-trained continuous predictors at 75% mask ratio (S3). **But part of the badness is the eval-time mask leak** (Failure #1) — we don't know the true number yet.

The encoder rank ceiling (`ctx_eff_rank ≈ 12`) has FOUR candidate root causes that the plan isolates:
1. The 1×1 channel squeeze in `encoder.py:37` (S4)
2. VICReg's geometric weakness at d=128 — satisfiable at low rank (S4, LeJEPA)
3. Token-output masking provides no encoder-side asymmetry pressure (S9)
4. The `spatial_tokens=64` hardcode — per-frame ceiling is `min(64, 128) = 64`, of which ctx_eff_rank=12 is at 19% utilization, not 9% (Reviewer C, E)

---

## 3. Plan — sequenced, with mandatory gates

**Total budget cap: ≤ 8 full 150-epoch runs across the entire month** (Reviewer B). Each "full run" is ~2h on M3.

### Phase 0 — Reliability & sanity (Day 1, no retraining)

Addresses all three Critical Failure Modes plus Reviewer C's NO-GO triggers.

| Change | File | Why |
|---|---|---|
| **DELETE / guard `apply_context_mask` calls at inference** (default `mask_ratio=0.0` in `degradation_curve`, `rollout_position_error`); add `--eval-mask-ratio` CLI flag separate from `cfg.training.mask_ratio` | [`scripts/evaluate.py:546, 669`](scripts/evaluate.py) | **FAILURE #1 fix.** Every V1 cos-sim and rollout number was measured with 75% mask still active. Until this is patched, Phase 1-bis and Phase 8a will reproduce the artifact. |
| Compute `ctx_effective_rank` over **≥10 batches per epoch**, log mean ± std | [`scripts/train.py:341-364`](scripts/train.py) | Reviewer C — the rank=12 diagnosis rests on a single batch. |
| Add **frozen-V1-encoder** pixel baseline variant: load V1 `state_dict` into `ContextEncoder`, freeze, train only decoder + predictor | [`baselines/pixel_predictor.py:84`](baselines/pixel_predictor.py), add `--freeze-encoder` flag | Reviewer C — breaks the "V-JEPA loses to its own architecture" self-reference. Keep the from-scratch baseline too; report BOTH. |
| Bump `RIDGE_CV_ALPHAS = np.logspace(-3, 6, 19)` (was `(-3, 3, 13)`) | [`scripts/evaluate.py:362`](scripts/evaluate.py) | S1 + Reviewer C — `LinAlgWarning rcond=8.5e-8` shows current ceiling 1e3 is silently truncated. |
| Fix `--max-eval-sequences` CLI default 800 → **2000** | [`scripts/evaluate.py:1248`](scripts/evaluate.py) | Reviewer A + D — the CLI default contradicts the function-arg default. |
| **Verify RAM budget at 2000 sequences** via smoke run before committing | smoke test | **FAILURE #3** — `RLIMIT_AS=15GB` was silently rejected today (`current limit exceeds maximum limit`). At 2000 sequences peak may be 16+ GB. |
| Skip raw-frame intrinsic-dim estimate (low value per Reviewer D); proceed straight to Phase 1-bis | — | Reviewer D — Phase 1-bis attentive-probe R² is a better signal than TwoNN on raw pixels. |

### Phase 1-bis — Probe rebuild (Day 2, no retraining)

Addresses S1, S2, S5, S7 + Reviewer A's velocity-difference gate.

| Change | File | Why |
|---|---|---|
| Switch headline encoder: EMA target → **online** (per V-JEPA / I-JEPA / DINOv2-3 / BYOL / MoCo convention). EMA → ablation row. | [`scripts/evaluate.py`](scripts/evaluate.py) `encode_all_frames_online` | S2 |
| **Mean-pool tokens + StandardScaler** before Ridge (drop the concat-32K-features pathology); use `Pipeline(StandardScaler, RidgeCV)` | [`scripts/evaluate.py:linear_probe`](scripts/evaluate.py) | S1, S5 |
| **Velocity-via-difference probe**: `Δz = pool_S(z_t) − pool_S(z_{t-1})`, then Ridge to velocity at frame t-1 | [`scripts/evaluate.py`](scripts/evaluate.py) new function | **S7 — the load-bearing methodology fix.** Concat-Ridge cannot recover velocity because V-JEPA encoder doesn't encode it per-token; finite differences match the physical definition. |
| **Attentive probe** (1 cross-attention head + linear, trained AdamW ~100 steps on train split) as the new headline number | [`scripts/evaluate.py`](scripts/evaluate.py) new function | S2 + S5 — V-JEPA paper standard, +4-17pp over Ridge. |
| **Sequence-level paired-bootstrap CI** (200 resamples) on V-JEPA vs Pixel R² gap; resample sequences, not frames | [`scripts/evaluate.py`](scripts/evaluate.py) new function | S5 — at N=300 the 0.008 gap is ~0.3σ; need bootstrap to make any claim about the sign. |

### Phase 8a — OCP + autoregressive rollout (Day 3, no retraining)

S6 + Reviewer B's load-bearing ship requirement.

| Change | File | Why |
|---|---|---|
| **OCP probe**: binary classifier "will balls i & j come within `2·radius + ε` in next K frames" on the frozen V-JEPA encoder. Logistic regression, ~100 LOC. Report AUC. | new `scripts/evaluate.py` function | S6, Physion-style. Cannot be faked by position probing — requires forward physical reasoning. |
| **Autoregressive rollout MSE**: predict `z_{t+1}`, feed back as context, predict `z_{t+2}`, etc. Report per-horizon latent MSE and decoded-position MAE | extend `degradation_curve` | S6 — catches the "single-shot predict cheats" failure mode that one-shot Δt-jump masks. |
| **VoE counterfactuals deferred to Phase 8b** (needs pymunk plumbing in `simulation/`, ~250 LOC; not on the critical path for the Day-3 decision gate) | `simulation/counterfactual.py` (new) | S6 — keep but defer. |

### DECISION GATE (end of Day 3)

If with the corrected probes:
- V-JEPA **online-encoder attentive-probe** pos R² ≥ 0.62 with sequence-bootstrap 95% CI excluding the pixel baseline, **AND**
- V-JEPA velocity-difference probe R² ≥ 0.30, **AND**
- V-JEPA OCP AUC ≥ 0.65 (pixel baseline can't do OCP well by construction — this is where the JEPA advantage should show)

→ **SHIP V1** with the new evaluation narrative: *"Evaluation pitfalls of joint-embedding world models at small scale."* The honest scientific story is at least as good as a marginal R² lift.

Else → proceed to Phase 2.

### Phase 2 — Canonical V-JEPA pixel-patch tubelet masking + L1 (Day 4-7, 1 full retrain)

Reviewer B's "biggest single training change", confirmed by S3, S8, S9.

| Change | File | Why |
|---|---|---|
| **DELETE `apply_context_mask` from `vjepa.py`**, move mask token machinery into `Predictor` | [`mini_vjepa/vjepa.py:54-66`](mini_vjepa/vjepa.py), [`mini_vjepa/predictor.py`](mini_vjepa/predictor.py) | S9 — token-output masking is the wrong asymmetry direction. |
| **Pixel-patch tubelet masking BEFORE encoder**: 2×8×8 tubelets on the (T=4, H=64, W=64) context. Multi-block: 4 short blocks @ 15% spatial coverage + 1 long block @ 50%, aspect U(0.75, 1.5). Mask token + temporal_pos inserted at predictor input. | new `mini_vjepa/masking.py`, [`mini_vjepa/dataset.py`](mini_vjepa/dataset.py) extension | S8 + S9 cross-paper consensus. |
| **Switch loss MSE → true L1** (`torch.abs(...).mean()`, NOT Smooth L1 which is ≡ MSE at β=1.0 for our residuals) | [`mini_vjepa/losses.py`](mini_vjepa/losses.py), [`mini_vjepa/vjepa.py`](mini_vjepa/vjepa.py) | S3 + S10 — V-JEPA 2 standard. |
| **Loss on masked positions only** (not all output tokens) | [`mini_vjepa/vjepa.py:compute_loss`](mini_vjepa/vjepa.py) | S9 — V-JEPA / I-JEPA / V-JEPA 2.1 canonical. |
| **Atomic update**: `evaluate.py:546, 669` must be patched in the SAME commit — they currently call the deleted method | [`scripts/evaluate.py`](scripts/evaluate.py) | Reviewer A — non-negotiable atomicity. |
| **Specify `temporal_pos` interaction with mask token**: masked positions receive `temporal_pos + mask_token`. Document decision. | [`mini_vjepa/predictor.py`](mini_vjepa/predictor.py) | Reviewer D + E. |
| Single-knob discipline: preserve **all iter-3 anti-collapse machinery** (`reg_target=both`, `cov_weight=25`, `ema_τ_start=0.99`, `lr_peak=5e-5`, `grad_value_clip=0.5`, dual collapse-stop, all `ctx_*` monitors, NaN-abort) | [`configs/default.yaml`](configs/default.yaml) | Iter-3 `strategies_log.md` §3 + Reviewer 1's NO-GO trigger from rev 1. |

**30-epoch gate** (`--epochs 30 --lr-schedule-epochs 150`):
- `ctx_effective_rank ≥ 8` sustained ep10-29 (matches iter-3 v3 floor; need ≥10-batch measurement from Phase 0)
- `z_pred eff_rank ≥ 30` by ep20
- **Predictor cos-sim ≥ 0.25 by ep29** (4× lift from 0.075 — the Reviewer-B-relaxed gate; 0.40 had no precedent)
- **Per-mask-ratio cos-sim monitor**: at ep29, sweep eval mask_ratio ∈ {0.0, 0.25, 0.5, 0.75, 0.9}. Cos-sim should DEGRADE monotonically as mask_ratio rises. Flat curve = predictor memorized one mask distribution. (Reviewer C)
- No NaN, no collapse-stop fire

### Phase 3 — Multi-step rollout predictor training (Day 8-12, 1 full retrain)

S3, S8 — required if predictor is to be useful at inference. Only attempted if Phase 2 lifts cos-sim past 0.25.

| Change | File |
|---|---|
| Append predicted `z_pred(t+1)` to context window, predict `z_pred(t+2)`, average loss across 2-3 steps | [`scripts/train.py`](scripts/train.py) |
| **Gate**: predictor cos-sim ≥ 0.40 at full 150-ep run AND rollout MAE at horizon=10 must beat `identity_mae` baseline | Reviewer C NO-GO trigger |

### Phase 4 — Architecture lift, single-knob (Day 13-17, 1 full retrain)

ONLY the wider CNN trunk + 2-layer projector (per S4 + Reviewer B's single-knob rule). Spatial pos-embed and AdaLN-Zero Δt conditioning deferred to separate week-3 single-knob runs.

| Change | File |
|---|---|
| `conv3: Conv2d(128, 256, …)` → `Conv2d(128, 384, …)`; replace `proj = Conv2d(256, 128, 1)` with two-layer block `Conv2d(384, 384, 1) → GN → GELU → Conv2d(384, latent_dim, 1)` | [`mini_vjepa/encoder.py`](mini_vjepa/encoder.py) |
| **Acknowledge `spatial_tokens=64` hardcode** (`encoder.py:26`) — widening channels alone may not lift `ctx_eff_rank` past 64 (the per-frame mathematical ceiling). If Phase 4 underperforms, the spatial bottleneck is the next target. (Reviewer E) | [`mini_vjepa/encoder.py:26`](mini_vjepa/encoder.py) doc |
| **Re-validate** Phase 2's per-mask-ratio + cos-sim gate on the new (wider) encoder before declaring Phase 2 still passes (Reviewer C) | runbook |

**Gate**: `ctx_effective_rank ≥ 20` ep20-29.

### CONTINGENT FALLBACK — `latent_dim 128→256`

If Phase 2's `ctx_effective_rank` caps at ≤15 (same V1 ceiling) WITH the new pixel-patch masking, run `latent_dim 128 → 256` as a single-knob diagnostic before Phase 3. (Reviewer A — addresses ceiling-isolation concern.) Counts against the 8-run budget cap.

---

## 4. What we explicitly are NOT doing

- **NOT removing iter-3 anti-collapse machinery.** Three iterations of evidence prove every "we don't need that anymore" has been wrong. Reviewer 1's NO-GO trigger from rev 1.
- **NOT shipping the MEC swap.** Reviewer B: research curiosity, +20-40 rank claim from d=2048 ImageNet may not transfer to d=128 small-data. Reviewer C: logdet on near-singular Gram matrix at MPS float32 likely NaNs.
- **NOT shipping the SIGReg replacement of VICReg.** Reviewer B: high-uncertainty intervention. LeJEPA M=1024 is overkill at our 10K samples; if revived, use M=256.
- **NOT shipping the 4-knob arch lift in one run.** Violates iter-3 single-knob discipline (Reviewers A + B + C + E). Wider trunk + projector counts as one coupled change because the projector requires the wider trunk to be meaningful.
- **NOT changing eval methodology mid-comparison.** Phase 2 deletes `apply_context_mask` — Phase 1-bis numbers and Phase 2 numbers become non-comparable, so V1 baselines must be re-measured against the patched eval before Phase 2 ships.
- **NOT trusting position-probe headline numbers** until Phase 0 + Phase 1-bis + Phase 8a land. The current V1 vs Pixel comparison is at ~0.3σ — statistically indistinguishable.

---

## 5. Caveats (consolidated from all 5 reviewers — all must land before Day 4)

1. **Phase 0 atomicity**: ship all five Phase-0 items together. Skipping any silently breaks the downstream measurements (Reviewers C + D + E).
2. **Eval-time mask must be explicit**, not inherited from `cfg.training.mask_ratio`. Add `--eval-mask-ratio` CLI flag (default 0.0) (Reviewer D).
3. **Per-mask-ratio cos-sim monitor** at Phase 2's 30-ep gate must run at INFERENCE mask ratios {0.0, 0.25, 0.5, 0.75, 0.9}, not just at the training ratio (Reviewer C + E).
4. **Frozen-V1-encoder pixel baseline does NOT replace the from-scratch one** — report both. From-scratch is the V-JEPA-paper-standard comparison; frozen is the diagnostic for "is this a fair comparison" (Reviewer D).
5. **`spatial_tokens=64` hardcode is the un-addressed bottleneck** for Phase 4. Widening channels alone may not lift `ctx_eff_rank` past 64. If Phase 4 underperforms, spatial token count is next (Reviewer E).
6. **≤8 full 150-epoch runs across the entire month**, week-3 deferred knobs count against the cap (Reviewer B).
7. **CLAUDE.md feature-review gate** triggers explicitly at each phase boundary:
   - Phase 0 + Phase 1-bis + Phase 8a → `evaluation-test-expert` reviewer
   - Phase 2 → `training-loop-test-expert` AND `model-architecture-test-expert` reviewers (touches both)
   - Phase 3 → `training-loop-test-expert`
   - Phase 4 → `model-architecture-test-expert`
   Surface verdicts verbatim. Never silently override a NO-GO.
8. **RAM smoke-test at N=2000** before committing Phase 1-bis. The `--ram-limit-gb 15` flag was silently rejected today (`current limit exceeds maximum limit`); at 2000 sequences peak may be 16+ GB. May require stride-subsample inside `flatten_for_probe`, not just in `linear_probe` (Reviewer E).
9. **Phase 4 re-validation**: if Phase 4 changes the encoder, re-run Phase 2's per-mask-ratio + cos-sim gate on the new encoder before declaring Phase 2 still passes (Reviewer C invalidation concern).

---

## 6. Hackathon timeline (deadline 2026-06-19/20, ~27 days from today)

| Day | Phase | Deliverable |
|---|---|---|
| 1 | Phase 0 | Reliability fixes shipped; V1 re-measured with corrected eval |
| 2 | Phase 1-bis | Online encoder + attentive probe + difference-velocity + bootstrap CI |
| 3 | Phase 8a + decision gate | OCP + autoregressive rollout; ship-V1 vs proceed decision |
| 4-7 | Phase 2 | Pixel-patch tubelet masking + L1, code + 30-ep gate + full retrain |
| 8-12 | Phase 3 (if Phase 2 cleared cos-sim ≥ 0.25) | Multi-step rollout training |
| 13-17 | Phase 4 (if Phase 3 cleared) | Wider trunk + 2-layer projector |
| 18-22 | Week 3 single-knob (KoLeo, spatial pos-embed, AdaLN-Δt) IF budget allows | Optional |
| 23-27 | Writeup, figures, README rewrite, LinkedIn post | Polish |

**Minimum viable submission**: Phase 0 + Phase 1-bis + Phase 8a + V1 re-evaluated (3 days). Defensible "evaluation pitfalls of small-scale JEPA" narrative.
**Target submission**: + Phase 2 (canonical V-JEPA with corrected predictor). 1 week.
**Stretch**: + Phase 3 (usable forecaster). 2 weeks.
**Bonus**: + Phase 4 (architectural lift) + VoE counterfactuals. 3 weeks.

---

## 7. Convergence audit

Two consultation rounds, 15 agent-reports total, all GO-with-caveats verdicts:

| Phase | Specialist support | Reviewer approval |
|---|---|---|
| Phase 0 reliability fixes | C (NO-GO trigger), D, E | unanimous |
| Phase 1-bis probe rebuild | S1, S2, S5, S7 | unanimous |
| Phase 8a OCP + autoregressive | S6 + B (load-bearing ship gate) | unanimous |
| Phase 2 pixel-patch masking | S3, S8, S9 + B (jumps ahead) + A (atomicity) | unanimous |
| Phase 3 multi-step rollout | S3, S8 + C (rollout MAE gate) | unanimous |
| Phase 4 wider trunk + projector | S4 + A (with caveat) + B (single-knob) | conditional |
| DROP MEC, SIGReg, dim 256 (default), 4-knob arch lift | B + C + E | unanimous |
| Preserve iter-3 anti-collapse machinery | iter-3 evidence + every reviewer | unanimous |

No reviewer voted NO-GO at any iteration. All disagreements between A, B, C were resolved by:
- A's `dim 256` → contingent fallback gated on Phase 2 outcome
- B's "skip MEC/dim/arch entirely" → adopted (Phases 2/3/4 only run if needed)
- C's NO-GO triggers → all three caveats addressed in Phase 0

---

## 8. Key references

- **V-JEPA 2** ([arXiv 2506.09985](https://arxiv.org/abs/2506.09985)) — L1 loss, 3D-RoPE, tubelet multi-block masking
- **LeJEPA** ([arXiv 2511.08544](https://arxiv.org/abs/2511.08544)) — SIGReg theory (deferred)
- **LeWorldModel** ([arXiv 2603.19312](https://arxiv.org/abs/2603.19312)) — closest precedent to our scale (15M params, single GPU, planning works)
- **Intuitive Physics Emerges from V-JEPA** ([arXiv 2502.11831](https://arxiv.org/abs/2502.11831)) — Meta's VoE protocol; explicit demonstration that linear probing is the wrong test
- **C-JEPA** ([arXiv 2410.19560](https://arxiv.org/abs/2410.19560)) — VICReg-on-target + invariance for small-scale collapse
- **jepa-wms** ([arXiv 2512.24497](https://arxiv.org/abs/2512.24497)) — "What drives success in physical planning with JEPA-WMs?" Dec 2025
- **Kobak et al.** ([JMLR 2020](https://jmlr.org/papers/v21/19-844.html)) — optimal ridge penalty can be zero/negative
- **Belinkov** ([CL 2022](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and)) — probing methodology pitfalls
- **Geometry of Projection Heads** ([arXiv 2605.17180](https://arxiv.org/html/2605.17180)) — 1×1 conv as mathematical rank ceiling
- **keon/jepa** ([github](https://github.com/keon/jepa)) — toy V-JEPA on Moving MNIST: canonical masking + L1 + NO VICReg
- **lucas-maes/le-wm** ([github](https://github.com/lucas-maes/le-wm)) — small-scale JEPA reference implementation

---

## 9. F7-F12 ownership audit

This is the explicit phase pinning for the remaining failure modes in [docs/eval_failure_modes.md](docs/eval_failure_modes.md). F11 is the only orphan: it is acknowledged in the plan, but it is not assigned to a build phase because it is not actionable.

| F# | Verbatim failure mode | Phase ownership | One-line ownership note |
|---|---|---|---|
| F7 | **`spatial_tokens=64` hardcode** | Phase 4 | Phase 4 owns this because the wider-trunk architecture review is where the `encoder.py` rank ceiling is already acknowledged, and that is the first place the 64-token bottleneck can be lifted or deferred cleanly. |
| F8 | **`temporal_pos` ↔ mask token interaction** | Phase 2 | Phase 2 owns this because the masking redesign explicitly moves mask-token handling into the predictor and must define how `temporal_pos` combines with masked positions before the retrain can be trusted. |
| F9 | **Concat-Ridge probe cannot decode velocity** | Phase 1-bis | Phase 1-bis owns this because the probe rebuild already replaces concat-Ridge with the finite-difference velocity probe, which is the direct fix for the methodology bug. |
| F10 | **VICReg-on-encoder suppresses high-frequency dims** | Phase 4 | Phase 4 owns this because the architecture lift is the first planned place to move VICReg pressure off the encoder and onto the projector head without breaking single-knob discipline. |
| F11 | **Held-out regime drop is partly covariate shift** | Orphan / plan note only | This is an orphan because the current plan records the covariate-shift caveat but does not assign a phase; keep it as a reporting caveat, not a build target, unless a later evaluation pass explicitly scopes regime normalization. |
| F12 | **Predictor learns conditional mean at high mask ratio** | Phase 2 | Phase 2 owns this because the canonical masking + L1 retrain is the planned response to the high-mask conditional-mean collapse, and the per-mask-ratio monitor is part of that same training gate. |
