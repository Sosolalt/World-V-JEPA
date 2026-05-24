# V2 Consultation Log — Iteration 2 (2026-05-24)

**Trigger:** V2-Phase-1 ran on the V1 checkpoint with a re-engineered evaluation pipeline. Results were partially informative, partially broken, and divergent enough from rev-1 expectations to warrant a second consultation before committing further compute.

**Format:** Two rounds — Round 1 = 10 parallel specialists doing focused literature passes; Round 2 = 5 reviewers consolidating findings into an updated PLAN_V2.md (3 mid-stage A/B/C, then 2 final-validation D/E).

**Outcome:** Unanimous GO-with-caveats; no NO-GO. PLAN_V2.md rev 2 supersedes rev 1.

---

## 0. V2-Phase-1 numbers that triggered the consultation

Ran `scripts/evaluate.py` on `runs/20260519_195827_iter3_full_150/ckpt_best.pt` + `runs/baseline_pixel/20260519_225206/ckpt_best.pt`. 1500 sequences (subsampled to fit the 15 GB cap). Outputs in `assets/results/v2_phase1/metrics.json`.

| Probe | V-JEPA | Pixel | Δ |
|---|---|---|---|
| Per-frame pos R² — EMA target (V1 headline) | 0.577 | 0.585 | pixel +0.008 |
| Per-frame pos R² — online encoder | 0.555 | 0.585 | pixel +0.030 |
| Per-frame vel R² (physically impossible task) | 0.005 | 0.030 | noise both sides |
| 4-frame context-window pos R² | 0.423 | 0.386 | V-JEPA +0.037 |
| 4-frame context-window vel R² | **−0.59** | **−1.93** | both broken |
| Held-out regime (`random_velocities`) pos | 0.328 | n/a | -14pp vs in-dist |
| Rollout MAE V-JEPA predictor (all horizons 1-12) | **3.95** | — | 40× worse than copy-last 0.093, 400× worse than identity 0.009 |

Sklearn also emitted `LinAlgWarning: ill-conditioned matrix detected, rcond = 8.5e-8` during the Ridge fit at 8192 features for the rollout-decode probe.

Three immediate observations that turned out to be only partially correct after consultation:
1. EMA target slightly beats online (0.577 ≥ 0.555) — refutes "EMA stale" hypothesis from rev-1.
2. Negative R² on ctx-window probe looks like an encoder failure but might be methodology.
3. Predictor is useless at inference (cos sim 0.075, rollout MAE 40× copy-last).

---

## 1. Round 1 — 10 specialist literature passes

Each agent given a focused question, files to read, and asked for paper title + URL + summary + concrete code change. Reports capped at ≤1500 words each.

### S1 — Why did the context-window probe get negative R²?

**Headline:** Textbook methodology failure, not encoder pathology. Three compounding causes:
1. **Ridge α=1.0 with `lsqr` solver on rank-12 features in 32K-d space.** [Kobak, Lomond, Sanchez (JMLR 2020)](https://jmlr.org/papers/v21/19-844.html) "The optimal ridge penalty for real-world high-dimensional data can be zero or negative due to implicit ridge regularization" — in n ≪ p regimes with anisotropic spectra, the minimum-norm interpolator already provides implicit shrinkage; a fixed positive α is essentially random shrinkage along the wrong axes. Our `RIDGE_CV_ALPHAS = logspace(-3, 3, 13)` ceiling is too low; should be `logspace(-2, 6, 17)` because optimal α scales with `trace(XᵀX)/n` which hits 1e4-1e5 for un-standardized 32K-d features.
2. **No feature standardization.** [Belinkov "Probing Classifiers" (CL 2022)](https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and) explicitly documents that probe results are unreliable without (a) cross-validated regularization, (b) feature standardization, (c) random-feature controls. Our pipeline bypasses (a) and (b) for the ctx-window probe.
3. **Per-token flatten of 256×128 → 32K features.** Standard SSL evaluation uses GAP or attentive pooling (~128 features). Flatten is non-standard and only appropriate when spatial layout matters AND token count is small (~16). At 256 tokens with rank-12 features, the n ≪ p pathology is severe.

**Three concrete fixes ranked:**
1. Mean-pool tokens before Ridge + `StandardScaler` + `RidgeCV(alphas=np.logspace(-2, 6, 17))`. Expected: lifts ctx-window R² from −0.59 to plausibly +0.1 to +0.4.
2. Attentive probe (1 cross-attention head + linear, trained with AdamW for ~50 epochs). V-JEPA paper standard. Expected: +0.1 to +0.2 additional.
3. `PLSRegression(n_components=min(16, r_eff))` for multi-output velocity decode. Headline: per-ball R² instead of variance-weighted joint R².

**Critical observation S1 ends with:** even with perfect probing, no probe can decode 18 independent target dims from a 12-dim representation. The underlying rank collapse is the real ceiling.

### S2 — EMA vs online encoder: which is canonical?

**Headline:** V-JEPA / I-JEPA / DINOv2-3 / BYOL / MoCo ALL probe the ONLINE encoder. Our V1 used the EMA target. The +0.022 Δ (target − online) is Polyak averaging artifact, not staleness signal.

**Citations walked through:**
- [V-JEPA paper §4 Table 3 caption](https://arxiv.org/html/2404.08471v1): "We pool the feature map output by the frozen V-JEPA encoder using an attentive probe." Eθ (context encoder, online) is what they probe; Ē_θ (EMA target) is used "only during pretraining."
- [I-JEPA codebase](https://github.com/facebookresearch/ijepa): downstream tasks (ImageNet linear, low-shot, depth, counting) use the student.
- [BYOL paper](https://arxiv.org/pdf/2006.07733): linear evaluation on `y_θ` (online network's representation).
- [MoCo paper](https://arxiv.org/abs/1911.05722): query encoder, not momentum encoder.
- [DINOv2/v3](https://arxiv.org/html/2508.10104v1): student backbone is the universal visual encoder.
- ["Rethinking JEPA: Compute-Efficient Video SSL with Frozen Teachers"](https://arxiv.org/pdf/2509.24317): EMA teacher can be replaced by frozen pretrained encoder with no accuracy loss — implies EMA carries no genuinely separate downstream signal.

**Why our finding is "expected and benign":** with τ_start=0.99 and cosine schedule to 1.0 over 150 epochs, the final EMA ≈ student weights Polyak-averaged over the last ~30-50 epochs. EMA marginally smoother → +0.022 R² is exactly the magnitude expected from "EMA-as-Polyak-average." A *stale* teacher would show Δ in the −0.05 to −0.20 range.

**Plus a methodology gap S2 found:**
- ["Attention, Please! Revisiting Attentive Probing"](https://arxiv.org/abs/2506.10178) (ICLR 2026): attentive probing adds 4-17% absolute over linear probing on patch-token-based SSL models. The V-JEPA paper attentive probe is a 4-block transformer with learnable queries (~200K params).

**Conclusion:** rewrite headline as **online encoder + 4-frame context window + attentive probe**, demote EMA-target / per-frame / Ridge to legacy ablation rows.

### S3 — Is a usable predictor achievable at our scale?

**Headline:** Yes — [LeWorldModel (arXiv 2603.19312)](https://arxiv.org/abs/2603.19312) is direct existence proof at 15M params, single GPU, physics-like tasks. Single highest-leverage change: **pixel-patch tubelet masking BEFORE the encoder** (cross-paper consensus across V-JEPA, V-JEPA 2, I-JEPA, MAE, VideoMAE).

**Cross-paper analysis:**
- **V-JEPA 2 ([arXiv 2506.09985](https://arxiv.org/abs/2506.09985))**: 300M-param action-conditioned predictor trained SEPARATELY after the encoder is frozen, with **block-causal attention**, **teacher-forcing + 2-step rollout loss**, **L1 distance** (not MSE), **3D-RoPE**, action conditioning per step (not one token concatenated). 100× bigger than ours but RoPE, L1, block-causal, rollout loss, per-step conditioning all transfer to small scale.
- **DreamerV3 ([Nature 2025](https://www.nature.com/articles/s41586-025-08744-2))**: symlog targets, two-hot discrete regression, KL-balanced distributional next-latent prediction. Distributional, not point estimate. Point predictors over continuous latents collapse to conditional mean — matches our cos sim 0.075.
- **IRIS, Δ-IRIS, GAIA-1, GAIA-2**: every published latent forecaster that actually rolls out either (a) tokenizes to discrete codes + CE loss, or (b) uses latent diffusion / flow-matching. **Nobody rolls out an MSE-trained continuous JEPA predictor.** Our objective, not just our size, is the bottleneck.
- **C-JEPA ([arXiv 2410.19560](https://arxiv.org/pdf/2410.19560))** explicitly: "the I-JEPA prediction mechanism struggles to accurately learn the mean of patch representations, and EMA has been found inadequate in preventing collapse." Fix: apply var/cov to TARGET side + invariance loss.
- **DiT (Peebles & Xie ICCV 2023)**: AdaLN-Zero is the architectural unlock for transformer-based denoisers. Every modern video DiT (Stable Video Diffusion, GenTron, GAIA-2) uses AdaLN-Zero for timestep/action injection. Our predictor concats Δt as one of 321 tokens — structurally indistinguishable from a context token.

**Mask-ratio diagnosis:** at 75% output-token masking with EMA targets and MSE, the conditional-mean solution is approximately Bayes-optimal because target variance is small (slow billiards) and the loss is unbounded above. Our predictor sitting at cos sim 0.075 is **expected**, not pathological.

**Verdict on the 3 sub-questions:**
- (a) Useful predictor at our scale: yes, LeWM = existence proof.
- (b) Single highest-leverage change: pixel-patch masking BEFORE encoder.
- (c) Does our rollout-MAE result confirm V-JEPA paper's "predictor is training-only" framing? **Partially — but the confirmation is weaker than it looks.** Confirmed for V-JEPA's original recipe; not confirmed for the JEPA family in general. V-JEPA 2, GAIA-2, DreamerV3, IRIS, LeWM all show predictors CAN be made into forecasters given (1) AdaLN action/Δt conditioning, (2) L1 or distributional loss, (3) explicit rollout training, (4) discrete tokens OR diffusion head OR SIGReg, (5) pre-encoder masking.

### S4 — What lifts the encoder rank ceiling?

**Headline:** The 1×1 conv `Conv2d(256, 128, 1)` at `encoder.py:37` IS the mathematical rank ceiling — [Geometry of Projection Heads (arXiv 2605.17180)](https://arxiv.org/html/2605.17180) Theorem 5.1: "Linear heads force dimensional bottlenecking — the backbone covariance becomes limited to the head's rank." Top 5 techniques to lift it, ranked:

| # | Technique | Expected lift | Risk |
|---|---|---|---|
| 1 | Wider trunk (256→384) + 2-layer non-linear projector | **+30 to +60 on ctx_rank, +20 on z_pred** | Low — orthodox SSL practice |
| 2 | **MEC** (`-logdet(I + μZᵀZ/n)`, [arXiv 2210.11464](https://arxiv.org/abs/2210.11464)), 15-line drop-in for VICReg cov term | **+20 to +40** | Low — well-studied |
| 3 | SIGReg ([LeJEPA arXiv 2511.08544](https://arxiv.org/abs/2511.08544)) on projector output with M=256 (NOT 1024 — overkill at our 10K-sample scale) | **+30 to +50 on projector, +10-20 on backbone** | Medium |
| 4 | KoLeo regularizer on per-sample pooled token (DINOv2/v3, 10 lines) | **+10 to +20** | Very low — DINOv2-proven |
| 5 | Multi-depth regularization (apply var+cov after `gn3` 256-d AND after 1×1 128-d) | **+15 to +30** | Low |

**Critical observation:** S4 explicitly says "wider trunk + 2-layer non-linear projector" is the single biggest lever and matches PLAN_V2 Phase 3 verbatim. MEC can land BEFORE the architecture lift as an isolated test of "is the loss term the bottleneck vs the architecture" — but **at d=128 with the 1×1 squeeze still in place, the ceiling argument from S4 itself caps achievable rank**. This tension surfaces in Reviewer A's critique.

**Sources S4 reviewed:** [LeJEPA arxiv](https://arxiv.org/abs/2511.08544), [LeJEPA explainer](https://arxiviq.substack.com/p/lejepa-provable-and-scalable-self), [MEC arxiv 2210.11464](https://arxiv.org/abs/2210.11464), [Matrix Information Theory for SSL arxiv 2305.17326](https://arxiv.org/pdf/2305.17326), [DINOv2/v3 KoLeo](https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py), [Projection Head Information Bottleneck arxiv 2503.00507](https://arxiv.org/html/2503.00507), [RankMe arxiv 2210.02885](https://arxiv.org/pdf/2210.02885), [T-REGS arxiv 2510.23484](https://arxiv.org/pdf/2510.23484), [Orthogonality Reg arxiv 2411.00392](https://arxiv.org/pdf/2411.00392).

### S5 — Linear probing methodology pitfalls

**Headline:** **The V-JEPA vs Pixel 0.008 R² gap is ~0.3σ at N=300 — statistically indistinguishable.** Need N≥2500 or sequence-level paired bootstrap.

**Specifically:** for R² near 0.5-0.6 with N test points, parametric SE ≈ `2·(1−R²)·sqrt(R²/N)`. With **N = 9600 frame-targets but only 300 sequences**, frames within a sequence are heavily correlated — the *effective* N for SE purposes is ~300, not 9600 (a 32× variance underestimate). At R²=0.58, N_eff=300 gives SE ≈ **0.027**. A 0.008-pp gap is **~0.3σ** — utterly inside noise. Bootstrap **at the sequence level** (resample sequences, not frames) is mandatory.

**Additional findings:**
- ["Attention, Please!" arXiv 2506.10178](https://arxiv.org/abs/2506.10178) — attentive probing standard for MIM-class encoders, +3-8 pp top-1 over LP.
- ["Rethinking Evaluation Protocols of Visual Representations Learned via SSL" arXiv 2304.03456](https://arxiv.org/abs/2304.03456) — BN/standardization before LP resolves k-NN vs LP inconsistency.
- ["A Closer Look at Benchmarking SSL" PMC 2024](https://pmc.ncbi.nlm.nih.gov/articles/PMC12289721/) — emphasizes standardized features + tuned regularization mandatory.
- ["Reduced Rank Ridge Regression" PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC3444519/) — sklearn `Ridge` with 2D `y` solves K independent ridge problems sharing the same `(X^T X + αI)^{-1}` factorization — mathematically identical to per-output Ridge with the same α.

**Shippable probing protocol from S5:**
```
Features = concat([mean_pool, max_pool, attention_pool_q1]) over tokens
Pipeline: StandardScaler(X) → StandardScaler(y) → RidgeCV(alphas=logspace(-1,5,13), cv=5 over SEQUENCES not frames)
Sequence-level bootstrap (200 resamples) for R² and 95% CI
Cross-check with KNeighborsRegressor(k=10) on L2-normalized pooled features
Headline number: attentive probe (1 cross-attn head, 100 AdamW steps)
```

### S6 — Physics world-model evaluation beyond probes

**Headline:** Add **OCP (binary collision prediction)** + **VoE counterfactuals (we have pymunk!)** + **autoregressive rollout MSE**. Meta's own "Intuitive Physics Emerges from V-JEPA" uses VoE exclusively, not probes.

**Top 6 protocols ranked:**

1. **Violation-of-Expectation (V-JEPA Intuitive Physics, [arXiv 2502.11831](https://arxiv.org/html/2502.11831v1), code at [facebookresearch/jepa-intuitive-physics](https://github.com/facebookresearch/jepa-intuitive-physics))**: surprise `S_t = ||p_phi(f_theta(V[t:t+C])) − g_psi(V[t:t+C+M])||_1`. Classify (possible, impossible) pair by relative surprise. With pymunk we can generate counterfactuals: teleport, vanish, fake-rebound, pass-through. Why linear probing can't fake: surprise tests the predictor's expectations, not encoder content.

2. **IntPhys 2 quadruplet structure ([arXiv 2506.09849](https://arxiv.org/abs/2506.09849), [code](https://github.com/facebookresearch/IntPhys2))**: 4-tuple cancels superficial appearance cues that a 2-tuple lets through.

3. **Physion OCP ([arXiv 2106.08261](https://arxiv.org/abs/2106.08261))**: binary "will the two cued objects contact within next H frames?" 2000 train + 150 test movies; humans ~80%, pixel CNNs ~55-60%. Calibrated binary task (chance = 50%). Requires forward simulation in encoder representation. ~100 LOC to adapt to billiards.

4. **PHYRE AUCCESS** — heavier, requires actual control task. Lower priority.

5. **Energy conservation via probed velocities** — predict velocities, compute `KE_t = 0.5·m·sum(v²)`, compare decay trajectory to friction model.

6. **Multi-step latent rollout MSE (DreamerV3, IRIS protocols)** — modify `degradation_curve` to autoregressively feed `z_pred` back as context. Exposes error compounding that single-shot `dt=h` jumps mask.

**Top 3 to add ranked by effort × info gain:** OCP (~100 LOC, highest signal per LOC) → VoE counterfactuals (~250 LOC, highest information gain) → autoregressive rollout + KE conservation (~60 LOC each, cheapest correctness check).

### S7 — Why did the velocity probe get negative R²?

**The most consequential single finding of Round 1.**

**Headline:** **V-JEPA's encoder is NOT designed to encode velocity per-token.** Velocity lives in the temporal DERIVATIVE: `v̂(t) ≈ W · (pool_S(z_t) − pool_S(z_{t-1}))`. Concat-Ridge probe is fundamentally the wrong instrument. Plus **VICReg-on-encoder actively suppresses high-frequency dimensions** where velocity would live.

**The three mechanisms compounding our failure:**

1. **V-JEPA's objective.** [Garrido et al. "Learning and leveraging world models in visual representation learning" arXiv 2403.00504](https://arxiv.org/abs/2403.00504): the encoder targets the EMA-target representation of a future frame — by construction it must represent **the static content needed to predict the future**, not the velocity. Velocity lives in the **predictor**, conditioned on Δt: `z_{t+Δt} ≈ predictor(z_{≤t}, Δt)`. The only paper that adds explicit velocity decoding is [MC-JEPA arXiv 2307.12698](https://arxiv.org/abs/2307.12698) with a separate flow-prediction branch. V-JEPA does not.

2. **Concat-Ridge has to discover the subtraction operator.** Our 32K-dim Ridge with no inductive bias for "z_t and z_{t-1} should be subtracted," under-regularized (α=1.0 lsqr), is a textbook recipe for negative held-out R².

3. **VICReg on encoder outputs actively suppresses fast-changing dimensions.** [VICReg arXiv 2105.04906](https://arxiv.org/abs/2105.04906) penalizes off-diagonal covariance across the WHOLE BATCH, mixing time steps. [VICRegL arXiv 2210.01571](https://arxiv.org/abs/2210.01571) and MC-JEPA §4 both note: applying VICReg to a video encoder pushes variance into STATIC dimensions and SUPPRESSES fast-oscillating ones. [Aubret et al. arXiv 2408.10864](https://arxiv.org/abs/2408.10864) measures this: VICReg on a video encoder gives 30-40% LOWER velocity-decoding R² than the same encoder without VICReg, while keeping position-decoding R² flat. Our `reg_target=both` (which CLAUDE.md flags as the "actual collapse-preventing signal at small scale") IS the mechanism that breaks the velocity probe.

**Existing precedents for difference-features as velocity probe:** Time-Equivariant Contrastive Learning (arXiv 2207.04050), Time-Contrastive Networks (Sermanet 2018), DINOv2 video probes §5, "Motion-Aware Mask Feature Reconstruction" 2024 — last one reports `Δz = z_t − z_{t-1}` gives ≥0.4 R² on velocity decoding where concat-Ridge gives near-zero.

**3-line code change S7 proposed:**
```python
diff_z = target_z[:, 1:].mean(axis=2) - target_z[:, :-1].mean(axis=2)  # (N, T-1, D)
vel_aligned = store.velocities[:, 1:]
print("diff-probe vel R^2 =", linear_probe(diff_z[:, :, None, :], vel_aligned, train_idx, test_idx)["overall_r2"])
```

### S8 — Small-scale JEPA reproductions on GitHub

**Headline:** No sub-1M JEPA exists publicly. Strongest precedents at 5-15M params. Single recipe to copy verbatim: canonical V-JEPA multi-block input masking.

**12 repos inspected, top 7:**

| Repo | Params | Dataset | Headline | Key recipe vs us |
|---|---|---|---|---|
| **[lucas-maes/le-wm (LeWorldModel)](https://github.com/lucas-maes/le-wm)** | ~15M, 192-d ViT, 1 GPU | pusht, cube, tworooms, reacher | Beats DINO-WM on Push-T + Reacher | SIGReg only (w=0.09), no VICReg, no EMA blocks, AdamW lr=5e-5, bs=128, MSE pred loss, AdaLN action conditioning, projector with 1024 random projections |
| **[facebookresearch/jepa-wms](https://github.com/facebookresearch/jepa-wms)** (Dec 2025) | ViT-S/14 → ViT-L/16 frozen | Metaworld, Push-T, PointMaze, DROID, RoboCasa | Beats DINO-WM and V-JEPA-2-AC on planning | AdaLN predictor, "ftprop" full-trajectory propagation, 2-step rollouts trained-time, predictor depth 6-24, frozen DINOv2/v3 encoder |
| **[keon/jepa](https://github.com/keon/jepa)** (`vjepa.py`) | tiny ~1-2M | Moving MNIST | I-JEPA variant in same repo: 52.7% CIFAR-10 linear probe at 100 ep | **L1 loss** (`.abs().mean()`), EMA 0.998→1.0, **8 short masks @15% + 2 long @70% (V-JEPA canonical)**, lr=3e-4, bs=32, AdamW, **NO VICReg/SIGReg** |
| [facebookresearch/eb_jepa](https://github.com/facebookresearch/eb_jepa) | configurable | CIFAR-10, Moving MNIST, Two Rooms | "Hours on single GPU" | VICReg primary anti-collapse; configs encode std/cov coefficients |
| [jonwiggins/H-JEPA](https://github.com/jonwiggins/H-JEPA) | ViT-Tiny ~13.8M | CIFAR-10, ImageNet-100 | CEM planner present | AdaLN-Zero action-conditioned predictor; SIGReg-only mode that DROPS EMA target entirely |
| [filipbasara0/simple-ijepa](https://github.com/filipbasara0/simple-ijepa) | 11M | STL-10 96×96, 100K imgs | **77.07% top-1 linear probe** | Simplified mask generator; EMA gamma scheduled |
| [facebookresearch/jepa](https://github.com/facebookresearch/jepa) ViT-L/16 | ViT-L | VideoMix2M | (canonical reference) | **Loss exponent 1.0 (L1)**, **reg coefficient 0.0** (NO VICReg), 8 short masks @0.15 + 2 long @0.70, EMA 0.998→1.0, predictor depth 12 dim 384 |

**Direct prior-art matches:**
- Pymunk billiards JEPA: NONE FOUND. Yours is the only public pymunk-billiards JEPA.
- Predictor actually useful at inference: **le-wm** (CEM planner in latent space, 48× faster than foundation WMs) and **jepa-wms** (planning under MPC) — both use **action-conditioned AdaLN predictors with multi-step rollouts**.
- <1M params with R²>0.70 on structured output: **not found**.

**Common failure modes documented across these repos:**
- EMA alone insufficient at small scale (Apple ML, VJ-VCR, search synthesis).
- Small batch / small expander → collapse (your bs=64 is borderline).
- Predictor learns conditional mean = exactly your iter-3 cos sim 0.075.
- **No public small repo reports "predictor useless at inference"** — because they either (a) train with action-conditioned multi-step rollouts so predictor must forecast, or (b) report only probe scores and never measure predictor cos-sim.

**Single recipe to copy verbatim:** canonical V-JEPA multi-block input masking (8 short masks @ spatial_scale=0.15 + 2 long masks @ spatial_scale=0.70) applied on input patches BEFORE the encoder (not as output `mask_token` substitution). Both FB V-JEPA (ViT-L config) and keon/jepa toy reproduction use this exact configuration; it's what lets keon ship working V-JEPA with **zero variance/covariance regularization**.

### S9 — Masking strategy deep dive

**Headline:** Our token-output `mask_token` replacement is the WRONG asymmetry direction. V-JEPA / I-JEPA / MAE mask BEFORE the encoder (input patches dropped from the sequence).

**Cross-paper consensus on masking:**
- V-JEPA paper: 2×16×16 tubelets, 8 short @ 15% + 2 long @ 70% → ~90% effective mask. Blocks span full temporal axis (tube). Loss L1 on predictor output vs EMA-target tokens. Masking at **encoder input** (patches dropped from x-encoder's input sequence). Mask tokens with pos embeddings appended at **predictor input**.
- V-JEPA 2: same tubelet recipe.
- I-JEPA: 4 target blocks at scale (0.15, 0.20), 1 context block at scale (0.85, 1.0). Target regions REMOVED from context.
- MAE: 75% random patch masking, 75% sweet spot for linear probe.
- VideoMAE: **tube masking @ 90%** (higher ratio needed because video has more temporal redundancy).
- SiamMAE: **asymmetric 0% past frame / 95% future frame** masking, cross-attention decoder.

**Critical answer to "does 75% token-output replacement leave enough context?":** **No.** Two compounding problems:
1. Encoder processes full clip, then 240 of 320 tokens replaced with same mask vector. Predictor sees 80 real + 240 identical masks. Target encoder ALSO sees full clip. The asymmetry V-JEPA depends on (predictor sees less than target) is **inverted**.
2. With VICReg pinning variance/covariance, cheapest loss minimum is "output the per-dim mean of EMA targets" — exactly the 0.075 cos sim and 40× copy-last rollout MAE we observe.

**V2 masking recipe from S9 (Phase 4a/5):**
| Item | Value |
|---|---|
| Tubelet | 2×8×8 (T×H×W) for our 64×64 inputs |
| Where to mask | Patch tokens BEFORE the encoder (drop from input sequence). Delete `apply_context_mask` |
| Mask shape | Multi-block, tube-style (mask spans full temporal axis) |
| Block budget | 4 short @ 15% spatial + 1 long @ 50% spatial → ~75% union mask |
| Mask token | Inserted at predictor input with pos embedding, not at encoder output |
| Predictor loss | Smooth-L1 or L1 on EMA target at masked positions only |
| Phase gate | Predictor cos-sim ≥ 0.40 on held-out at ep 30; ctx_eff_rank ≥ Phase 3 value |

### S10 — Loss function ablation literature

**Headline:** **L1 (V-JEPA 2) > Smooth L1 (V-JEPA 1) ≈ MSE at our current setup** because PyTorch `smooth_l1` β=1.0 makes it ≡ MSE for our residual range (|r| < 1 with normalized latents at std≈1).

**Cross-paper analysis:**
- V-JEPA 2 ([arXiv 2506.09985](https://arxiv.org/html/2506.09985v1)) uses plain L1 (`‖P − sg(Ē)‖₁`). No published ablation justifies L1 over Smooth L1 or L2.
- BYOL MSE on L2-normalized vectors ≡ cosine. [Feng et al. arXiv 2309.16109](https://arxiv.org/pdf/2309.16109) shows L2-norm induces 6th-order dynamics that escape collapsed initializations.
- DINO: CE on K=65536 sharpened-softmax prototypes. MSE works but strictly worse.
- L1 vs L2 robustness: arXiv 1705.09954, arXiv 1804.07090 link L1 reconstruction to higher-rank, more robust representations.
- VICReg coefficients at d=128: original used μ=25 (var), ν=1 (cov) at d=8192. Cov term scales as O(d), so at d=128 cov gradient is ~64× smaller per dim. Our `cov=25` is defensible; pushing further has diminishing return.
- Loss on masked tokens only vs all tokens: V-JEPA 2.1 ([arXiv 2603.14482](https://arxiv.org/html/2603.14482v2)) studied this — context-loss-on-visible with distance weighting lifts ADE20K 22.2→33.8 mIoU but drops ImageNet linear-probe 82.2→72.6 if weight too high.

**Direct answer to "would MSE → Smooth L1 lift R²?":** **+0 to +0.02 R² expected, small.** Smooth L1 ≡ MSE at β=1.0 for our residuals. Plain L1 (V-JEPA 2 style) would be a stronger lever (full sign gradient).

**Ranked changes from S10:**
1. `loss_fn: l1` (true L1, add to losses.py) — **+0.03 to +0.08 R²**, low risk.
2. Distance-weighted loss: weight predicted tokens by 1/(1+Δt) — +0.02 to +0.05.
3. `smooth_l1` with β=0.1 (not default 1.0) — +0.01 to +0.03.
4. Drop ν from 25 → 5 with cov-sum normalized by d²/2 — +0.00 to +0.02.
5. L2-normalize before similarity term (BYOL-style) — 0 to +0.02.

---

## 2. Round 2 part 1 — 3 mid-stage reviewers (A, B, C)

Each reviewer given the compressed 10-specialist findings and asked to consolidate into an updated plan with verdict.

### Reviewer A — plan integrity

**Verdict: GO with caveats** (3 must-add items).

**Key consolidation moves:**
- "Phase 2 = `latent_dim 128→256` (one yaml flip, lifts the rank ceiling); Phase 2.5 = MEC swap. Rationale: per S4, the 1×1 channel squeeze is the mathematical rank ceiling — running MEC against an unmoved ceiling produces ambiguous results."
- "Phase 5 (canonical V-JEPA masking) must atomically update `scripts/evaluate.py`: `degradation_curve` (line 546) and `rollout_position_error` (line 669) currently call `model.apply_context_mask`, which Phase 5 deletes."
- "Phase 1-bis gates require N ≥ 2000 test sequences; the current `--max-eval-sequences` default is 800 (evaluate.py:1248). Bump default or pass `--max-eval-sequences 2500` explicitly."
- "Velocity headline gate must use the velocity-via-difference probe (S7), not concat-Ridge ctx-window — the latter is the instrument S7 identified as fundamentally wrong."

**Single biggest risk A identified:** Phase 2 (MEC) failing silently because the 1×1 squeeze caps rank below the gate. Mitigation: invert order — dim 256 first, then MEC.

### Reviewer B — pragmatist / shipping order

**Verdict: GO with caveats on a reordered, trimmed version.**

**Cost/benefit table B produced:**

| Phase | ΔR² pos | ΔR² vel | Eng days | Compute hrs | Ship-clean % |
|---|---|---|---|---|---|
| 1-bis | +0.05 to +0.20 | **+0.30 to +0.55** | 1-1.5 | 0 | 95% |
| 2 (MEC) | +0.00 to +0.05 | +0.00 to +0.05 | 0.5 + 0.5 gate | 0.7 + 2 | 70% |
| 3 (dim 256) | +0.02 to +0.08 | +0.02 to +0.08 | 0.1 + 0.5 gate | 0.7 + 2 | 60% |
| 4 (4-knob lift) | +0.05 to +0.15 | +0.05 to +0.15 | 3 | 0.7 + 2 | **35% (violates single-knob)** |
| 5 (pixel-patch + L1) | +0.05 to +0.15 | +0.05 to +0.20 | 3 | 0.7 + 2 | 50% |
| 6 (multi-step rollout) | +0.00 to +0.05 | +0.00 to +0.05 | 2 | 2 | 30% |
| 7 (SIGReg) | unknown | unknown | 2 | 2 | 25% |
| 8 (OCP+VoE+AR) | n/a (new axes) | n/a | 1.5 | 0 | 85% |

**B's reorder:** Phase 5 (pixel-patch masking) jumps ahead of 2/3/4. Three independent reasons:
1. S3 + S8 cross-paper consensus: pixel-patch input masking is THE highest-leverage training change.
2. Iter-3 evidence: token-output masking (E4) produced silent encoder collapse — the predictor faked diversity via `mask_token`. Pixel-patch input masking removes that escape valve.
3. Phase 5 makes Phase 6 possible. Current predictor (cos-sim 0.075) cannot be rolled out at inference. No amount of dim widening or MEC fixes that. Phase 5 is on the critical path to a forecaster.

**B's drop list:** Phase 7 (SIGReg) — drop now. Phase 2 (MEC) — research curiosity. Phase 3 (latent_dim 256) — fallback only. Phase 6 — depends on Phase 5 working first. Phase 4 4-knob — hard NO.

**B's "happy path":** Ship V1 if Phase 1-bis (with corrected probes + bootstrap CI) AND OCP AUC ≥ 0.65 vindicate V-JEPA.

**B's hard caveats:**
- Hard budget cap: ≤8 full 150-epoch runs across the whole month.
- CLAUDE.md feature-review gate at each phase boundary explicit (which skill triggers when).
- Preserve all iter-3 anti-collapse machinery.
- Predictor cos-sim gate of 0.40 had no precedent — propose 0.25 falsification, 0.40 ship-it.

### Reviewer C — adversarial skeptic (found things 10 specialists missed)

**Verdict: GO with caveats, with 3 NO-GO triggers if unaddressed.**

**6 things 10 specialists missed (the most consequential review output):**

1. **The 8×8 token grid is the rank ceiling, not the 1×1 squeeze.** `encoder.py:26` has `assert spatial_tokens == 64`. Encoder downsamples 64×64 → 8×8 (three stride-2 convs). With 9 balls on 64×64, each 8-pixel cell contains ~1 ball center. After mean-pooling, spatial info averaged away. **S4 blamed the 1×1 conv; the real ceiling is the 8×8 grid + mean-pool in evaluation.** Widening `latent_dim 128→256` (Phase 3) does nothing if per-token information is already saturated.

2. **Pixel baseline shares the V-JEPA encoder.** `baselines/pixel_predictor.py:84` instantiates `ContextEncoder` directly. The 0.616 vs 0.567 gap is two encoders of identical capacity trained on different objectives. **If we improve the encoder (Phases 3-4), the pixel baseline improves too** — the "win" may be permanently <1pp. The plan has no plan to break this equivalence.

3. **`ctx_effective_rank` is computed on ONE batch per epoch.** `vjepa.py:101-104` comment + `train.py:341-364`. Rank=12 may be measurement noise.

4. **`RIDGE_CV_ALPHAS = np.logspace(-3, 3, 13)` already in code, wrong upper bound.** S1's recommended grid is `logspace(-2, 6, 17)`. The current 1e3 ceiling silently truncates.

5. **The held-out -14pp drop on `random_velocities` may be covariate shift in the probe target distribution.** Train set biased toward `break`/`midgame_*` (balls cluster near table center); `random_velocities` more uniform spatial coverage. **This is not "OOD encoder brittleness" — it's covariate shift in the probe itself.**

6. **Dataset intrinsic dimensionality.** With friction enabled, the dissipative dynamical system collapses onto low-dim manifolds rapidly. **Rank 12 may be the actual intrinsic dimensionality of the dataset**, in which case no architectural fix exceeds it. Plan should include non-parametric intrinsic-dim estimate BEFORE multi-day architecture runs.

**Plus 2 self-contradictions in the candidate plan:**
- "Phase 4 (pixel-patch masking BEFORE encoder) breaks Phase 3 (use_spatial_pos_embed + projector)" — Phase 3's pos-embed is calibrated for unmasked tubelets; Phase 4 is a re-train of encoder from scratch.
- "Phase 4c says `ema_τ_start back to 0.996 + lr_peak back to 1.5e-4`" while §4 explicitly says "NOT removing iter-3 anti-collapse machinery." Phase 4c contradicts §4 verbatim.

**3 NO-GO triggers C identified:**
- Phase 2 (MEC) NaNs on default.npz within 10 epochs.
- Phase 1-bis with attentive + RidgeCV + EMA→online still yields position R² < 0.60 AND pixel baseline < 0.70 (both encoders capped → dataset intrinsic dim is the ceiling).
- Phase 3 lifts ctx_eff_rank to ≥30 but rollout MAE at horizon=10 stays ≥3.0 (rank lift doesn't transfer to predictor utility).

**C's must-address caveats:**
1. Phase 0 intrinsic-dim sanity (TwoNN or MLE-ID).
2. Bump RIDGE_CV_ALPHAS to logspace(-3, 6, 19).
3. Measure ctx_eff_rank over ≥10 batches per epoch.
4. Per-mask-ratio cos-sim monitor at Phase 4a 30-ep gate.
5. Resolve Phase 3 → Phase 4 invalidation.
6. Replace shared-encoder pixel baseline with frozen-V1-encoder version.

---

## 3. Round 2 part 2 — 2 final-validation reviewers (D, E)

Given the unified candidate plan synthesizing A + B + C, asked: does this address every reviewer's concern, find new failure modes, vote final.

### Reviewer D — pragmatic validation

**Verdict: GO with caveats** (7 caveats, all specific).

**Key contributions:**

- **NEW failure mode D caught:** the `mask_ratio` at eval time inherits `cfg.training.mask_ratio` = 0.75 in `degradation_curve` (`evaluate.py:499`) AND `rollout_position_error` (`evaluate.py:604`). Every cos-sim and rollout number we report uses **training-distribution masking**. The cos sim 0.075 is partially this artifact, partially the predictor. Phase 2's move of masking into the predictor must thread an explicit `--eval-mask-ratio` CLI argument.
- **NEW failure mode D caught:** `temporal_pos` (vjepa.py:43) is added AFTER encoder output. Phase 2's "mask BEFORE encoder" plan must specify whether masked positions get `temporal_pos` or not — if yes, the predictor can leak frame identity through the mask.
- **NEW failure mode D caught:** `--max-eval-sequences` CLI default is 800 (`evaluate.py:1248`); function signature default at line 975 is 2000. Mismatch silently caps.
- **NEW failure mode D caught:** the EMA target encoder runs on the FULL unmasked target frame (`vjepa.py:83-85`). Once masking moves to predictor and tubelets cover multiple frames, target encoding strategy may need change (V-JEPA 2 encodes target tubelets, not single frames). Not flagged anywhere.
- **NEW correction to C:** the per-frame ceiling is min(spatial_tokens, latent_dim) = min(64, 128) = **64**, not 128. So ctx_eff_rank=12 is at 19% of its actual ceiling, not 9%. Dim 128→256 cannot lift a per-frame ceiling of 64. What dim 256 lifts is the cross-token rank ceiling, which is what VICReg `cov_reg` operates on.

**D's recommendation:** drop Phase 0's raw-frame TwoNN (low value — Phase 1-bis attentive-probe R² is a better proxy). Keep frozen-V1-encoder pixel baseline (critical, ~2-3h compute) + batch-level eff_rank (non-negotiable, ~2h work) + ridge grid bump (one-line edit).

**D on the cos-sim gate of 0.25 vs 0.40:** 0.25 + per-mask-ratio monitor is the right compromise — Reviewer C correct that measuring only at training mask ratio is meaningless. But measurement must be at INFERENCE mask_ratio=0.0, not training ratio.

### Reviewer E — final adversarial check

**Verdict: GO with caveats.**

**E walked through A/B/C concerns explicitly. Status table:**

| A's concerns | Status |
|---|---|
| Atomic update of evaluate.py:546,669 | **PARTIALLY** — plan says delete in Phase 2, but Phase 1-bis (Day 1) runs current code with mask still active |
| max-eval-sequences 800 → ≥2000 | ADDRESSED in plan, NOT in code (`evaluate.py:1248` still 800) |
| Velocity headline via difference-probe | ADDRESSED |
| Phase 4c rejection (LR back to 1.5e-4) | ADDRESSED — dropped explicitly |
| latent_dim 128→256 first | **NOT ADDRESSED** — dropped entirely. A will object. Mitigation: contingent fallback |

| B's concerns | Status |
|---|---|
| Reject 4-knob arch lift | ADDRESSED |
| Phase 5 jumped ahead | ADDRESSED via renaming |
| Drop SIGReg, MEC, dim 256 | ADDRESSED |
| ≤8 full runs / month | PARTIALLY — counting: 2+1+2+more = 7-8 worst case, at the cap |
| Phase-boundary reviewer triggers | NOT EXPLICITLY ADDRESSED |
| OCP AUC ≥ 0.65 ship gate | ADDRESSED |
| Cos-sim 0.25 falsification / 0.40 ship | ADDRESSED |

| C's concerns | Status |
|---|---|
| Phase 0 intrinsic-dim BEFORE anything | ADDRESSED |
| Fix ctx_eff_rank to ≥10 batches/epoch | ADDRESSED in plan, NOT in code |
| Replace shared-encoder pixel baseline | ADDRESSED in plan |
| Bump RIDGE_CV_ALPHAS | ADDRESSED in plan, NOT in code |
| per_mask_ratio_cos_sim monitor | ADDRESSED |
| Phase 3 → Phase 4 invalidation | **NOT ADDRESSED** — Phase 4 reorder makes it worse |
| NO-GO if caveats 1+3+6 unaddressed | LIFTED conditional on Phase 0 code-complete |
| Rollout MAE at horizon=10 beats identity | ADDRESSED |

**3 NEW failure modes E surfaced:**

1. **Inference-time mask leak.** `scripts/evaluate.py:546, 669` call `apply_context_mask(ctx_tokens, mask_ratio=0.75)` at inference time. **This is the source of the cos sim 0.075 catastrophe.** Phase 1-bis claims "no retrain" but if we don't patch eval to call `predict(ctx_tokens, dt)` WITHOUT masking, OCP and rollout MSE in Phase 8a will reproduce the same artifact. Must add to Phase 0: "drop `apply_context_mask` call at inference; mask is training-time perturbation."

2. **Encoder hardcoded `spatial_tokens == 64`** (`encoder.py:26`). Phase 4 widens `conv3` to 384 channels, but `assert spatial_tokens == 64` and fixed 8×8 reshape silently keep token count fixed. **If wider trunk is meant to address rank ceiling caused by spatial bottleneck (C's diagnosis), it won't** — token count IS the bottleneck.

3. **`--ram-limit-gb 15` silently rejected** at our actual run today (`could not set RLIMIT_AS to 15.0 GB: current limit exceeds maximum limit`, `scripts/evaluate.py:55`). At `max_eval_sequences=2000` flatten step is ~16 GB peak. **At 800 sequences we already touched the cap; at 2000 we OOM.**

**E's 7 final caveats:**
1. Phase 0 MUST include deleting/guarding `apply_context_mask` calls at evaluate.py:546, 669, with regression test that V1 cos-sim degradation curve recomputes to non-0.075.
2. Phase 0 MUST verify `--max-eval-sequences 2000` runs without OOM under existing 15 GB cap.
3. Add re-validation step: "if Phase 4 changes encoder, re-run Phase 2's per-mask-ratio + cos-sim gate on new encoder."
4. Add contingent fallback: "if Phase 2 ctx_eff_rank ≤15 with new mask, run latent_dim=256 as single-knob diagnostic before Phase 3."
5. Assert "≤8 full 150-epoch runs across entire month, week-3 deferred knobs count against this cap."
6. Write CLAUDE.md feature-review-gate triggers into each phase boundary.
7. Phase 4 documentation should flag `spatial_tokens=64` hardcode as the un-addressed bottleneck.

---

## 4. Convergence audit

**15 agent-reports total** (10 specialists + 3 mid + 2 final). **All voted GO with caveats; none voted NO-GO.**

**Strong consensus (all 5 reviewers):**
- Phase 1-bis (probe rebuild + no retrain) MUST come first.
- Phase 4 four-knob arch lift REJECTED (single-knob discipline mandatory).
- Preserve iter-3 anti-collapse machinery.
- Bump `max-eval-sequences` from 800 to ≥2000.
- ≤8 full runs / month budget cap.

**Disagreement that was resolved by contingent fallback (A vs B):**
- A wanted `latent_dim 128→256` first to lift ceiling before MEC.
- B wanted to drop dim 256 + MEC entirely, jump straight to Phase 5 (pixel masking).
- **Resolution:** B's ordering adopted (Phase 5 first as Phase 2 in rev 2). dim 256 kept as contingent fallback gated on "Phase 2 ctx_eff_rank caps at ≤15."

**Disagreement that was resolved by deletion (B vs C on MEC):**
- B: research curiosity, drop.
- C: needs NaN safeguards + sanity gate on simple.npz first.
- **Resolution:** dropped per B; if revived later, C's safeguards apply.

**Critical issues no one would have caught without C:**
- 8×8 token grid as real ceiling.
- Shared-encoder pixel baseline.
- 1-batch ctx_eff_rank measurement.

**Critical issues no one would have caught without D/E:**
- Inference-time mask leak (E) — explains cos sim 0.075 partially.
- `--max-eval-sequences` CLI default 800 vs function default 2000 mismatch (D).
- `RLIMIT_AS` silently rejected (E).
- `temporal_pos` interaction with mask token unspecified (D).
- `spatial_tokens=64` hardcode (E).

---

## 5. Critical NEW failure modes recorded for future iterations

Catalogue of failure modes the consultation surfaced. These are claims about the codebase that should be verified before any future plan touches them.

| # | Failure mode | File:line evidence | Effect | Fix |
|---|---|---|---|---|
| F1 | Inference-time mask leak | `evaluate.py:546, 669` (reads `cfg.training.mask_ratio` = 0.75 at eval) | Every cos-sim & rollout V1 number measured with training-time mask still active | Add `--eval-mask-ratio` CLI flag default 0.0; default the helper functions to 0.0 not training value |
| F2 | `ctx_effective_rank` 1-batch measurement | `train.py:341-364` uses last batch only | rank=12 diagnosis may be ±5 noise | Compute over ≥10 batches per epoch, log mean±std |
| F3 | Pixel baseline shares V-JEPA encoder architecture | `baselines/pixel_predictor.py:84` re-instantiates `ContextEncoder` | "V-JEPA loses to pixel" is partly comparing same architecture | Add frozen-V1-encoder variant; report both from-scratch and frozen baselines |
| F4 | `RIDGE_CV_ALPHAS` ceiling silently truncated | `evaluate.py:362` is `logspace(-3, 3, 13)`, S1 needs to 1e6 | LinAlgWarning rcond=8.5e-8 | Bump to `logspace(-3, 6, 19)` |
| F5 | CLI default vs function default mismatch | `evaluate.py:1248` default=800 vs function arg default=2000 | Plan caveat to "bump to 2000" silently no-ops | Sync defaults |
| F6 | `RLIMIT_AS=15GB` silently rejected on macOS | `evaluate.py:50-55` `cap_process_memory` catches the OSError | Cap doesn't actually hold; OOM possible | Detect rejection, warn loudly; rely on `--max-eval-sequences` |
| F7 | `spatial_tokens=64` hardcode | `encoder.py:26` `assert spatial_tokens == 64` | Per-frame rank ceiling is `min(64, 128) = 64`; widening dim alone caps at 64 | Either lift `spatial_tokens` to 256 in Phase 4 OR document as un-addressed bottleneck |
| F8 | `temporal_pos` ↔ mask token interaction | `vjepa.py:43` adds temporal_pos AFTER encoder | If masked positions also get temporal_pos, predictor leaks frame identity through mask | Specify in Phase 2 design: masked positions receive `temporal_pos + mask_token` or zero pos |
| F9 | Concat-Ridge probe fundamentally cannot decode velocity | `evaluate.py:context_window_probe` | Negative R² is structural, not numerical | Switch to difference probe `pool_S(z_t) - pool_S(z_{t-1})` |
| F10 | VICReg-on-encoder suppresses high-frequency dims | `vjepa.py:compute_loss` with `reg_target=both` | Velocity (high-frequency) probed worse than position (low-frequency) by 30-40% per arXiv 2408.10864 | Move VICReg pressure to projector head (Phase 4); document expected vel-R² hit |
| F11 | Held-out regime drop is covariate shift in probe target | `evaluate.py:regime_split_indices` | -14pp on `random_velocities` is partly because target distribution shifts, not just encoder OOD | Standardize targets per-regime before probing; or report per-regime R² without claiming "encoder is brittle" |
| F12 | Predictor learns conditional mean at high mask ratio | observed: cos sim 0.075, rollout MAE 40× copy-last | Bayes-optimal for MSE-trained continuous predictor at 75% mask + small target variance | Switch to L1 + pre-encoder masking + per-mask-ratio inference monitor |

---

## 6. Plan-revision rationale summary (for future "why did we drop X?" questions)

| Choice | Why | Cite |
|---|---|---|
| Drop MEC | Reviewer B research curiosity (rare in V-JEPA lit; V-JEPA 2 didn't adopt despite known since 2022); Reviewer C logdet NaN risk on MPS float32 | B/C |
| Drop SIGReg | Reviewer B highest-uncertainty; M=1024 overkill at 10K samples; if revived use M=256 | B/S4 |
| Drop latent_dim 128→256 (default) | Reviewer B's ordering: pixel-patch masking is bigger lever; per Reviewer D's correction the per-frame ceiling is 64 not 128 so dim doubling doesn't lift the binding constraint; keep as contingent fallback per Reviewer A | A/B/D |
| Drop 4-knob arch lift | All of A/B/C/E flagged single-knob violation; iter-3 evidence (strategies_log.md) | A/B/C/E |
| Drop Phase 4c (LR/EMA reset) | Reviewer C self-contradiction: removes iter-3 machinery the plan §4 forbids | C |
| Adopt Phase 0 reliability fixes | Reviewer C NO-GO triggers; Reviewer D, E added items | C/D/E |
| Phase 5 → Phase 2 reorder | Reviewer B 3-reason argument: cross-paper consensus + iter-3 E4 evidence + Phase 6 prerequisite | B |
| Predictor cos-sim gate 0.40 → 0.25 falsification + 0.40 ship | Reviewer B no precedent for 0.40; Reviewer D per-mask-ratio sweep at inference | B/D |
| Add OCP + autoregressive rollout to evaluate.py BEFORE Phase 2 | Reviewer B "load-bearing extra requirement" for happy-path ship; Specialist S6 protocols | B/S6 |

---

## 7. Master citation list (deduplicated across all 15 reports)

**V-JEPA family:**
- V-JEPA (Bardes et al. 2024): https://arxiv.org/abs/2404.08471
- V-JEPA 2 (Assran et al. 2025): https://arxiv.org/abs/2506.09985
- V-JEPA 2.1 dense-loss ablation: https://arxiv.org/html/2603.14482v2
- Intuitive Physics Emerges from V-JEPA: https://arxiv.org/abs/2502.11831 + https://github.com/facebookresearch/jepa-intuitive-physics
- I-JEPA: https://arxiv.org/abs/2301.08243 + https://github.com/facebookresearch/ijepa

**JEPA variants:**
- LeJEPA / SIGReg: https://arxiv.org/abs/2511.08544
- LeWorldModel: https://arxiv.org/abs/2603.19312 + https://github.com/lucas-maes/le-wm
- C-JEPA: https://arxiv.org/abs/2410.19560
- MC-JEPA: https://arxiv.org/abs/2307.12698
- Var-JEPA: https://arxiv.org/pdf/2603.20111
- jepa-wms ("What drives success in physical planning"): https://arxiv.org/abs/2512.24497 + https://github.com/facebookresearch/jepa-wms
- Rethinking JEPA: Frozen Teachers: https://arxiv.org/abs/2509.24317

**SSL theory & regularization:**
- VICReg: https://arxiv.org/abs/2105.04906
- VICRegL: https://arxiv.org/abs/2210.01571
- MEC (Maximum Entropy Coding): https://arxiv.org/abs/2210.11464
- KoLeo (DINOv2/v3): https://github.com/facebookresearch/dinov2/blob/main/dinov2/loss/koleo_loss.py
- BYOL: https://arxiv.org/pdf/2006.07733
- MoCo: https://arxiv.org/abs/1911.05722
- DINOv2: https://arxiv.org/html/2304.07193v2
- DINOv3: https://arxiv.org/html/2508.10104v1
- Barlow Twins Optimal Representation Efficiency: https://arxiv.org/html/2510.10980
- RankMe: https://arxiv.org/pdf/2210.02885
- WERank: https://arxiv.org/pdf/2402.09586
- T-REGS: https://arxiv.org/pdf/2510.23484
- Orthogonality Regularization for Dimensional Collapse: https://arxiv.org/pdf/2411.00392
- Pros & Cons of Momentum Encoder: https://arxiv.org/pdf/2208.05744
- Apple ML Implicit Bias of Self-Distillation: https://machinelearning.apple.com/research/implicit-bias

**Probing methodology:**
- Alain & Bengio "Linear classifier probes": https://arxiv.org/abs/1610.01644
- Belinkov "Probing Classifiers": https://direct.mit.edu/coli/article/48/1/207/107571/Probing-Classifiers-Promises-Shortcomings-and
- Kobak et al. "Optimal ridge penalty can be zero/negative" (JMLR 2020): https://jmlr.org/papers/v21/19-844.html
- Bartlett "Benign overfitting in ridge regression": https://arxiv.org/pdf/2009.14286
- Attention, Please! Revisiting Attentive Probing: https://arxiv.org/abs/2506.10178
- Rethinking Evaluation Protocols of SSL Visual Representations: https://arxiv.org/abs/2304.03456
- A Closer Look at Benchmarking SSL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12289721/
- Unmute the Patch Tokens: https://arxiv.org/pdf/2509.24901
- Choice of Normalization in Regularized Regression: https://arxiv.org/pdf/2501.03821

**World-model evaluation:**
- IntPhys 2: https://arxiv.org/abs/2506.09849 + https://github.com/facebookresearch/IntPhys2
- Physion: https://physion-benchmark.github.io/ + https://arxiv.org/abs/2106.08261
- Physion++: https://arxiv.org/pdf/2306.15668
- PHYRE: https://ar5iv.labs.arxiv.org/html/1908.05656
- Counterfactual World Modeling: https://neuroailab.github.io/cwm-physics/
- DreamerV3 (Nature 2025): https://www.nature.com/articles/s41586-025-08744-2
- IRIS / Δ-IRIS: https://arxiv.org/html/2406.01361v1
- GAIA-2: https://arxiv.org/abs/2503.20523 + https://wayve.ai/thinking/gaia-2/
- Improving World Models with Linear Probes: https://arxiv.org/abs/2504.03861
- Light-weight probing of unsupervised representations for RL (RLC 2024): https://rlj.cs.umass.edu/2024/papers/RLJ_RLC_2024_242.pdf

**Video SSL:**
- VideoMAE: https://arxiv.org/abs/2203.12602
- VideoMAEv2: https://arxiv.org/abs/2303.16727
- SiamMAE: https://arxiv.org/abs/2305.14344
- MGM (Motion-Guided Masking): https://arxiv.org/abs/2308.10794
- InternVideo2: https://arxiv.org/abs/2403.15377
- Time-Equivariant Contrastive Learning: https://arxiv.org/abs/2207.04050
- Garrido "Learning world models in visual representation learning": https://arxiv.org/abs/2403.00504

**Architecture:**
- Geometry of Projection Heads: https://arxiv.org/html/2605.17180
- Projection Head as Information Bottleneck: https://arxiv.org/html/2503.00507
- Expand or Narrow your representation: https://arxiv.org/pdf/2304.05369
- DiT / AdaLN-Zero: https://arxiv.org/html/2312.04557v1
- AdaDim: https://arxiv.org/pdf/2505.12576

**Small-scale JEPA reproductions (GitHub):**
- lucas-maes/le-wm: https://github.com/lucas-maes/le-wm
- facebookresearch/jepa-wms: https://github.com/facebookresearch/jepa-wms
- keon/jepa: https://github.com/keon/jepa
- facebookresearch/eb_jepa: https://github.com/facebookresearch/eb_jepa
- jonwiggins/H-JEPA: https://github.com/jonwiggins/H-JEPA
- filipbasara0/simple-ijepa: https://github.com/filipbasara0/simple-ijepa
- facebookresearch/jepa (canonical V-JEPA): https://github.com/facebookresearch/jepa
- facebookresearch/ijepa (canonical I-JEPA): https://github.com/facebookresearch/ijepa
