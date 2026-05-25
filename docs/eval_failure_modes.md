# Evaluation pipeline failure modes (V1)

Twelve bugs in the V1 evaluation and training pipelines, catalogued during the
iter-2 consultation (15 agents, 2026-05-24). Load-bearing for any future change
to `scripts/evaluate.py`, `scripts/train.py`, or `mini_vjepa/vjepa.py`.

Each row is a claim about the codebase that **MUST be verified against the
current code before any plan touches it** — the codebase changes; this document
is point-in-time.

| # | Failure mode | File:line evidence (verify) | Effect | Fix per PLAN_V2.md |
|---|---|---|---|---|
| F1 | **Inference-time mask leak** | `scripts/evaluate.py:546, 669` (degradation_curve, rollout_position_error) read `cfg.training.mask_ratio` = 0.75 at eval time | Every V1 cos-sim and rollout number was measured with training-time mask still active. Cos sim 0.075 is partially this artifact, partially the predictor. | Phase 0: add `--eval-mask-ratio` CLI flag default 0.0 |
| F2 | **`ctx_effective_rank` 1-batch measurement** | `scripts/train.py:341-364` uses last batch only | rank=12 diagnosis driving all architecture work rests on single batch — may be ±5 noise | Phase 0: compute over ≥10 batches per epoch, log mean±std |
| F3 | **Pixel baseline shares V-JEPA encoder architecture** | `baselines/pixel_predictor.py:84` re-instantiates `ContextEncoder` from `mini_vjepa.encoder` | "V-JEPA loses to pixel" is comparing two encoders of identical capacity trained on different objectives — not an independent baseline | Phase 0: add frozen-V1-encoder variant; report both from-scratch and frozen baselines |
| F4 | **`RIDGE_CV_ALPHAS` ceiling silently truncated** | `scripts/evaluate.py:362` is `logspace(-3, 3, 13)`, S1 needs to 1e6 | LinAlgWarning rcond=8.5e-8 during Ridge fit; α grid silently truncated | Phase 0: bump to `logspace(-3, 6, 19)` |
| F5 | **CLI default vs function default mismatch** | `scripts/evaluate.py:1248` default=800 vs function signature arg default=2000 | Plan caveat to "bump to 2000" silently no-ops; we ran at N=1500 not 2000 today | Phase 0: sync defaults |
| F6 | **`RLIMIT_AS=15GB` silently rejected on macOS** | `scripts/evaluate.py:50-55` `cap_process_memory` catches OSError ("current limit exceeds maximum limit") | Hard RAM cap doesn't actually hold; relies entirely on `--max-eval-sequences` | Phase 0: detect rejection, warn loudly; rely on `--max-eval-sequences` |
| F7 | **`spatial_tokens=64` hardcode** | `mini_vjepa/encoder.py:26` `assert spatial_tokens == 64` | Per-frame rank ceiling is `min(64, 128) = 64`; widening `latent_dim` alone caps at 64; the 8×8 grid IS the ceiling per Reviewer C | Phase 4 docs flag this; possibly lift `spatial_tokens` to 256 in a later iteration |
| F8 | **`temporal_pos` ↔ mask token interaction** | `mini_vjepa/vjepa.py:43` adds `temporal_pos` AFTER encoder | If Phase 2 masks before encoder and masked positions also receive `temporal_pos`, predictor leaks frame identity through the mask | Phase 2 design: specify masked positions receive `temporal_pos + mask_token` or zero pos |
| F9 | **Concat-Ridge probe cannot decode velocity** | `scripts/evaluate.py:context_window_probe` flattens (4, 64, 128) → 32K features | Negative R² is structural — V-JEPA encoder does NOT encode velocity per-token (per [Garrido et al. arXiv 2403.00504](https://arxiv.org/abs/2403.00504)). Velocity lives in temporal derivative. | Phase 1-bis: switch to difference probe `pool_S(z_t) - pool_S(z_{t-1})` |
| F10 | **VICReg-on-encoder suppresses high-frequency dims** | `mini_vjepa/vjepa.py:compute_loss` with `reg_target=both` | Velocity (high-frequency) probed worse than position (low-frequency) by 30-40% per [Aubret et al. arXiv 2408.10864](https://arxiv.org/abs/2408.10864). VICReg's variance term clamps per-dim std to ≥1 across the batch, mixing time steps. | Phase 4: move VICReg pressure to projector head; document expected vel-R² hit |
| F11 | **Held-out regime drop is partly covariate shift** | `scripts/evaluate.py:regime_split_indices` splits by `init_strategy` | Train set biased toward `break`/`midgame_*` (balls cluster near table center); `random_velocities` has more uniform spatial coverage. -14pp drop is partly because target distribution shifts, not just encoder OOD. | Plan note only — not actionable; report per-regime R² without claiming "encoder is brittle" |
| F12 | **Predictor learns conditional mean at high mask ratio** | observed: cos sim 0.075, rollout MAE 40× copy-last | Bayes-optimal for MSE-trained continuous predictor at 75% mask + small target variance per [DreamerV3 Nature 2025](https://www.nature.com/articles/s41586-025-08744-2) framing. Textbook collapse. | Phase 2: switch to L1 + pre-encoder masking + per-mask-ratio inference monitor |

**Cross-cutting recommendation per [PLAN_V2.md](../PLAN_V2.md) §3 Phase 0:** patch
F1, F2, F3, F4, F5, F6 atomically before ANY downstream phase. F7 + F8 must be
decided in Phase 2's design. F9 + F10 + F12 are addressed by Phase 1-bis (probe
rebuild) and Phase 2 (pixel-patch masking + L1). F11 is acknowledged but not
actionable.

**How to apply:** before changing `scripts/evaluate.py`, `scripts/train.py`, or
`mini_vjepa/vjepa.py`, check this list. If a change might touch one of these
failure modes, follow the prescribed fix or document why the fix is
intentionally not applied.

**Related:** [PLAN_V2.md](../PLAN_V2.md) (the rev-2 plan that addresses these),
[assets/results/v2_phase1/metrics.json](../assets/results/v2_phase1/metrics.json)
(the raw numbers these failure modes produced).
