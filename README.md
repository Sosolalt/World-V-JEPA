# Mini V-JEPA: Learning World Models for 2D Physics

> A from-scratch implementation of V-JEPA that learns to predict billiard
> dynamics in **latent space** — no pixel reconstruction, no generative decoder.

**Status:** 🚧 Work in progress. See [PLAN_Mini_V-JEPA.md](PLAN_Mini_V-JEPA.md) for the full roadmap.

---

## Why JEPA, not generative?

A single frame of billiards contains only positions. To predict the future, a
model must understand *velocities* — which requires a temporal context. Generative
video models spend most of their capacity reconstructing pixels (felt texture,
lighting, ball shading) instead of the underlying dynamics.

V-JEPA flips this: the model predicts a **representation** of the future frame
rather than the frame itself. Pixel-level detail is discarded; the predictive
signal is forced into the part of the latent space that actually encodes
physics.

This repo implements a small (~5–8 M parameters), reproducible version of that
idea on a 2D billiard simulator. The goal is not to scale — it's to demonstrate
the paradigm cleanly enough to be measurable and inspectable on a single
laptop.

---

## Architecture (overview)

```
Context (4 frames)
        │
        ▼
Context Encoder f_θ  (CNN + 1× spatial self-attention, per frame)
        │
        ▼
Temporal aggregation  (4 × 64 tokens + temporal pos. embeddings)
        │
        ▼
Predictor g_φ  (3-layer Transformer + Δt time token)
        │
        ▼
                  ─── MSE + variance regularization ───
        ▲
        │
Target Encoder f_θ̄  (EMA of f_θ, stop-gradient, τ: 0.996 → 1.0)
        ▲
        │
Future frame (frame t + Δt)
```

| Component | Output shape | Notes |
|-----------|--------------|-------|
| Encoder (per frame) | `(B, 64, 128)` | 64 spatial tokens, dim 128 |
| Temporal aggregation | `(B, 256, 128)` | 4 frames × 64 tokens |
| Predictor | `(B, 64, 128)` | Predicted future tokens |
| Target encoder | `(B, 64, 128)` | EMA copy, no backprop |

Full hyperparameters and the data pipeline contract live in
[PLAN_Mini_V-JEPA.md](PLAN_Mini_V-JEPA.md).

---

## Data: 2D billiard simulation

A `pymunk`-based simulator generates 32-frame sequences of 9 balls bouncing on
a 2:1 frictioned table (no pockets, no spin — both invisible at 64×64). Rendered
headless with OpenCV.

| Config  | Balls | Sequences | Use |
|---------|-------|-----------|-----|
| simple  | 2–3   | 2 000     | Debug, sanity, scaling reference |
| default | 9     | 10 000    | Main training run |
| stress  | 15    | 1 000     | Generalization probe |

Frames are stored as `uint8` NPZ, with positions/velocities as `float32`
metadata for downstream linear probing.

---

## Repository layout

```
mini-vjepa/
├── configs/          # YAML configs: simple / default / stress
├── mini_vjepa/       # Model: encoder, predictor, EMA, masking, losses, dataset
├── simulation/       # pymunk physics + OpenCV renderer + data generator
├── scripts/          # CLI entrypoints: generate_data, train, evaluate
├── baselines/        # Pixel-prediction baseline
├── notebooks/        # Reproducible evaluation notebook
├── tests/            # Physics, dataset, model, training, evaluation
├── assets/           # Architecture diagram, GIFs, result figures
├── data/sample/      # 16 pre-generated sequences (committed)
├── PLAN_Mini_V-JEPA.md
├── DESIGN.md         # Design decisions & what didn't work
└── README.md
```

---

## Quick start

> Once the implementation lands. The commands below describe the intended
> interface; they will work as the corresponding modules get committed.

```bash
git clone https://github.com/<user>/mini-vjepa.git
cd mini-vjepa
pip install -r requirements.txt

# 1. Generate dataset (~3–5 min on M3)
python scripts/generate_data.py --config configs/default.yaml

# 2. Train (~2 h on M3 MPS)
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/train.py --config configs/default.yaml

# 3. Evaluate
python scripts/evaluate.py --checkpoint runs/latest/best.pt
```

---

## Results

*To be filled in once training completes.*

Headline metrics the project tracks:

- Linear probe R² on ball **positions** (target: > 0.7)
- Linear probe R² on ball **velocities** (target: > 0.5)
- Cosine similarity vs. prediction horizon (graceful degradation over 12 steps)
- JEPA vs. pixel-prediction baseline on downstream probing

Figures (PCA latent trajectory, t-SNE colored by velocity, training composite,
JEPA-vs-pixel 2×2) are regenerated end-to-end by
`notebooks/evaluation.ipynb` from a saved checkpoint.

---

## Design decisions

The non-obvious choices (variance regularization vs. covariance, EMA τ schedule,
explicit rolling friction, effective-rank-based checkpoint selection, MPS
constraints) are documented in `DESIGN.md` as the project progresses.

---

## References

| Paper | Role |
|-------|------|
| [V-JEPA (Bardes et al., 2024)](https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video-joint-embedding-predictive-architecture/) | Architecture reference |
| [V-JEPA 2 (Meta, 2025)](https://ai.meta.com/blog/v-jepa-2-world-model-benchmarks/) | World model + planning |
| [I-JEPA (Assran et al., 2023)](https://github.com/facebookresearch/ijepa) | Image predecessor |
| [A Path Towards Autonomous Machine Intelligence (LeCun, 2022)](https://openreview.net/pdf?id=BZ5a1r-kVsf) | Philosophical foundation |
| [LeWorldModel (2025)](https://le-wm.github.io/) | JEPA end-to-end without EMA |
| [VICReg (Bardes et al., 2021)](https://arxiv.org/abs/2105.04906) | Anti-collapse regularization |

---

## License

MIT.
