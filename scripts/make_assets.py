"""Regenerate the architecture diagram and the simulation GIF.

Outputs:
- assets/architecture.png  (matplotlib boxes + arrows, deterministic)
- assets/simulation.gif    (one 32-frame `break` sequence, ~10 FPS)

Both are reproducible from the same code so the README never references stale
artifacts.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulation.generator import (  # noqa: E402
    FRAME_STRIDE,
    _init_break,
    _make_world,
    _simulate_sequence,
    GenConfig,
)


ASSETS = REPO_ROOT / "assets"


def make_architecture(out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x, y, w, h, text, color):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            linewidth=1.4, edgecolor="black", facecolor=color,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=10, family="monospace")

    def arrow(x1, y1, x2, y2, label=None):
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="->", lw=1.4, color="black"),
        )
        if label:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2,
                    label, fontsize=8, family="monospace", color="gray")

    box(2.0, 8.6, 6.0, 0.9, "Context (4 frames)  [B, 4, 3, 64, 64]", "#e8f0fe")
    box(0.4, 6.8, 4.6, 1.1,
        "Context Encoder f_theta\nCNN + 1x spatial self-attn", "#cfe1ff")
    box(0.4, 5.0, 4.6, 1.1,
        "Temporal aggregation\n4 x 64 tokens + pos emb", "#cfe1ff")
    box(0.4, 3.0, 4.6, 1.3,
        "Predictor g_phi\n3-layer Transformer\n+ delta_t token", "#aac8ff")
    box(0.4, 1.2, 4.6, 0.9, "z_pred  [B, 64, 128]", "#dceaff")

    box(5.4, 8.6, 4.0, 0.9,
        "Future frame  (frame t+delta_t)", "#e8f0fe")
    box(5.4, 6.4, 4.0, 1.6,
        "Target encoder f_theta_bar\n(EMA copy)\ntau: 0.996 -> 1.0\nstop-gradient", "#ffe5b4")
    box(5.4, 4.6, 4.0, 0.9, "z_target  [B, 64, 128]", "#ffeccc")

    box(3.0, 0.05, 4.0, 0.85,
        "Loss = MSE(z_pred, z_target.detach())\n         + var_reg(z_pred)",
        "#ffd7d7")

    arrow(5.0, 8.6, 3.0, 7.9)
    arrow(2.7, 6.8, 2.7, 6.1)
    arrow(2.7, 5.0, 2.7, 4.3)
    arrow(2.7, 3.0, 2.7, 2.1)

    arrow(7.4, 8.6, 7.4, 8.0)
    arrow(7.4, 6.4, 7.4, 5.5)

    arrow(2.7, 1.2, 4.2, 0.9)
    arrow(7.4, 4.6, 5.8, 0.9)

    ax.annotate(
        "", xy=(7.4, 7.4), xytext=(4.6, 7.4),
        arrowprops=dict(arrowstyle="->", lw=1.2, color="#888", linestyle="dashed"),
    )
    ax.text(5.7, 7.55, "EMA copy of weights",
            fontsize=8, color="#666", family="monospace")

    ax.set_title("Mini V-JEPA — predict latents, not pixels", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_simulation_gif(out_path: Path, seed: int = 7) -> None:
    cfg = GenConfig(
        n_balls=9,
        n_sequences=1,
        sequence_length=32,
        frame_size=64,
        table_ratio=(2, 1),
        ball_radius=0.03,
        physics_fps=120,
        init_strategies={"break": 1.0},
        output_path="",
        seed=seed,
    )
    world, w, h = _make_world(cfg)
    world.reset()
    rng = random.Random(seed)
    _init_break(world, w, h, cfg.n_balls, rng)
    frames, _, _ = _simulate_sequence(world, cfg, rng)

    upscale = 4
    pil_frames = []
    for f in frames:
        img = Image.fromarray(f)
        img = img.resize(
            (f.shape[1] * upscale, f.shape[0] * upscale),
            resample=Image.NEAREST,
        )
        pil_frames.append(img)

    duration_ms = int(1000 / (cfg.physics_fps / FRAME_STRIDE))
    pil_frames[0].save(
        out_path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=duration_ms,
        optimize=False,
        disposal=2,
    )
    print(f"wrote {out_path} ({len(pil_frames)} frames, {duration_ms} ms/frame)")


def main() -> int:
    ASSETS.mkdir(parents=True, exist_ok=True)
    make_architecture(ASSETS / "architecture.png")
    make_simulation_gif(ASSETS / "simulation.gif")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
