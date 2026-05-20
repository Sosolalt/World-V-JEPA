"""Dataset generator: simulate billiard sequences and save to NPZ.

Each sequence is `sequence_length` frames sampled every `frame_stride` physics
steps. Positions and velocities are normalized so positions live in
[0, 1] x [0, 0.5] (table units) and velocities are in table-units/second.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import yaml
from tqdm import tqdm

from .physics import BilliardParams, BilliardWorld
from .renderer import render_frame


FRAME_STRIDE = 4  # physics ticks between saved frames -> 30 FPS at 120 Hz physics


@dataclass
class GenConfig:
    n_balls: int
    n_sequences: int
    sequence_length: int
    frame_size: int
    table_ratio: Tuple[int, int]
    ball_radius: float
    physics_fps: int
    init_strategies: Dict[str, float]
    output_path: str
    seed: int = 42


def load_config(path: str) -> Tuple[GenConfig, dict]:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    d = raw["data"]
    cfg = GenConfig(
        n_balls=int(d["n_balls"]),
        n_sequences=int(d["n_sequences"]),
        sequence_length=int(d["sequence_length"]),
        frame_size=int(d["frame_size"]),
        table_ratio=tuple(d["table_ratio"]),
        ball_radius=float(d["ball_radius"]),
        physics_fps=int(d["physics_fps"]),
        init_strategies=dict(d["init_strategies"]),
        output_path=str(d["output_path"]),
        seed=int(raw.get("training", {}).get("seed", 42)),
    )
    return cfg, raw


def _make_world(cfg: GenConfig) -> Tuple[BilliardWorld, float, float]:
    ratio_x, ratio_y = cfg.table_ratio
    table_width = 1.0
    table_height = float(ratio_y) / float(ratio_x)
    params = BilliardParams(
        table_width=table_width,
        table_height=table_height,
        ball_radius=cfg.ball_radius,
        physics_fps=cfg.physics_fps,
    )
    return BilliardWorld(params), table_width, table_height


def _project_clear(
    x: float,
    y: float,
    placed: List[Tuple[float, float]],
    w: float,
    h: float,
    r: float,
) -> Tuple[float, float]:
    """Push (x,y) outside every previously placed ball, clamped to table bounds.

    Iterative because pushing away from one ball can push into another. Falls
    back to a deterministic grid scan if iterative projection cannot find a
    clear spot inside the table.
    """
    margin = r * 1.5
    min_sep = 2.05 * r
    min_d2 = min_sep ** 2
    cx, cy = x, y
    for _ in range(64):
        worst_overlap = 0.0
        worst_idx = -1
        for i, (px, py) in enumerate(placed):
            dx = cx - px
            dy = cy - py
            d2 = dx * dx + dy * dy
            if d2 < min_d2:
                overlap = min_sep - math.sqrt(max(d2, 0.0))
                if overlap > worst_overlap:
                    worst_overlap = overlap
                    worst_idx = i
        if worst_idx < 0:
            cx = min(max(cx, margin), w - margin)
            cy = min(max(cy, margin), h - margin)
            ok = all((cx - px) ** 2 + (cy - py) ** 2 >= min_d2 for px, py in placed)
            if ok:
                return cx, cy
            break
        px, py = placed[worst_idx]
        dx = cx - px
        dy = cy - py
        dist = math.hypot(dx, dy)
        if dist < 1e-12:
            ang = 0.0
            dx, dy = 1.0, 0.0
            dist = 1.0
        push = (min_sep - dist) + 1e-6
        cx = cx + dx / dist * push
        cy = cy + dy / dist * push
        cx = min(max(cx, margin), w - margin)
        cy = min(max(cy, margin), h - margin)
    # Deterministic grid scan as last resort.
    steps = 64
    for j in range(steps + 1):
        gy = margin + (h - 2 * margin) * j / steps
        for i in range(steps + 1):
            gx = margin + (w - 2 * margin) * i / steps
            if all((gx - px) ** 2 + (gy - py) ** 2 >= min_d2 for px, py in placed):
                return gx, gy
    raise RuntimeError("no non-overlapping placement available on table")


def _random_positions_no_overlap(
    n: int,
    w: float,
    h: float,
    r: float,
    rng: random.Random,
    max_tries: int = 2000,
    existing: List[Tuple[float, float]] | None = None,
) -> List[Tuple[float, float]]:
    margin = r * 1.5
    existing = list(existing) if existing else []
    placed: List[Tuple[float, float]] = []
    min_d2 = (2.05 * r) ** 2
    for _ in range(n):
        last_x = last_y = 0.0
        for _ in range(max_tries):
            x = rng.uniform(margin, w - margin)
            y = rng.uniform(margin, h - margin)
            last_x, last_y = x, y
            ok = True
            for px, py in existing + placed:
                if (px - x) ** 2 + (py - y) ** 2 < min_d2:
                    ok = False
                    break
            if ok:
                placed.append((x, y))
                break
        else:
            px, py = _project_clear(last_x, last_y, existing + placed, w, h, r)
            placed.append((px, py))
    return placed


def _init_break(world: BilliardWorld, w: float, h: float, n: int, rng: random.Random) -> None:
    r = world.params.ball_radius
    apex_x = w * 0.7
    apex_y = h * 0.5
    spacing = 2.05 * r
    rack = []
    row = 0
    placed = 0
    needed = max(0, n - 1)
    while placed < needed:
        for k in range(row + 1):
            if placed >= needed:
                break
            x = apex_x + row * spacing * math.cos(math.radians(30)) * 1.0
            y = apex_y + (k - row * 0.5) * spacing
            x = min(max(x, 2 * r), w - 2 * r)
            y = min(max(y, 2 * r), h - 2 * r)
            rack.append((x, y))
            placed += 1
        row += 1
    cue_x = w * 0.2
    cue_y = h * 0.5 + rng.uniform(-h * 0.05, h * 0.05)
    speed = rng.uniform(3.0, 6.0)
    aim_x = apex_x - cue_x
    aim_y = apex_y - cue_y
    norm = math.hypot(aim_x, aim_y) + 1e-9
    vx = speed * aim_x / norm
    vy = speed * aim_y / norm
    world.add_ball((cue_x, cue_y), (vx, vy))
    for pos in rack:
        world.add_ball(pos, (0.0, 0.0))


def _init_midgame_cluster(world: BilliardWorld, w: float, h: float, n: int, rng: random.Random) -> None:
    r = world.params.ball_radius
    cluster_n = min(n, rng.randint(3, 5))
    cx = rng.uniform(w * 0.3, w * 0.7)
    cy = rng.uniform(h * 0.3, h * 0.7)
    cluster_positions: List[Tuple[float, float]] = []
    min_d2 = (2.05 * r) ** 2
    for _ in range(cluster_n):
        last_x = last_y = 0.0
        for _ in range(500):
            x = cx + rng.uniform(-3 * r, 3 * r)
            y = cy + rng.uniform(-3 * r, 3 * r)
            x = min(max(x, 2 * r), w - 2 * r)
            y = min(max(y, 2 * r), h - 2 * r)
            last_x, last_y = x, y
            ok = all((x - px) ** 2 + (y - py) ** 2 >= min_d2 for px, py in cluster_positions)
            if ok:
                cluster_positions.append((x, y))
                break
        else:
            px, py = _project_clear(last_x, last_y, cluster_positions, w, h, r)
            cluster_positions.append((px, py))
    remaining = n - len(cluster_positions)
    extras = (
        _random_positions_no_overlap(remaining, w, h, r, rng, existing=cluster_positions)
        if remaining > 0
        else []
    )
    all_pos = cluster_positions + extras
    moving_idx = rng.randrange(len(all_pos))
    for i, pos in enumerate(all_pos):
        if i == moving_idx:
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1.5, 3.5)
            vel = (speed * math.cos(angle), speed * math.sin(angle))
        else:
            vel = (0.0, 0.0)
        world.add_ball(pos, vel)


def _init_midgame_spread(world: BilliardWorld, w: float, h: float, n: int, rng: random.Random) -> None:
    r = world.params.ball_radius
    positions = _random_positions_no_overlap(n, w, h, r, rng)
    n_moving = min(n, rng.randint(1, 3))
    moving = set(rng.sample(range(len(positions)), n_moving))
    for i, pos in enumerate(positions):
        if i in moving:
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(1.0, 3.0)
            vel = (speed * math.cos(angle), speed * math.sin(angle))
        else:
            vel = (0.0, 0.0)
        world.add_ball(pos, vel)


def _init_two_ball(world: BilliardWorld, w: float, h: float, n: int, rng: random.Random) -> None:
    if n < 2:
        _init_random_velocities(world, w, h, n, rng)
        return
    r = world.params.ball_radius
    y_line = h * 0.5 + rng.uniform(-h * 0.1, h * 0.1)
    x_a = w * rng.uniform(0.15, 0.3)
    x_b = w * rng.uniform(0.6, 0.85)
    pos_a = (x_a, y_line)
    pos_b = (x_b, y_line + rng.uniform(-r * 0.5, r * 0.5))
    world.add_ball(pos_a, (rng.uniform(2.0, 4.0), rng.uniform(-0.2, 0.2)))
    world.add_ball(pos_b, (0.0, 0.0))
    if n > 2:
        extras = _random_positions_no_overlap(
            n - 2, w, h, r, rng, existing=[pos_a, pos_b]
        )
        for pos in extras:
            world.add_ball(pos, (0.0, 0.0))


def _init_random_velocities(world: BilliardWorld, w: float, h: float, n: int, rng: random.Random) -> None:
    r = world.params.ball_radius
    positions = _random_positions_no_overlap(n, w, h, r, rng)
    for pos in positions:
        if rng.random() < 0.4:
            angle = rng.uniform(0, 2 * math.pi)
            speed = rng.uniform(0.8, 3.0)
            vel = (speed * math.cos(angle), speed * math.sin(angle))
        else:
            vel = (0.0, 0.0)
        world.add_ball(pos, vel)


STRATEGY_FUNCS = {
    "break": _init_break,
    "midgame_cluster": _init_midgame_cluster,
    "midgame_spread": _init_midgame_spread,
    "two_ball": _init_two_ball,
    "random_velocities": _init_random_velocities,
}


def _sample_strategy(weights: Dict[str, float], rng: random.Random) -> str:
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=np.float64)
    w = w / w.sum()
    return names[int(rng.choices(range(len(names)), weights=w.tolist(), k=1)[0])]


def _simulate_sequence(
    world: BilliardWorld,
    cfg: GenConfig,
    rng: random.Random,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = cfg.sequence_length
    n = cfg.n_balls
    H = W = cfg.frame_size
    frames = np.zeros((T, H, W, 3), dtype=np.uint8)
    positions = np.zeros((T, n, 2), dtype=np.float32)
    velocities = np.zeros((T, n, 2), dtype=np.float32)

    table_w = world.params.table_width
    table_h = world.params.table_height

    for t in range(T):
        pos, vel = world.get_state()
        positions[t] = pos
        velocities[t] = vel
        frames[t] = render_frame(
            pos,
            frame_size=cfg.frame_size,
            table_width=table_w,
            table_height=table_h,
            ball_radius=world.params.ball_radius,
        )
        world.step(FRAME_STRIDE)
    return frames, positions, velocities


def _try_init(cfg: GenConfig, world: BilliardWorld, w: float, h: float, rng: random.Random) -> str:
    strategy = _sample_strategy(cfg.init_strategies, rng)
    world.reset()
    func = STRATEGY_FUNCS[strategy]
    func(world, w, h, cfg.n_balls, rng)
    while len(world.balls) > cfg.n_balls:
        body = world.balls.pop()
        shape = world.shapes.pop()
        world.space.remove(body, shape)
    while len(world.balls) < cfg.n_balls:
        r = world.params.ball_radius
        margin = r * 1.5
        min_d2 = (2.05 * r) ** 2
        last_x = last_y = margin
        for _ in range(2000):
            x = rng.uniform(margin, w - margin)
            y = rng.uniform(margin, h - margin)
            last_x, last_y = x, y
            ok = True
            for body in world.balls:
                if (body.position.x - x) ** 2 + (body.position.y - y) ** 2 < min_d2:
                    ok = False
                    break
            if ok:
                world.add_ball((x, y), (0.0, 0.0))
                break
        else:
            placed = [(b.position.x, b.position.y) for b in world.balls]
            px, py = _project_clear(last_x, last_y, placed, w, h, r)
            world.add_ball((px, py), (0.0, 0.0))
    return strategy


def generate_dataset(cfg: GenConfig, show_progress: bool = True) -> str:
    rng = random.Random(cfg.seed)
    np_rng = np.random.RandomState(cfg.seed)

    world, w, h = _make_world(cfg)

    N = cfg.n_sequences
    T = cfg.sequence_length
    H = W = cfg.frame_size
    n = cfg.n_balls

    frames_all = np.zeros((N, T, H, W, 3), dtype=np.uint8)
    positions_all = np.zeros((N, T, n, 2), dtype=np.float32)
    velocities_all = np.zeros((N, T, n, 2), dtype=np.float32)
    strategies: List[str] = []

    it = range(N)
    if show_progress:
        it = tqdm(it, desc="Generating sequences")
    for i in it:
        seq_seed = cfg.seed * 1_000_003 + i
        seq_rng = random.Random(seq_seed)
        strategy = _try_init(cfg, world, w, h, seq_rng)
        frames, positions, velocities = _simulate_sequence(world, cfg, seq_rng)
        frames_all[i] = frames
        positions_all[i] = positions
        velocities_all[i] = velocities
        strategies.append(strategy)

    out_dir = os.path.dirname(cfg.output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    cfg_dict = {
        "n_balls": cfg.n_balls,
        "n_sequences": cfg.n_sequences,
        "sequence_length": cfg.sequence_length,
        "frame_size": cfg.frame_size,
        "table_ratio": list(cfg.table_ratio),
        "ball_radius": cfg.ball_radius,
        "physics_fps": cfg.physics_fps,
        "init_strategies": cfg.init_strategies,
        "seed": cfg.seed,
        "frame_stride": FRAME_STRIDE,
    }
    cfg_hash = hashlib.sha1(json.dumps(cfg_dict, sort_keys=True).encode()).hexdigest()

    np.savez_compressed(
        cfg.output_path,
        frames=frames_all,
        positions=positions_all,
        velocities=velocities_all,
        strategies=np.array(strategies),
        config_json=np.array(json.dumps(cfg_dict)),
        config_hash=np.array(cfg_hash),
    )
    return cfg.output_path
