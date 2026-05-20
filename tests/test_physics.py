"""Physics validation: energy decay, momentum conservation, reflection law."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simulation.physics import BilliardParams, BilliardWorld


def _quiet_params(**overrides) -> BilliardParams:
    base = dict(
        table_width=1.0,
        table_height=0.5,
        ball_radius=0.03,
        ball_mass=0.170,
        elasticity_ball=0.92,
        friction_ball=0.05,
        elasticity_cushion=0.75,
        friction_cushion=0.15,
        rolling_friction=0.005,
        velocity_threshold=0.001,
        physics_fps=240,
    )
    base.update(overrides)
    return BilliardParams(**base)


def test_energy_decay_monotone_with_friction():
    world = BilliardWorld(_quiet_params())
    world.add_ball((0.5, 0.25), (1.0, 0.0))
    energies = [world.total_kinetic_energy()]
    for _ in range(60):
        world.step(20)
        energies.append(world.total_kinetic_energy())
    diffs = np.diff(np.array(energies))
    assert np.all(diffs <= 1e-9), f"energy increased somewhere: max delta = {diffs.max()}"
    assert energies[-1] < energies[0] * 0.95


def test_momentum_conserved_elastic_collision_pair():
    params = _quiet_params(
        rolling_friction=0.0,
        friction_ball=0.0,
        elasticity_ball=1.0,
        velocity_threshold=0.0,
    )
    world = BilliardWorld(params)
    world.space.collision_slop = 1e-4
    world.space.collision_bias = 0.0
    world.add_ball((0.3, 0.25), (1.0, 0.0))
    world.add_ball((0.7, 0.25), (-1.0, 0.0))
    p0 = world.total_momentum()
    ke0 = world.total_kinetic_energy()
    for _ in range(2000):
        world.step(1)
        vxs = [b.velocity.x for b in world.balls]
        if vxs[0] < 0 and vxs[1] > 0:
            break
    p1 = world.total_momentum()
    ke1 = world.total_kinetic_energy()
    assert abs(p1[0] - p0[0]) < 1e-3, f"px not conserved: {p0[0]} -> {p1[0]}"
    assert abs(p1[1] - p0[1]) < 1e-3, f"py not conserved: {p0[1]} -> {p1[1]}"
    assert abs(ke1 - ke0) / max(ke0, 1e-9) < 0.1


def test_reflection_law_against_cushion():
    params = _quiet_params(
        rolling_friction=0.0,
        friction_cushion=0.0,
        elasticity_cushion=1.0,
        elasticity_ball=1.0,
        velocity_threshold=0.0,
    )
    world = BilliardWorld(params)
    v_in = (2.0, 1.0)
    world.add_ball((0.1, 0.25), v_in)
    pre_v = None
    post_v = None
    for _ in range(400):
        vx_before, vy_before = world.balls[0].velocity
        world.step(1)
        vx_after, vy_after = world.balls[0].velocity
        if vx_before > 0 and vx_after < 0 and pre_v is None:
            pre_v = (vx_before, vy_before)
            post_v = (vx_after, vy_after)
            break
    assert pre_v is not None, "no cushion bounce detected"
    angle_in = math.atan2(abs(pre_v[1]), abs(pre_v[0]))
    angle_out = math.atan2(abs(post_v[1]), abs(post_v[0]))
    assert abs(angle_in - angle_out) < math.radians(5), (
        f"reflection law violated: {math.degrees(angle_in):.2f} vs {math.degrees(angle_out):.2f}"
    )
    assert abs(pre_v[1] - post_v[1]) < 5e-2
    rel_err = abs(abs(pre_v[0]) - abs(post_v[0])) / max(abs(pre_v[0]), 1e-9)
    assert rel_err < 0.15, f"|vx| changed too much: {pre_v[0]} -> {post_v[0]}"


def test_velocity_snap_to_zero():
    params = _quiet_params(rolling_friction=0.5, velocity_threshold=0.01)
    world = BilliardWorld(params)
    world.add_ball((0.5, 0.25), (0.05, 0.0))
    for _ in range(2000):
        world.step(1)
        vx, vy = world.balls[0].velocity
        if vx == 0.0 and vy == 0.0:
            break
    vx, vy = world.balls[0].velocity
    assert vx == 0.0 and vy == 0.0, f"ball did not stop, v={vx, vy}"
