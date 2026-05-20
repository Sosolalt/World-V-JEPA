"""2D billiard physics in normalized units using pymunk.

Table is the rectangle [0, table_width] x [0, table_height] with no pockets.
All quantities are in normalized table units; the table is 1.0 x 0.5 by default.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import numpy as np
import pymunk


BALL_COLLISION_TYPE = 1
CUSHION_COLLISION_TYPE = 2


@dataclass
class BilliardParams:
    table_width: float = 1.0
    table_height: float = 0.5
    ball_radius: float = 0.03
    ball_mass: float = 0.170
    elasticity_ball: float = 0.92
    friction_ball: float = 0.05
    elasticity_cushion: float = 0.75
    friction_cushion: float = 0.15
    rolling_friction: float = 0.005
    velocity_threshold: float = 0.001
    physics_fps: int = 120
    substeps: int = 24


class BilliardWorld:
    """Stateful pymunk world. Construct, add balls, step, and query state."""

    def __init__(self, params: BilliardParams):
        self.params = params
        self.substeps = max(1, int(params.substeps))
        self.dt = 1.0 / float(params.physics_fps)
        self.sub_dt = self.dt / float(self.substeps)

        self.space = pymunk.Space()
        self.space.gravity = (0.0, 0.0)
        self.space.damping = 1.0
        self.space.iterations = 20
        self.space.collision_slop = 1e-4

        self._build_cushions()
        self.balls: List[pymunk.Body] = []
        self.shapes: List[pymunk.Shape] = []

    def _build_cushions(self) -> None:
        p = self.params
        w, h = p.table_width, p.table_height
        static = self.space.static_body
        thickness = 0.01
        segments = [
            ((0.0, 0.0), (w, 0.0)),
            ((w, 0.0), (w, h)),
            ((w, h), (0.0, h)),
            ((0.0, h), (0.0, 0.0)),
        ]
        for a, b in segments:
            seg = pymunk.Segment(static, a, b, thickness)
            seg.elasticity = p.elasticity_cushion
            seg.friction = p.friction_cushion
            seg.collision_type = CUSHION_COLLISION_TYPE
            self.space.add(seg)

    def add_ball(self, position: Tuple[float, float], velocity: Tuple[float, float]) -> int:
        p = self.params
        moment = pymunk.moment_for_circle(p.ball_mass, 0.0, p.ball_radius)
        body = pymunk.Body(p.ball_mass, moment)
        body.position = position
        body.velocity = velocity
        shape = pymunk.Circle(body, p.ball_radius)
        shape.elasticity = p.elasticity_ball
        shape.friction = p.friction_ball
        shape.collision_type = BALL_COLLISION_TYPE
        self.space.add(body, shape)
        idx = len(self.balls)
        self.balls.append(body)
        self.shapes.append(shape)
        return idx

    def reset(self) -> None:
        for body, shape in zip(self.balls, self.shapes):
            self.space.remove(body, shape)
        self.balls.clear()
        self.shapes.clear()

    def step(self, n: int = 1) -> None:
        p = self.params
        rf = p.rolling_friction
        thr = p.velocity_threshold
        dt = self.dt
        sub_dt = self.sub_dt
        substeps = self.substeps
        for _ in range(n):
            for body in self.balls:
                vx, vy = body.velocity
                speed = (vx * vx + vy * vy) ** 0.5
                if speed < thr:
                    body.velocity = (0.0, 0.0)
                    continue
                inv = 1.0 / speed
                ax = -rf * vx * inv
                ay = -rf * vy * inv
                new_vx = vx + ax * dt
                new_vy = vy + ay * dt
                if new_vx * vx + new_vy * vy <= 0.0:
                    body.velocity = (0.0, 0.0)
                else:
                    body.velocity = (new_vx, new_vy)
            for _ in range(substeps):
                self.space.step(sub_dt)
            for body in self.balls:
                vx, vy = body.velocity
                if (vx * vx + vy * vy) ** 0.5 < thr:
                    body.velocity = (0.0, 0.0)

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        n = len(self.balls)
        positions = np.zeros((n, 2), dtype=np.float32)
        velocities = np.zeros((n, 2), dtype=np.float32)
        for i, body in enumerate(self.balls):
            positions[i, 0] = body.position.x
            positions[i, 1] = body.position.y
            velocities[i, 0] = body.velocity.x
            velocities[i, 1] = body.velocity.y
        return positions, velocities

    def total_kinetic_energy(self) -> float:
        m = self.params.ball_mass
        total = 0.0
        for body in self.balls:
            vx, vy = body.velocity
            total += 0.5 * m * (vx * vx + vy * vy)
        return total

    def total_momentum(self) -> Tuple[float, float]:
        m = self.params.ball_mass
        px = 0.0
        py = 0.0
        for body in self.balls:
            px += m * body.velocity.x
            py += m * body.velocity.y
        return px, py
