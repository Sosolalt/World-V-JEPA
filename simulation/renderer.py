"""Stateless renderer for billiard states to RGB numpy arrays.

Color convention: returned arrays are RGB (not BGR). cv2 operates on the array
without color conversion; the channel ordering is decided at render time by
choosing tuple values that match the RGB interpretation.
"""
from __future__ import annotations

from typing import Sequence, Tuple

import cv2
import numpy as np


# RGB tuples.
FELT_COLOR = (34, 139, 34)
BORDER_COLOR = (101, 67, 33)

BALL_COLORS = [
    (255, 255, 255),  # white (cue)
    (220, 20, 60),    # red
    (30, 90, 220),    # blue
    (30, 200, 80),    # green
    (255, 140, 0),    # orange
    (160, 60, 200),   # violet
    (240, 220, 30),   # yellow
    (139, 69, 19),    # brown
    (20, 20, 20),     # black
    (0, 200, 200),    # cyan
    (255, 105, 180),  # pink
    (160, 200, 30),   # lime
    (200, 30, 100),   # magenta
    (80, 80, 80),     # grey
    (200, 160, 60),   # gold
]


def _ball_color(i: int) -> Tuple[int, int, int]:
    return BALL_COLORS[i % len(BALL_COLORS)]


def render_frame(
    positions: np.ndarray,
    frame_size: int = 64,
    table_width: float = 1.0,
    table_height: float = 0.5,
    ball_radius: float = 0.03,
    border_px: int = 2,
) -> np.ndarray:
    """Render a single frame given ball positions in table units.

    Returns an (H, W, 3) uint8 RGB array, where H = frame_size // 2 if the
    table is 2:1, but to keep a square 64x64 we letterbox: the felt occupies
    the central band, top/bottom margins are dark borders.
    """
    H = W = int(frame_size)
    img = np.zeros((H, W, 3), dtype=np.uint8)

    aspect_table = table_width / table_height
    aspect_frame = W / H
    if aspect_table >= aspect_frame:
        felt_w = W
        felt_h = int(round(W / aspect_table))
    else:
        felt_h = H
        felt_w = int(round(H * aspect_table))
    x0 = (W - felt_w) // 2
    y0 = (H - felt_h) // 2

    cv2.rectangle(img, (0, 0), (W - 1, H - 1), BORDER_COLOR, thickness=-1)
    cv2.rectangle(img, (x0, y0), (x0 + felt_w - 1, y0 + felt_h - 1), FELT_COLOR, thickness=-1)

    sx = felt_w / table_width
    sy = felt_h / table_height
    r_px = max(3, int(round(ball_radius * 0.5 * (sx + sy))))

    n = positions.shape[0]
    for i in range(n):
        x_t, y_t = float(positions[i, 0]), float(positions[i, 1])
        px = x0 + int(round(x_t * sx))
        py = y0 + felt_h - 1 - int(round(y_t * sy))
        if 0 <= px < W and 0 <= py < H:
            cv2.circle(img, (px, py), r_px, (0, 0, 0), thickness=1, lineType=cv2.LINE_AA)
            cv2.circle(img, (px, py), r_px - 1, _ball_color(i), thickness=-1, lineType=cv2.LINE_AA)

    if border_px > 1:
        cv2.rectangle(
            img, (x0, y0), (x0 + felt_w - 1, y0 + felt_h - 1), BORDER_COLOR, thickness=border_px - 1
        )

    return img
