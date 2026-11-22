"""Procedural table plates.

A plate is a linear-light reflectance map ``beta(x, y, z)`` in [0, 1] with three
channels.  The generator returns the reflectance *and* the plate painted on a
receiving surface for a specific irradiance, so the caller can produce both the
empty plate and the "second plate" used for transfer tests.

Texture families deliberately include high-frequency structure: a shadow
extractor that merely re-draws a dark blob is indistinguishable from one that
copies texture; a true occlusion layer must be texture-free.
"""
from __future__ import annotations

import numpy as np

from .geometry import Scene, Camera


def _fbm(shape, rng, octaves=5, persistence=0.5, lacunarity=2.0, base=4):
    h, w = shape
    out = np.zeros(shape)
    amp = 1.0
    total = 0.0
    freq = base
    for _ in range(octaves):
        small = rng.normal(size=(max(2, int(h / freq * freq / h * h / h * freq + 2)), 2))
        # simpler: generate low-res noise then resize with cv2
        freq = int(max(2, freq))
        low = rng.normal(size=(freq + 1, int(freq * w / h) + 1))
        from cv2 import resize, INTER_CUBIC
        up = resize(low, (w, h), interpolation=INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= persistence
    return out / total


def smooth_noise(shape, rng, scale, octaves=4, persistence=0.55):
    """Value-noise fBm with a controllable base scale (pixels)."""
    import cv2
    h, w = shape
    out = np.zeros((h, w), dtype=np.float64)
    amp = 1.0
    total = 0.0
    s = float(scale)
    for _ in range(octaves):
        gh = max(2, int(round(h / s)) + 1)
        gw = max(2, int(round(w / s)) + 1)
        g = rng.normal(size=(gh, gw))
        up = cv2.resize(g, (w, h), interpolation=cv2.INTER_CUBIC)
        out += amp * up
        total += amp
        amp *= persistence
        s = max(1.2, s * 0.5)
    out /= max(total, 1e-9)
    return out


def _norm01(x):
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    if hi - lo < 1e-9:
        return np.full_like(x, 0.5)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def make_plate(shape, rng, kind="paper", base_rgb=(0.62, 0.58, 0.52), contrast=0.22):
    """Return linear reflectance map in [0,1], shape (h, w, 3)."""
    h, w = shape
    base = np.asarray(base_rgb, dtype=np.float64)

    if kind == "paper":
        low = _norm01(smooth_noise(shape, rng, 140, octaves=4))
        grain = _norm01(smooth_noise(shape, rng, 2.2, octaves=2))
        struct = 0.6 * low + 0.4 * grain
    elif kind == "wood":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        warp = smooth_noise(shape, rng, 90, octaves=4) * 6.0
        rings = 0.5 + 0.5 * np.sin((xx * 0.045 + warp) * np.pi * 2.0)
        fine = _norm01(smooth_noise(shape, rng, 3.0, octaves=2))
        struct = 0.72 * rings + 0.28 * fine
    elif kind == "marble":
        warp = smooth_noise(shape, rng, 120, octaves=5) * 3.0
        veins = np.abs(np.sin((smooth_noise(shape, rng, 60, octaves=5) + warp * 0.3) * 6.0) + 3.0 * (smooth_noise(shape, rng, 40) + 0.4))
        struct = _norm01(1.0 / (1.0 + veins))
        grain = _norm01(smooth_noise(shape, rng, 2.0))
        struct = 0.8 * struct + 0.2 * grain
    elif kind == "fabric":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
        weave = 0.5 + 0.5 * np.sin(xx * 2.1) * np.sin(yy * 2.1)
        fuzz = _norm01(smooth_noise(shape, rng, 2.5, octaves=2))
        struct = 0.55 * weave + 0.45 * fuzz
        struct = _norm01(smooth_noise(shape, rng, 5.0, octaves=3) * 0.4 + struct)
    elif kind == "flat":
        struct = np.full(shape, 0.5)
    else:
        raise ValueError(kind)

    tint = 1.0 + contrast * 0.35 * np.stack([smooth_noise(shape, rng, 200), smooth_noise(shape, rng, 200), smooth_noise(shape, rng, 200)], axis=2) * 0.15
    plate = base[None, None, :] * (1.0 + contrast * (struct[..., None] - 0.5) * 2.0) * tint
    return np.clip(plate, 0.01, 1.0)
