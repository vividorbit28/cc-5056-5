"""Colour-space helpers.

All physical estimation happens in linear light.  Input PNGs are sRGB-encoded;
working in gamma space makes a multiplicative shadow ``I = t B`` look like
``I' = t^(1/2.2) B'`` and biases every ratio, so conversions are explicit and
centralised here.
"""
from __future__ import annotations

import numpy as np

_SRGB_LO = 0.04045
_SRGB_KNEE = 12.92


def srgb_to_linear(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x <= _SRGB_LO, x / _SRGB_KNEE,
                    ((np.clip(x, 0.0, None) + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(x: np.ndarray) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=np.float64), 0.0, None)
    return np.where(x <= 0.0031308, x * _SRGB_KNEE,
                    1.055 * x ** (1.0 / 2.4) - 0.055)


def luminance(rgb: np.ndarray, weights=(0.2126, 0.7152, 0.0722)) -> np.ndarray:
    """Linear relative luminance of an (..., 3) array."""
    rgb = np.asarray(rgb, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    return np.tensordot(rgb, w, axes=([-1], [0]))


def rgb_to_gray(rgb: np.ndarray) -> np.ndarray:
    return luminance(rgb)
