"""Image formatting helpers for the UI, contact sheets and CLI reports."""
from __future__ import annotations

import io

import cv2
import numpy as np

from .color import linear_to_srgb


def to_u8_srgb(lin_rgb: np.ndarray) -> np.ndarray:
    s = np.clip(linear_to_srgb(np.asarray(lin_rgb, dtype=np.float64)), 0.0, 1.0)
    return np.round(s * 255.0).astype(np.uint8)


def u8_to_lin(u8_rgb: np.ndarray) -> np.ndarray:
    from .color import srgb_to_linear
    return srgb_to_linear(np.asarray(u8_rgb, dtype=np.float64) / 255.0)


def scalar_map(x: np.ndarray, vmin=0.0, vmax=1.0, cmap=cv2.COLORMAP_TURBO) -> np.ndarray:
    """Scalar field -> RGB uint8 image (output is RGB, not OpenCV BGR)."""
    t = np.clip((np.asarray(x, dtype=np.float64) - vmin) / max(vmax - vmin, 1e-9), 0.0, 1.0)
    bgr = cv2.applyColorMap((t * 255).astype(np.uint8), cmap)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def checkerboard(h, w, cell=16, a=0.60, b=0.78):
    yy, xx = np.mgrid[0:h, 0:w]
    c = ((yy // cell) + (xx // cell)) % 2
    g = np.where(c == 1, a, b).astype(np.float64)
    return np.repeat(g[..., None], 3, axis=2)


def composite_srgb(alpha, P_lin, bg_lin):
    """Standard source-over in linear light."""
    a = np.asarray(alpha, dtype=np.float64)[..., None]
    return a * np.asarray(P_lin, dtype=np.float64) + (1 - a) * np.asarray(bg_lin, dtype=np.float64)


def render_shadow_on(bg_lin, t=None, q=None):
    """Apply extracted layer to a background plate: I = B*t + B*q."""
    B = np.asarray(bg_lin, dtype=np.float64)
    if t is None:
        t = np.ones(B.shape[:2])
    if q is None:
        q = np.zeros_like(B)
    return B * np.asarray(t, dtype=np.float64)[..., None] + B * np.asarray(q, dtype=np.float64)


def png_bytes(u8_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(np.ascontiguousarray(u8_rgb), cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("png encode failed")
    return buf.tobytes()


def jpg_bytes(u8_rgb: np.ndarray, quality=88) -> bytes:
    ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(u8_rgb), cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpg encode failed")
    return buf.tobytes()


def contact_sheet(items, cols=5, cell_scale=1.0, label_height=26):
    """items: list of (label, u8_rgb_image).  All images should share size."""
    imgs = [(lab, cv2.resize(im, None, fx=cell_scale, fy=cell_scale,
                             interpolation=cv2.INTER_AREA if cell_scale < 1 else cv2.INTER_NEAREST))
            for lab, im in items]
    h, w = imgs[0][1].shape[:2]
    rows = (len(imgs) + cols - 1) // cols
    sheet = np.full((rows * (h + label_height), cols * w, 3), 24, dtype=np.uint8)
    for i, (lab, im) in enumerate(imgs):
        r, c = divmod(i, cols)
        y = r * (h + label_height)
        x = c * w
        sheet[y + label_height:y + label_height + h, x:x + w] = im[:h, :w]
        cv2.putText(sheet, lab, (x + 6, y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    return sheet


def error_map(pred, gt, scale=1.0, symmetric=True):
    """Visualize error between two scalar or RGB fields."""
    d = np.abs(np.asarray(pred, float) - np.asarray(gt, float))
    if d.ndim == 3:
        d = d.mean(axis=2)
    if symmetric:
        vmax = max(float(np.percentile(d, 99.5)), 1e-6)
        vmin = 0.0
    else:
        vmin, vmax = -scale, scale
    return scalar_map(d, vmin, vmax, cv2.COLORMAP_MAGMA)
