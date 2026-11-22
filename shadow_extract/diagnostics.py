"""Diagnose differences between the empty-plate and product captures.

Two images from the same fixed camera still differ in ways that must be
identified *before* any shadow is interpreted:

* geometric misalignment (tripod bump, lens breathing),
* exposure / white-balance drift,
* focus or motion blur mismatch,
* sensor noise floor,
* genuine changes of the background itself (creases, new objects, texture).

A shadow is only inferred from differences that can be explained by a smooth,
non-positive multiplicative attenuation plus (bounded) additive light.  The
report produced here is therefore a gate: if the background changed, or the
pair cannot be aligned, the tool says so instead of calling the change a
shadow.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np
from skimage.restoration import estimate_sigma

from .color import luminance


@dataclass
class Diagnostics:
    shift_px: tuple
    shift_confidence: float
    gain_rgb: tuple
    gain_residual: float
    blur_px: float
    noise_empty: float
    noise_product: float
    noise_diff: float
    texture_corr: float
    texture_ratio: float
    background_changed: bool
    notes: list

    def as_dict(self):
        return asdict(self)


def _luma_lin(img_lin):
    return luminance(img_lin)


def estimate_shift(empty_lin, product_lin, mask=None):
    """Sub-pixel translation of ``product`` relative to ``empty``.

    Phase correlation on band-passed luminance.  ``shift`` is the vector to add
    to empty coordinates to land on product coordinates.  The full frame is
    used: the plate texture covers it, whereas the shadow is only a small
    region, so masking (with an inevitably imperfect product mask) throws away
    the most reliable alignment information.
    """
    a = _luma_lin(empty_lin).astype(np.float32)
    b = _luma_lin(product_lin).astype(np.float32)
    a = a - cv2.GaussianBlur(a, (0, 0), 3.0)
    b = b - cv2.GaussianBlur(b, (0, 0), 3.0)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(a * win, b * win)
    # ``phaseCorrelate(src1, src2)`` returns the shift that must be applied to
    # src1 (the empty plate) to align it with src2 (the product capture).
    return (dx, dy), float(resp)


def _shift_image(img, shift):
    M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def estimate_gain(empty_lin, product_lin, bg_mask=None, iters=6):
    """Per-channel gain ``g`` with ``product ~= g * empty`` on unchanged pixels.

    The background mask almost always includes shadowed pixels (their position
    is not known before extraction).  A plain least-squares fit is therefore
    biased toward the shadow's darkening, so the estimate uses the *mode* of the
    per-pixel ratio ``product / empty``: most pixels are unshadowed ground, which
    piles up at the true gain, whereas the shadow forms a broad tail.
    """
    if bg_mask is None:
        bg_mask = np.ones(empty_lin.shape[:2], dtype=bool)
    mask = bg_mask & (empty_lin.mean(axis=2) > 5e-3)
    g = np.ones(3)
    if mask.sum() < 64:
        return g, float("nan")
    e = empty_lin[mask]
    p = product_lin[mask]
    for c in range(3):
        r = p[:, c] / np.maximum(e[:, c], 1e-6)
        # Keep the plausible band, then take the peak of a fine histogram.
        r = r[(r > 0.3) & (r < 3.0)]
        if len(r) < 64:
            continue
        hist, edges = np.histogram(r, bins=240, range=(0.3, 3.0))
        k = int(np.argmax(hist))
        peak = 0.5 * (edges[k] + edges[k + 1])
        near = r[np.abs(r - peak) < 0.03]
        g[c] = float(np.median(near)) if len(near) else float(peak)
    g = np.clip(g, 0.2, 5.0)
    pred = empty_lin * g[None, None, :]
    err = np.abs(pred - product_lin).mean(axis=2)
    good = mask & (err <= np.percentile(err[mask], 70))
    resid = float(np.mean(np.abs(pred[good] - product_lin[good]))) if good.any() else float("nan")
    return g, resid


def estimate_blur(empty_lin, product_lin, mask=None, sigmas=(0.0, 0.3, 0.5, 0.8, 1.1, 1.5, 2.0)):
    """Extra Gaussian sigma needed to match empty's sharpness to product's.

    The statistic is the energy in a mid-frequency band (Laplacian of the
    luminance after light denoising).  Band-passing keeps a small noise floor
    from masquerading as a focus difference, which a raw Laplacian variance
    would do.
    """
    a = _luma_lin(empty_lin).astype(np.float32)
    b = _luma_lin(product_lin).astype(np.float32)
    if mask is None:
        mask = np.ones(a.shape, dtype=bool)
    a = cv2.GaussianBlur(a, (0, 0), 0.8)
    b = cv2.GaussianBlur(b, (0, 0), 0.8)

    def band_energy(img):
        lap = cv2.Laplacian(img, cv2.CV_32F, ksize=3)
        return float(np.var(lap[mask]))

    e_a = band_energy(a)
    e_b = band_energy(b)
    if e_b >= 0.97 * e_a:
        return 0.0
    best, best_err = 0.0, np.inf
    for s in sigmas:
        blur = cv2.GaussianBlur(a, (0, 0), s) if s > 0 else a
        err = abs(band_energy(blur) - e_b)
        if err < best_err:
            best, best_err = s, err
    return float(best) if (e_b / max(e_a, 1e-12)) < 0.95 else 0.0


def estimate_noise(img_lin):
    """Approximate per-pixel noise sigma in linear light."""
    srgb_like = np.clip(img_lin, 0, 1)
    try:
        sig = estimate_sigma(srgb_like, channel_axis=-1, average_sigmas=True)
    except Exception:
        g = cv2.cvtColor(srgb_like.astype(np.float32), cv2.COLOR_RGB2GRAY)
        sig = float(np.std(cv2.Laplacian(g, cv2.CV_32F, ksize=3))) / 6.0
    return float(sig)


def texture_change(empty_lin, product_lin, mask):
    """High-frequency agreement on a background mask.

    Returns (correlation, std-ratio).  A true shadow scales high-frequency
    texture but does not change its *pattern*; correlation therefore stays high.
    Background replacement or added objects break it.
    """
    if mask.sum() < 256:
        return float("nan"), float("nan")
    a = _luma_lin(empty_lin)
    b = _luma_lin(product_lin)
    ah = a - cv2.GaussianBlur(a, (0, 0), 2.0)
    bh = b - cv2.GaussianBlur(b, (0, 0), 2.0)
    ah = ah[mask]
    bh = bh[mask]
    denom = np.std(ah) * np.std(bh)
    corr = float(np.mean((ah - ah.mean()) * (bh - bh.mean())) / denom) if denom > 1e-12 else float("nan")
    ratio = float(np.std(bh) / max(np.std(ah), 1e-9))
    return corr, ratio


def diagnose(empty_lin, product_lin, product_mask=None, support=None, noise_floor=None):
    """Full capture-difference diagnosis.

    ``product_mask`` / ``support`` are initial guesses used only to choose an
    unambiguous background sample; they need not be exact.
    """
    notes = []
    excluded = np.zeros(empty_lin.shape[:2], dtype=bool)
    if product_mask is not None:
        excluded |= cv2.dilate(product_mask.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    if support is not None:
        excluded |= cv2.dilate(support.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    bg = ~excluded
    if bg.sum() < 0.05 * bg.size:
        # Fallback: use the brightest, least-changed half.
        d = np.abs(product_lin - empty_lin).mean(axis=2)
        bg = d <= np.percentile(d, 50)

    shift, conf = estimate_shift(empty_lin, product_lin)
    empty_r = _shift_image(empty_lin, shift) if max(abs(shift[0]), abs(shift[1])) > 0.05 else empty_lin
    # Recompute background exclusion in the aligned frame.
    gain, gain_resid = estimate_gain(empty_r, product_lin, bg)
    empty_g = empty_r * gain[None, None, :]

    blur = estimate_blur(empty_g, product_lin, bg)
    n_e = estimate_noise(empty_lin)
    n_p = estimate_noise(product_lin)
    diff = product_lin - empty_g
    n_d = float(1.4826 * np.median(np.abs(diff[bg] - np.median(diff[bg])))) if bg.any() else float("nan")
    corr, ratio = texture_change(empty_g, product_lin, bg)

    if max(abs(shift[0]), abs(shift[1])) > 0.75:
        notes.append(f"captures are misaligned by {shift[0]:+.2f},{shift[1]:+.2f} px; "
                     "align before trusting fine contact detail")
    if abs(gain[0] - 1) + abs(gain[1] - 1) + abs(gain[2] - 1) > 0.06:
        notes.append("global exposure / white balance differs; background was gain-matched")
    if blur > 0.45:
        notes.append(f"product capture is softer (extra sigma ~{blur:.2f} px); "
                     "sharp shadow edges will be underestimated")
    if np.isfinite(corr) and corr < 0.55:
        notes.append("background high-frequency structure disagrees between captures; "
                     "changes here are not attributable to shadow")
    elif np.isfinite(ratio) and (ratio < 0.55 or ratio > 1.8):
        notes.append("background texture energy changed substantially; shadow fit may absorb it")
    if noise_floor is not None and n_d > noise_floor * 3:
        notes.append("difference noise is above the declared floor; support confidence is reduced")

    changed = bool((np.isfinite(corr) and corr < 0.55) or
                   (np.isfinite(ratio) and (ratio < 0.5 or ratio > 2.0)))
    if changed:
        notes.append("VERDICT: background changed between captures — shadow/shade separation is "
                     "unreliable outside the marked support")

    return Diagnostics(
        shift_px=(float(shift[0]), float(shift[1])),
        shift_confidence=float(conf),
        gain_rgb=tuple(float(v) for v in gain),
        gain_residual=float(gain_resid),
        blur_px=float(blur),
        noise_empty=float(n_e),
        noise_product=float(n_p),
        noise_diff=float(n_d),
        texture_corr=float(corr) if np.isfinite(corr) else float("nan"),
        texture_ratio=float(ratio) if np.isfinite(ratio) else float("nan"),
        background_changed=changed,
        notes=notes,
    )


def align_pair(empty_lin, product_lin, diag: Diagnostics):
    """Apply the diagnosed shift and gain so both live in the product frame."""
    e = empty_lin
    if max(abs(diag.shift_px[0]), abs(diag.shift_px[1])) > 0.05:
        e = _shift_image(e, diag.shift_px)
    e = e * np.asarray(diag.gain_rgb)[None, None, :]
    return e
