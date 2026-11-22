"""Evaluate an extraction against ground truth and against a second plate.

Two questions are answered separately, because they are not the same:

1. *Can the layers rebuild the original capture?*  This is a weak check: any
   decomposition that partitions the observed image can do it.
2. *Do the layers still explain a different, unseen plate?*  This is the real
   test of physical validity.  A shadow that merely reproduces a dark blob on
   the original background will fail on a plate with different texture, colour
   or brightness, because ``t`` must be texture-free and ``q = A/B`` must carry
   the product's bounce colour, not the old plate's.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field

import numpy as np
from scipy import ndimage

from .color import luminance


def _psnr(a, b, peak=1.0):
    mse = float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))
    return 10.0 * np.log10(peak ** 2 / max(mse, 1e-12))


def _ssim_masked(a, b, mask):
    """SSIM over a spatial mask, computed by zeroing the complement.

    A masked SSIM is reported as the SSIM restricted to the region's bounding
    box with out-of-mask pixels replaced by their in-mask mean, which keeps the
    local-window statistics meaningful.
    """
    from skimage.metrics import structural_similarity
    ys, xs = np.where(mask)
    if len(ys) < 64:
        return float("nan")
    y0, y1 = ys.min(), ys.max() + 1
    x0, x1 = xs.min(), xs.max() + 1
    if y1 - y0 < 7 or x1 - x0 < 7:
        return float("nan")
    m = mask[y0:y1, x0:x1]
    aa = np.clip(a[y0:y1, x0:x1], 0, 1)
    bb = np.clip(b[y0:y1, x0:x1], 0, 1)
    if aa.ndim == 3:
        fill_a = aa[m].mean(axis=0)
        fill_b = bb[m].mean(axis=0)
        aa = np.where(m[..., None], aa, fill_a)
        bb = np.where(m[..., None], bb, fill_b)
        return float(structural_similarity(aa, bb, channel_axis=2, data_range=1.0))
    aa = np.where(m, aa, aa[m].mean())
    bb = np.where(m, bb, bb[m].mean())
    return float(structural_similarity(aa, bb, data_range=1.0))


def _ssim_global(a, b):
    from skimage.metrics import structural_similarity
    a = np.clip(a, 0, 1)
    b = np.clip(b, 0, 1)
    if a.ndim == 3:
        return float(structural_similarity(a, b, channel_axis=2, data_range=1.0))
    return float(structural_similarity(a, b, data_range=1.0))


@dataclass
class LayerMetrics:
    t_mae: float
    t_rmse: float
    t_bias: float
    t_deep_mae: float
    q_mae: float
    q_bias: float
    q_cos: float
    alpha_iou: float
    alpha_mae: float
    texture_leak: float
    recon_psnr: float
    recon_ssim: float
    recon_psnr_ground: float


@dataclass
class TransferMetrics:
    psnr: float
    ssim: float
    psnr_ground: float
    ssim_ground: float
    psnr_alt: float          # alternative: shadow scale copied in display space
    artifact_ratio: float     # texture left on the transferred shadow (lower better)
    per_regions: dict = field(default_factory=dict)


def shadow_region(gt, alpha_thresh=0.2, t_thresh=0.98):
    a = gt["alpha"]
    t = gt["t"]
    return (a < alpha_thresh) & (t < t_thresh)


def texture_leak(t_pred, beta, shadow_mask):
    """Correlation between predicted transmission and plate reflectance.

    Physical occlusion transmission is independent of the receiving surface, so
    a non-zero correlation means the estimator has absorbed plate texture into
    the shadow layer.  Evaluated on the shadow region only.
    """
    if shadow_mask.sum() < 100:
        return float("nan")
    tv = t_pred[shadow_mask]
    bv = luminance(beta)[shadow_mask]
    tv = tv - tv.mean()
    bv = bv - bv.mean()
    denom = np.std(tv) * np.std(bv)
    if denom < 1e-12:
        return 0.0
    return float(np.mean(tv * bv) / denom)


def evaluate_layers(dec, gt, shadow_pad=1):
    """Compare the extracted layers to ground-truth layers."""
    t_gt = gt["t"]
    z_gt = gt["q"]
    a_gt = gt["alpha"]
    sm = shadow_region(gt)
    # Slight erosion avoids counting the silhouette boundary, where alpha and
    # t are genuinely ambiguous.
    sm_in = ndimage.binary_erosion(sm, iterations=shadow_pad) if shadow_pad else sm
    if sm_in.sum() < 50:
        sm_in = sm

    t_pred = dec.t
    z_pred = dec.q
    deep = sm_in & (t_gt < 0.6)
    mae = float(np.mean(np.abs(t_pred[sm_in] - t_gt[sm_in])))
    rmse = float(np.sqrt(np.mean((t_pred[sm_in] - t_gt[sm_in]) ** 2)))
    bias = float(np.mean(t_pred[sm_in] - t_gt[sm_in]))
    deep_mae = float(np.mean(np.abs(t_pred[deep] - t_gt[deep]))) if deep.sum() > 20 else float("nan")

    q_mae = float(np.mean(np.abs(z_pred[sm_in] - z_gt[sm_in])))
    q_bias = float(np.mean(z_pred[sm_in] - z_gt[sm_in]))
    pv = z_pred[sm_in].reshape(-1, 3)
    gv = z_gt[sm_in].reshape(-1, 3)
    num = np.sum(pv * gv, axis=1)
    den = np.linalg.norm(pv, axis=1) * np.linalg.norm(gv, axis=1) + 1e-12
    q_cos = float(np.mean(num / den)) if len(num) else float("nan")

    pa = dec.alpha > 0.5
    ga = a_gt > 0.5
    iou = float((pa & ga).sum() / max((pa | ga).sum(), 1))
    alpha_mae = float(np.mean(np.abs(dec.alpha - a_gt)))

    leak = texture_leak(t_pred, gt["beta1"], sm_in)

    recon = dec.recon()
    I_gt = gt["I1"]
    ps = _psnr(recon, I_gt)
    ss = _ssim_global(recon, I_gt)
    ground_only = (~ga)
    ps_g = _psnr(recon[ground_only], I_gt[ground_only]) if ground_only.any() else float("nan")
    return LayerMetrics(t_mae=mae, t_rmse=rmse, t_bias=bias, t_deep_mae=deep_mae,
                        q_mae=q_mae, q_bias=q_bias, q_cos=q_cos,
                        alpha_iou=iou, alpha_mae=alpha_mae, texture_leak=leak,
                        recon_psnr=ps, recon_ssim=ss, recon_psnr_ground=ps_g)


def evaluate_transfer(dec, gt, second_plate=True):
    """Apply the extracted layer to the *second* plate and compare to truth.

    The comparison target is the independently rendered ``I2``.  The ``psnr_alt``
    reference deliberately uses the wrong model -- it reuses the original
    capture's darkening ``(t + q)`` as a plain multiplicative scale after
    display encoding -- to show the gap between "explains this plate" and a
    physically transferable layer.
    """
    if not second_plate:
        return None
    B2 = gt["B2"]
    I2_gt = gt["I2"]
    a_gt = gt["alpha"]
    P_gt = gt["P"]
    pred = dec.transfer(B2)
    pred_full = dec.transfer(B2, alpha2=a_gt, P2=P_gt) if "P" in gt else pred
    ground = a_gt < 0.2
    ps = _psnr(pred_full, I2_gt)
    ss = _ssim_global(pred_full, I2_gt)
    ps_g = _psnr(pred_full[ground], I2_gt[ground]) if ground.any() else float("nan")
    ss_g = _ssim_masked(pred_full, I2_gt, ground)
    # "copy the darkening" alternative: the naive 2-D compositor model.  From the
    # first capture it records the *display-space* darkening ratio and applies
    # it to the second plate's display values.  Two errors compound: the sRGB
    # ratio of ``B*(t+q)`` depends on the first plate's brightness, so it does
    # not commute to a different plate, and the arithmetic is done in the wrong
    # (gamma-encoded) space.  This is the baseline the extracted layer beats.
    from .color import linear_to_srgb, srgb_to_linear

    def enc(x):
        return np.clip(linear_to_srgb(np.clip(x, 0, 1)), 0, 1)

    ratio_srgb = enc(dec.ground_recon()) / np.maximum(enc(dec.B), 1e-4)
    alt = srgb_to_linear(np.clip(enc(B2) * ratio_srgb, 0, 1))
    alt_full = a_gt[..., None] * P_gt + (1 - a_gt[..., None]) * alt
    ps_alt = _psnr(alt_full, I2_gt)

    # How much old-plate texture remains in the transferred shadow?  Compare the
    # shadow's high-frequency content on plate 2 to plate 1's.
    sm = ground & (dec.t < 0.98)
    leak = float("nan")
    if sm.sum() > 100:
        from .color import luminance as lum
        hf1 = lum(pred)[sm] - ndimage.gaussian_filter(lum(pred), 2.0)[sm]
        hf2 = lum(I2_gt)[sm] - ndimage.gaussian_filter(lum(I2_gt), 2.0)[sm]
        leak = float(np.std(hf1) / max(np.std(hf2), 1e-9))

    return TransferMetrics(psnr=ps, ssim=ss, psnr_ground=ps_g, ssim_ground=ss_g,
                           psnr_alt=ps_alt, artifact_ratio=leak)


def evaluate(dec, gt, second_plate=True):
    L = evaluate_layers(dec, gt)
    T = evaluate_transfer(dec, gt, second_plate)
    return L, T


def to_dict(L, T):
    d = {"layers": asdict(L)}
    d["transfer"] = asdict(T) if T is not None else None
    return d
