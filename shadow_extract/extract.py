"""Physical shadow/foreground separation for a fixed-camera plate pair.

Forward model (linear light, per pixel):

    empty plate            B(x)  = beta(x) * E0
    with product present   I(x)  = alpha(x) * P(x) + (1 - alpha(x)) * (t(x) * B(x) + A(x))

``t`` is the achromatic transmission of the total irradiance (ambient occlusion
times direct visibility).  ``A`` is additive light reflected off the product.
For a Lambertian receiver ``A = beta * Q``, with ``Q`` a function of the product
lighting and geometry, not of the table.  Hence the plate-independent quantity
is the ratio field ``q = A / B = Q / E0`` and the ground under the product is
``I = B * (t + q)``.

Write ``y = I / B`` per channel.  Then for every ground pixel

    y_c = g * (t + q_c) + n_c ,        q_c = beta_c * s        (s = Q/E0, scalar)

so the unknown quantities per pixel are the occlusion ``t`` (shared by all
channels) and the three-channel ``q`` (rank one in beta).  A per-channel
``g = 1 + delta_c`` absorbs residual white-balance drift.

The solver runs two strategies and reports both:

* ``joint``     -- beyond the product silhouette, estimate ``t`` and ``q``
                   together under smoothness on ``t`` and low-rank/strong
                   smoothness on ``q``.
* ``fit_only``  -- take the visible ground just outside the silhouette as
                   "known lit", lock that surface (its beta and the light
                   level), and fit ``t``; ``q`` is whatever residual is needed.

Robustness rules that matter for the "black halo" failure mode:

* ratios are only trusted where ``B`` is safely above the sensor noise floor;
  the ratio ``y`` is clipped to a bounded range and down-weighted where it is
  unreliable, so it can never blow up as ``B -> 0``;
* ``t`` is constrained to ``[t_min, 1]``; the shadow cannot be an arbitrary
  dark blob;
* smoothness is imposed on ``t`` and on ``q``, never on ``I``, so the plate's
  own texture is not smoothed into the shadow;
* pixels whose residual stays large are labelled "unexplained" instead of being
  forced into the shadow layer.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy import ndimage, sparse
from scipy.sparse import linalg as splinalg

from .color import linear_to_srgb, luminance, srgb_to_linear
from .diagnostics import Diagnostics, align_pair, diagnose

# --------------------------------------------------------------------------- #
# Tunables (exposed in the UI)
# --------------------------------------------------------------------------- #

@dataclass
class Params:
    # Support / region handling
    support_erode: int = 0            # px to erode the user's support region
    support_feather: int = 3          # px feather of support weight
    # Ratiometric estimation
    noise_k: float = 2.0              # ratio reliability: B > noise_k * sigma_B
    y_clip: tuple = (0.0, 2.5)        # hard bound on I/B
    min_t: float = 0.02               # darkest physically-plausible transmission
    # Smoothness (in normalised image units; scaled by resolution)
    t_smooth: float = 0.9
    q_smooth: float = 6.0
    q_decay: float = 0.25             # pull q -> 0 away from the contact
    q_material: float = 2.0           # pull q toward a single material direction
    q_max: float = 0.12               # upper bound on the additive A/B term
    # Iterations
    outer_iters: int = 6
    pcg_iters: int = 260
    # Unexplained-region threshold
    unexplained_sigma: float = 3.0
    unexplained_abs: float = 0.045
    # Diagnostics / alignment
    auto_align: bool = True


# --------------------------------------------------------------------------- #
# Region construction
# --------------------------------------------------------------------------- #

def _disk(radius):
    r = int(round(radius))
    if r <= 0:
        return np.ones((1, 1), np.uint8)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy * yy + xx * xx) <= r * r).astype(np.uint8)


def initial_product_mask(product_lin, empty_lin, support):
    """Rough foreground seed, restricted to ``support``.

    Two independent cues, because neither alone is reliable:

    * **Chromatic** -- a shadow is (approximately) achromatic transmission of the
      plate, so ``I/B`` stays close to one value across channels; a product
      replaces the surface and its ``I/B`` is chromatically unrelated.  This
      finds coloured products but fails for a near-neutral product.
    * **Texture occlusion** -- the plate's fine texture is *scaled* under a
      shadow (its high-frequency pattern is preserved) but *replaced* by the
      product.  Local correlation between the high-pass images is high on ground
      and low on the product.  This finds neutral products such as a white mug
      on a light plate, which the chromatic cue misses.

    A product that merely shadows the plate must satisfy both "not a scaled copy
    of the texture" and "not a neutral scaling of the plate"; requiring the two
    cues to agree avoids mistaking a strong shadow edge for a product.
    """
    B = np.maximum(empty_lin, 1e-4)
    y = np.clip(product_lin / B, 0.02, 6.0)
    ly = np.log(y)
    chroma_spread = ly.std(axis=2)
    luma_ratio = np.exp(ly.mean(axis=2))
    strong_chroma = (chroma_spread > 0.10) | (luma_ratio > 1.20)

    # --- local texture-correlation cue ------------------------------------ #
    from scipy import ndimage
    ge = ndimage.gaussian_filter(luminance(empty_lin).astype(np.float32), 1.0)
    ge = luminance(empty_lin).astype(np.float32) - ge
    gp = luminance(product_lin).astype(np.float32)
    gp = gp - ndimage.gaussian_filter(gp, 1.0)
    win = 4.0
    mue = ndimage.gaussian_filter(ge, win)
    mup = ndimage.gaussian_filter(gp, win)
    cov = ndimage.gaussian_filter(ge * gp, win) - mue * mup
    vare = ndimage.gaussian_filter(ge * ge, win) - mue * mue
    varp = ndimage.gaussian_filter(gp * gp, win) - mup * mup
    corr = cov / np.sqrt(np.maximum(vare * varp, 1e-12))
    corr_s = ndimage.gaussian_filter(corr.astype(np.float32), 2.5)
    plate_has_texture = float((vare > 5e-6).mean()) > 0.15

    # Chroma is the primary cue: a cast shadow transmits the plate's colour
    # ratios almost unchanged (chroma near 0) while a product -- even a near
    # white one -- deviates measurably.  The threshold is *relative* to the
    # strongest chroma in the support region, so it adapts to a neutral product
    # and to white-balance drift instead of assuming an absolute level.
    ch_in = chroma_spread[support]
    ch_peak = float(np.percentile(ch_in, 99)) if ch_in.size else 0.0
    if ch_peak > 0.05:
        thresh = max(0.45 * ch_peak, 0.012)
        chroma_evidence = support & (chroma_spread > thresh)
    else:
        chroma_evidence = np.zeros_like(support)

    # Texture cue adds the strongly-textured neutral case (product covering a
    # busy plate) and, more importantly, lets us drop the shadow's own soft edge,
    # which has low chroma but *preserved* texture.
    texture_evidence = support & (vare > 5e-6) & (corr_s < 0.60) if plate_has_texture \
        else np.zeros_like(support)

    cand = chroma_evidence | texture_evidence
    cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, _disk(3)) > 0
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros(support.shape, bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    cand = lab == k

    # Final discrimination inside the largest component: a physical product is
    # chromatically distinct from the plate or replaces its texture.  Keep the
    # union of those, then restore interior holes (a neutral highlight region
    # inside a coloured product must not punch a hole).
    chroma_strong = cand & chroma_evidence
    texture_strong = cand & texture_evidence
    keep = chroma_strong | texture_strong
    # Drop only the *shadow-edge* part of the texture evidence: it is the region
    # where corr is broken but chroma is already at the plate level and which is
    # spatially attached to low-occlusion ground.
    shadowish = cand & (chroma_spread < 0.4 * ch_peak) & (~chroma_strong)
    keep = keep & ~shadowish
    if keep.sum() < 0.15 * cand.sum():
        keep = cand
    keep = ndimage.binary_fill_holes(keep)
    keep = cv2.morphologyEx(keep.astype(np.uint8), cv2.MORPH_CLOSE, _disk(7)) > 0
    return keep


def expand_mask(mask, px):
    if px <= 0:
        return mask.copy()
    return cv2.dilate(mask.astype(np.uint8), _disk(px)) > 0


def support_weight(support, feather):
    """Soft 0..1 weight from a hard support polygon."""
    s = support.astype(np.float32)
    if feather <= 0:
        return s
    return cv2.GaussianBlur(s, (0, 0), feather)


# --------------------------------------------------------------------------- #
# Small sparse helpers
# --------------------------------------------------------------------------- #

def _poisson_solve(known_idx, known_val, w_data, lam, shape, iters=260, w_extra=None,
                   extra_anchor_idx=None, extra_anchor_val=None, extra_lambda=0.0):
    """Solve  sum_p w_p (u_p - d_p)^2 + lam * sum_{p~q} (u_p - u_q)^2.

    ``known_idx`` are flat pixel indices with data terms ``known_val`` and
    weights ``w_data``; all other pixels are pure smoothness.  A graph-
    Laplacian system in sparse form, solved with CG + Jacobi.
    """
    H, W = shape
    N = H * W
    rows, cols, vals = [], [], []
    rhs = np.zeros(N)
    diag = np.zeros(N)

    # Data terms
    diag[known_idx] += w_data
    rhs[known_idx] += w_data * known_val

    # Smoothness terms (4-neighbour)
    idx = np.arange(N).reshape(H, W)
    pairs = []
    for dy, dx in ((0, 1), (1, 0)):
        a = idx[:H - dy, :W - dx].ravel()
        b = idx[dy:, dx:].ravel()
        pairs.append((a, b))
    for a, b in pairs:
        rows.append(a); cols.append(b); vals.append(-lam * np.ones(len(a)))
        rows.append(b); cols.append(a); vals.append(-lam * np.ones(len(b)))
        diag[a] += lam
        diag[b] += lam

    if extra_anchor_idx is not None and extra_lambda > 0:
        diag[extra_anchor_idx] += extra_lambda
        rhs[extra_anchor_idx] += extra_lambda * extra_anchor_val

    # Pin one pixel to remove the nullspace when no data term exists there.
    if not np.any(w_data > 0):
        diag[0] += 1.0
        rhs[0] += 1.0

    rows = np.concatenate(rows + [np.arange(N)])
    cols = np.concatenate(cols + [np.arange(N)])
    vals = np.concatenate(vals + [diag + 1e-9])
    A = sparse.csr_matrix((vals, (rows, cols)), shape=(N, N))
    diagA = A.diagonal()
    M = sparse.diags(1.0 / np.maximum(diagA, 1e-9))
    x, info = splinalg.cg(A, rhs, M=M, x0=known_val_full(known_idx, known_val, N),
                          maxiter=iters, rtol=1e-5, atol=0.0)
    if info != 0:
        x = splinalg.spsolve(A.tocsc(), rhs)
    return x.reshape(H, W)


def known_val_full(idx, val, N):
    x = np.full(N, np.nan)
    x[idx] = val
    x[np.isnan(x)] = np.median(val) if len(val) else 0.0
    return x


def _smooth_step(img, lam, data=None, data_w=None, iters=1):
    """Weighted least-squares smoothing of an image with optional data terms."""
    H, W = img.shape[:2]
    if img.ndim == 3:
        return np.stack([_smooth_step(img[..., c], lam,
                                      None if data is None else data[..., c],
                                      data_w, iters) for c in range(img.shape[2])], axis=2)
    N = H * W
    if data is None:
        idx = np.array([], dtype=int)
        w = np.zeros(0)
        d = np.zeros(0)
    else:
        idx = np.flatnonzero(data_w.ravel() > 0)
        w = data_w.ravel()[idx]
        d = data.ravel()[idx]
    return _poisson_solve(idx, d, w, lam, (H, W), iters=iters)


# --------------------------------------------------------------------------- #
# Core decomposition
# --------------------------------------------------------------------------- #

@dataclass
class Decomposition:
    alpha: np.ndarray
    P_lin: np.ndarray
    t: np.ndarray
    q: np.ndarray
    B: np.ndarray
    g: np.ndarray
    residual: np.ndarray
    unexplained: np.ndarray
    product_mask: np.ndarray
    support: np.ndarray
    diag: Diagnostics
    params: Params
    method: str
    info: dict = field(default_factory=dict)

    # -- derived, always in the aligned product frame ---------------------- #

    def ground_recon(self):
        return self.B * (self.t[..., None] + self.q)

    def recon(self):
        a = self.alpha[..., None]
        return a * self.P_lin + (1 - a) * self.ground_recon()

    def transfer(self, B2, alpha2=None, P2=None):
        """Re-light the extracted layer on a different plate.

        ``I2 = B2*(t + q)`` with the shadow untouched; optionally composite the
        product (or another one) on top with its own matte.
        """
        g2 = B2 * (self.t[..., None] + self.q)
        if alpha2 is not None and P2 is not None:
            a = np.asarray(alpha2, float)[..., None]
            return a * np.asarray(P2, float) + (1 - a) * g2
        return g2

    def combined_product(self, B2):
        """Product composited over the transferred shadow on plate B2."""
        return self.alpha[..., None] * self.P_lin + (1 - self.alpha[..., None]) * self.transfer(B2)


# --------------------------------------------------------------------------- #
# Segmentation / matte
# --------------------------------------------------------------------------- #

def _matte_from_residual(alpha0, residual_lum, support, feath=1.5):
    """Refine a hard product mask into a soft alpha using residual magnitude.

    Residual (observed minus shadow-only prediction) is large on the product and
    small on the ground.  The soft edge comes from a local sigmoid, not from a
    blur painted around the silhouette.
    """
    a = alpha0.astype(np.float32)
    r = cv2.GaussianBlur(residual_lum.astype(np.float32), (0, 0), 1.0)
    inside = a > 0.5
    if inside.sum() < 32:
        return a
    hi = np.percentile(r[inside], 75)
    lo = np.percentile(r[~inside & support], 50) if (~inside & support).any() else 0.0
    mid = 0.5 * (hi + lo)
    width = max(0.25 * (hi - lo), 1e-4)
    soft = 1.0 / (1.0 + np.exp(-np.clip((r - mid) / width, -60.0, 60.0)))
    # Only allow the matte to change near the boundary.
    dil = cv2.dilate(inside.astype(np.uint8), _disk(3)) > 0
    ero = cv2.erode(inside.astype(np.uint8), _disk(3)) > 0
    edge_band = dil & ~ero
    out = a.copy()
    out[edge_band] = soft[edge_band] * 0.7 + a[edge_band] * 0.3
    out = np.where(dil, np.maximum(out, a * (ero)), out)
    return np.clip(out, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #

def prepare_inputs(empty_srgb_u8, product_srgb_u8):
    """uint8 sRGB -> linear float RGB float64."""
    e = srgb_to_linear(empty_srgb_u8.astype(np.float64) / 255.0)
    p = srgb_to_linear(product_srgb_u8.astype(np.float64) / 255.0)
    return e, p


def parse_support(support, shape):
    """Normalise the support region (bool array, polygon or None) to bool."""
    if support is None:
        return np.ones(shape, dtype=bool)
    if isinstance(support, np.ndarray):
        if support.shape != shape:
            raise ValueError(f"support shape {support.shape} != image {shape}")
        return support.astype(bool)
    # Polygon points (x, y)
    m = np.zeros(shape, dtype=np.uint8)
    pts = np.asarray(support, dtype=np.int32)
    if len(pts) >= 3:
        cv2.fillPoly(m, [pts], 1)
    return m.astype(bool)


def extract(empty_srgb_u8, product_srgb_u8, support=None,
            method="joint", params: Params | None = None,
            external_mask=None, noise_floor=None):
    """Extract (alpha, P, t, q) from one empty/product pair.

    Parameters
    ----------
    empty_srgb_u8, product_srgb_u8 : (H, W, 3) uint8 sRGB
    support : bool array or list of (x, y) polygon points.  The estimated
              shadow is confined to this region; outside it the plate is
              returned untouched.
    method : "joint" | "fit_only"
    """
    params = params or Params()
    if method not in ("joint", "fit_only"):
        raise ValueError(method)
    H, W = product_srgb_u8.shape[:2]
    empty_lin, prod_lin = prepare_inputs(empty_srgb_u8, product_srgb_u8)
    support = parse_support(support, (H, W))

    # --- 1. Diagnose capture differences and align ------------------------- #
    if external_mask is not None:
        seed_mask = external_mask.astype(bool)
    else:
        seed_mask = initial_product_mask(prod_lin, empty_lin, support)
    diag = diagnose(empty_lin, prod_lin, product_mask=seed_mask, support=support,
                    noise_floor=noise_floor)
    if params.auto_align:
        B = align_pair(empty_lin, prod_lin, diag)
    else:
        B = empty_lin.copy()
    if seed_mask is None or not seed_mask.any():
        seed_mask = initial_product_mask(prod_lin, B, support)

    # --- 2. Support handling ---------------------------------------------- #
    if params.support_erode > 0:
        support = cv2.erode(support.astype(np.uint8), _disk(params.support_erode)) > 0
    sw = support_weight(support, params.support_feather)

    # --- 3. Product silhouette seed --------------------------------------- #
    prod_mask = expand_mask(seed_mask, 2) & support
    # Ground = support minus the product, plus a thin band we will not insist on.
    ground_candidate = support & ~expand_mask(prod_mask, 1)

    # --- 4. Noise floor on B ---------------------------------------------- #
    sigma_B = np.maximum(diag.noise_diff, 1e-4)
    reliable = B.mean(axis=2) > max(params.noise_k * sigma_B, 1e-3)

    # --- 5. Joint / fit-only solve ---------------------------------------- #
    if method == "joint":
        t, q, g, residual, unexplained = _solve_joint(
            B, prod_lin, prod_mask, ground_candidate, reliable, diag, params, sw, support)
    else:
        t, q, g, residual, unexplained = _solve_fit_only(
            B, prod_lin, prod_mask, ground_candidate, reliable, diag, params, sw)

    # --- 6. Matte + product radiance -------------------------------------- #
    from .color import luminance
    residual_lum = np.abs(luminance(residual))
    alpha = _matte_from_residual(prod_mask.astype(np.float32), residual_lum, support)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[~support] = 0.0
    P = _recover_product_radiance(prod_lin, B, t, q, alpha)
    # Avoid a dark halo: fringes take product colour, not a mix toward black.
    P = _fill_product_color(P, alpha)

    info = dict(method=method, n_ground=int(ground_candidate.sum()),
                n_product=int((alpha > 0.5).sum()),
                n_unexplained=int(unexplained.sum()),
                t_min=float(t.min()), t_max=float(t.max()),
                q_max=float(q.max()), g=list(map(float, g)))
    return Decomposition(alpha=alpha, P_lin=P, t=t, q=q, B=B, g=g,
                         residual=residual, unexplained=unexplained,
                         product_mask=alpha > 0.5, support=support,
                         diag=diag, params=params, method=method, info=info)


# --------------------------------------------------------------------------- #
# Solver: joint estimation of projection and plate direction
# --------------------------------------------------------------------------- #

def _solve_joint(B, I, prod_mask, ground, reliable, diag, params, sw, support_mask):
    """Joint estimation of occlusion ``t`` and bounce ``q`` from the plate ratios.

    Model per ground pixel and channel::

        y_c = g_c * (t + s * d_c),      q_c = s * d_c

    ``t`` is achromatic (the same attenuation in every channel) and ``d`` is a
    single, global bounce direction (the product's bounce colour divided by the
    illuminant).  With three channels and two unknowns per pixel plus a global
    ``d``, the split is identifiable from colour -- this is what separates the
    additive bounce from occlusion instead of leaving it to a smoothness prior
    alone.  Smoothness then regularises ``t`` (sharp, compact) and ``s``
    (broad), and refitting ``g`` removes residual white-balance drift.
    """
    H, W = B.shape[:2]
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    N = H * W
    Bs = np.maximum(B, 1e-6)
    y = np.clip(I / Bs, *params.y_clip)

    lB = B.mean(axis=2)
    ratio_snr = lB / max(params.noise_k * max(diag.noise_diff * 1.4826, 1e-4), 1e-6)
    reliable_w = np.clip(ratio_snr - 1.0, 0.0, 1.0)
    reliable_w[~reliable] = 0.0

    valid_geom = (~prod_mask) & ground & (sw > 0)
    data_w = (valid_geom * reliable_w * sw).astype(np.float64)
    sel = valid_geom & (data_w > 0)
    sm_w = (support_mask & (sw > 0)).astype(np.float32)
    lam_t = params.t_smooth * N / 1e5
    lam_s = params.q_smooth * N / 1e5

    t_seed = _transport_t(B, I, prod_mask, data_w, params)
    t = t_seed.copy()
    s = np.zeros((H, W))
    g = np.ones(3)
    d = np.ones(3)
    q = np.zeros((H, W, 3))

    if not sel.any():
        return t, q, g, I - B, np.zeros((H, W), bool)

    for it in range(params.outer_iters):
        yg = y / g[None, None, :]

        # ---- achromatic level and colour direction ------------------------ #
        # Remove the estimated bounce first, then average channels: this is the
        # per-pixel transmission evidence.
        u_level = (yg - q).mean(axis=2)

        # ---- t: smooth, sharp-field solve from observed ground ------------ #
        t_data = np.where(sel, u_level, t)
        idx = np.arange(N)
        w_vec = np.where(data_w.ravel() > 0, data_w.ravel(), 1e-3)
        t = _poisson_solve(idx, t_data.ravel(), w_vec, lam_t, (H, W), iters=params.pcg_iters)
        t = np.clip(t, params.min_t, 1.0)
        t[sw <= 0] = 1.0
        t = _laplace_fill(t, prod_mask, boundary_mask=~prod_mask)

        # ---- s: broad additive pedestal ----------------------------------- #
        resid = np.clip(yg - t[..., None], 0.0, None)      # ~ q on the ground
        # Rank-one direction from the chromaticity of the residual.
        if sel.sum() > 64:
            m_est = np.median(resid[sel], axis=0)
            d_new = np.maximum(m_est, 0.0)
            if d_new.max() > 1e-6:
                d = d_new / d_new.max()
        # Scalar amplitude: project the residual onto d, smooth it broadly over
        # the whole support so it also illuminates the hidden ground.
        dd = max(float(d @ d), 1e-6)
        s_raw = np.where(sel, (resid * d[None, None, :]).sum(axis=2) / dd, 0.0)
        s = _smooth_field_keep(np.clip(s_raw, 0.0, None), sm_w, params.q_smooth,
                               extra_decay=params.q_decay)
        s = np.clip(s, 0.0, params.q_max)
        s = np.minimum(s, np.maximum(t + s - params.min_t, 0.0))
        q = s[..., None] * d[None, None, :]

        # ---- gain ---------------------------------------------------------- #
        pred = t[..., None] + q
        if sel.sum() > 128:
            gs = []
            for c in range(3):
                a = pred[..., c][sel]
                b = y[..., c][sel]
                denom = float(np.sum(a * a))
                gs.append(float(np.sum(a * b) / denom) if denom > 1e-9 else 1.0)
            g = np.clip(np.array(gs), 0.7, 1.4)

    pred = B * (t[..., None] + q)
    residual = I - pred
    res_lum = np.abs((residual * lum_w).sum(axis=2))
    sig = max(diag.noise_diff * 1.4826, 1e-4)
    thr = max(params.unexplained_sigma * sig, params.unexplained_abs)
    unexplained = (res_lum > thr) & (~prod_mask) & sw.astype(bool)
    return t, q, g, residual, unexplained


def _transport_t(B, I, prod_mask, data_w, params):
    """Initial transmission: observed values, then nearest-rim fill inward.

    Under the silhouette the shadow is not directly observable.  Copying the
    *nearest visible* transmission (a Voronoi fill from the silhouette rim)
    preserves the deep contact attenuation instead of starting from an all-open
    field that the smoothness prior would then be loath to darken.
    """
    H, W = B.shape[:2]
    Bs = np.maximum(B, 1e-6)
    y = np.clip(I / Bs, *params.y_clip)
    lum_w = np.array([0.2126, 0.7152, 0.0722])
    t_obs = (y * lum_w[None, None, :]).sum(axis=2)
    t = np.ones((H, W))
    have = data_w > 0
    t[have] = np.clip(t_obs[have], params.min_t, 1.0)
    # Fill unobserved pixels from the nearest observed one.
    if have.any() and (~have).any():
        ind = ndimage.distance_transform_edt(~have, return_distances=False, return_indices=True)
        t[~have] = t[ind[0][~have], ind[1][~have]]
    return t


def _laplace_fill(field, hole, boundary_mask):
    """Harmonic in-painting of ``field`` inside ``hole`` from its boundary.

    The known values just outside the hole act as Dirichlet conditions; the
    hole is solved for the Laplace equation so the field continues smoothly
    across the silhouette without radial artefacts.
    """
    if not hole.any():
        return field
    known = ~hole
    # Boundary = known pixels adjacent to the hole.
    boundary = known & cv2.dilate(hole.astype(np.uint8), _disk(1)).astype(bool)
    if boundary.sum() < 8:
        return field
    fill = field.copy()
    N = field.size
    idx = np.arange(N).reshape(field.shape)
    hole_idx = np.flatnonzero(hole.ravel())
    b_idx = np.flatnonzero(boundary.ravel())
    b_val = field.ravel()[b_idx]
    solved = _poisson_solve(b_idx, b_val, np.ones(len(b_idx)) * 50.0, 12.0,
                           field.shape, iters=400)
    fill.ravel()[hole_idx] = solved.ravel()[hole_idx]
    return fill


def _smooth_field_keep(field, weight, lam, iters=1, extra_decay=0.0):
    """Smooth ``field`` where ``weight`` is high, decay it where it is zero.

    ``extra_decay`` adds a weak prior pulling the field to zero, which keeps the
    bounce term from leaking far from the product.
    """
    lam_px = lam * field.size / 1e5
    idx = np.flatnonzero(weight.ravel() > 0)
    if len(idx) < 16:
        return field.copy()
    w_data = weight.ravel()[idx] + 1e-3
    if extra_decay > 0:
        all_idx = np.arange(field.size)
        w_all = np.full(field.size, extra_decay)
        w_all[idx] += w_data
        d_all = np.zeros(field.size)
        d_all[idx] = field.ravel()[idx]
        return _poisson_solve(all_idx, d_all, w_all, lam_px, field.shape, iters=220)
    return _poisson_solve(idx, field.ravel()[idx], w_data,
                          lam_px, field.shape, iters=220)


# --------------------------------------------------------------------------- #
# Solver: lock the known lit surface, fit the shadow only
# --------------------------------------------------------------------------- #

def _solve_fit_only(B, I, prod_mask, ground, reliable, diag, params, sw):
    """Assume the plate just outside the product is fully lit.

    That fixes the plate's reflectance and the light level.  The shadow is then
    a single scalar attenuation ``t`` fitted to the ground band; the additive
    term ``q`` is estimated as a smooth residual but is not allowed to carry
    the plate texture, because the surface is locked.
    """
    H, W = B.shape[:2]
    Bs = np.maximum(B, 1e-6)
    y = np.clip(I / Bs, *params.y_clip)
    lum_w = np.array([0.2126, 0.7152, 0.0722])

    # The *known base surface* is the lit periphery of the support: ground that
    # is near the support boundary and far from the product.  Cast shadows hug
    # the product, so this band is unshadowed by construction -- that is the
    # physical premise of this method (the user marks an area whose outer ground
    # is fully lit).  Using the whole ground as the lock would pin the shadow to
    # full brightness, because the ground also contains the shadow itself.
    valid = ground & reliable
    margin = max(8, int(round(0.05 * min(H, W))))
    lit_ring = valid & ~ndimage.binary_erosion(sw > 0.5, iterations=margin)
    if lit_ring.sum() < 64:
        # Fall back to the brightest quarter of the ground when the support is
        # too tight for a boundary band.
        gv = (y * lum_w[None, None, :]).sum(axis=2)[valid]
        if gv.size:
            thr = np.percentile(gv, 75)
            lit_ring = valid & (((y * lum_w[None, None, :]).sum(axis=2)) >= thr)
    if valid.sum() < 64 or lit_ring.sum() < 32:
        return (np.ones((H, W)), np.zeros((H, W, 3)), np.ones(3),
                I - B, np.zeros((H, W), bool))

    # White-balance from the *lit* ring only, so the shadow does not drag it.
    g = np.clip(np.median(y[lit_ring], axis=0), 0.7, 1.4)
    yg = y / g[None, None, :]

    data_lum = (yg * lum_w[None, None, :]).sum(axis=2)
    q = np.zeros((H, W, 3))
    t = np.ones((H, W))
    w_pin = 50.0
    for it in range(params.outer_iters):
        q_lum = (q * lum_w[None, None, :]).sum(axis=2)
        t_data = data_lum - q_lum
        w_data = valid.astype(np.float32) * sw + 1e-3
        d_data = np.where(valid, t_data, 1.0)
        # Pin the lit periphery to full transmission with a stiff weight.
        d_data = np.where(lit_ring, 1.0, d_data)
        w_data = np.where(lit_ring, w_pin, w_data)
        t = _poisson_solve(np.arange(H * W), d_data.ravel(), w_data.ravel(),
                           params.t_smooth * H * W / 1e5, (H, W), iters=params.pcg_iters)
        t = np.clip(t, params.min_t, 1.0)
        # q: smooth residual inside the shadow band, forced to 0 on the lit ring.
        resid = yg - t[..., None]
        s = np.maximum(resid.mean(axis=2), 0.0)
        s[~valid] = 0.0
        s[lit_ring] = 0.0
        s = _smooth_field_keep(s, valid.astype(np.float32) * sw, params.q_smooth)
        dirn = np.median(np.clip(resid[valid], 0.0, None), axis=0) if valid.any() else np.zeros(3)
        q = s[..., None] * np.maximum(dirn, 0.0)[None, None, :]
        t[lit_ring] = np.maximum(t[lit_ring], 1.0 - 1e-3)

    pred = B * (t[..., None] + q)
    residual = I - pred
    res_lum = np.abs((residual * lum_w).sum(axis=2))
    sig = max(diag.noise_diff * 1.4826, 1e-4)
    thr = max(params.unexplained_sigma * sig, params.unexplained_abs)
    unexplained = (res_lum > thr) & (~prod_mask) & sw.astype(bool)
    return t, q, g, residual, unexplained


# --------------------------------------------------------------------------- #
# Product radiance recovery
# --------------------------------------------------------------------------- #

def _recover_product_radiance(I, B, t, q, alpha):
    ground = B * (t[..., None] + q)
    a = alpha[..., None]
    P = np.where(a > 0.02, (I - (1 - a) * ground) / np.maximum(a, 1e-3), 0.0)
    return np.clip(P, 0.0, None)


def _fill_product_color(P, alpha, iters=2):
    """Replace empty product colours with nearby valid ones (no black halo)."""
    P = P.copy()
    valid = alpha > 0.25
    if not valid.any():
        return P
    ys, xs = np.where(valid)
    from scipy.spatial import cKDTree
    tree = cKDTree(np.stack([ys, xs], axis=1))
    qy, qx = np.where((alpha > 0.01) & ~valid)
    if len(qy):
        _, nn = tree.query(np.stack([qy, qx], axis=1), k=1)
        P[qy, qx] = P[ys[nn], xs[nn]]
    return P
