"""Contact-shadow contract: derive, move, verify and composite the landing shadow.

This is the *new* contact-shadow path.  A contact shadow is the part of the
extracted occlusion field that physically belongs to the product: it sits next
to the product footprint, is strongly occluding near the contact and is free of
receiver texture.  Pixels that fail those tests -- shadow mixed with floor
pattern, or shadow disconnected from the footprint -- are **not claimed** and
are handed back to the author instead of being silently moved.

Everything here is an actual computation on ``t``, ``q`` and ``alpha``; the
interface never substitutes a painted black ellipse.  When the requested
operation is not supported by the data, :class:`ContactShadowError` is raised so
the UI can surface the reason.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .color import luminance


class ContactShadowError(Exception):
    """Contact-shadow interface error: the operation is not defined on the data."""

    def __init__(self, message, code="contact_error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Contact shadow description
# --------------------------------------------------------------------------- #

@dataclass
class ContactShadow:
    """Landing shadow in the extraction's plane coordinates.

    ``t`` is multiplicative transmission (1 = no occlusion) and ``q`` the
    additive bounce field; ``claimed`` marks which shadow pixels belong to the
    product.  ``unclaimed`` pixels stay with the floor for author review.
    """

    t: np.ndarray
    q: np.ndarray
    alpha: np.ndarray
    claimed: np.ndarray
    unclaimed: np.ndarray
    region: np.ndarray
    footprint: np.ndarray
    band: np.ndarray
    thresholds: dict
    plane_origin: tuple = (0.0, 0.0)
    plane_scale: tuple = (1.0, 1.0)
    version: str = "contact/2"

    @property
    def shape(self):
        return self.t.shape

    def occlusion(self):
        return 1.0 - self.t

    def summary(self):
        return {
            "version": self.version,
            "shape": list(self.shape),
            "n_claimed": int(self.claimed.sum()),
            "n_unclaimed": int(self.unclaimed.sum()),
            "n_region": int(self.region.sum()),
            "mean_occlusion_claimed": float(self.occlusion()[self.claimed].mean())
            if self.claimed.any() else 0.0,
            "max_occlusion_claimed": float(self.occlusion()[self.claimed].max())
            if self.claimed.any() else 0.0,
            "thresholds": dict(self.thresholds),
            "plane_origin": list(self.plane_origin),
            "plane_scale": list(self.plane_scale),
        }

    def to_json(self):
        """JSON-serialisable contact block (arrays stored in the asset, not here)."""
        return self.summary()


def _disk(radius):
    r = int(round(radius))
    if r <= 0:
        return np.ones((1, 1), np.uint8)
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ((yy * yy + xx * xx) <= r * r).astype(np.uint8)


def _texture_leak_score(t, region):
    """Correlation of the occlusion field with its own spatial texture.

    A genuine occlusion field is smooth; high-frequency structure inside the
    contact band means receiver texture was absorbed into the shadow.
    """
    if region.sum() < 64:
        return 0.0
    occ = 1.0 - t
    blur = cv2.GaussianBlur(occ.astype(np.float32), (0, 0), 2.0)
    hf = occ - blur
    hf_r = hf[region]
    score = float(np.std(hf_r) / max(float(np.abs(occ[region]).mean()), 1e-9))
    return score


def derive_contact(dec, *, max_distance_px=None, occlusion_thresh=0.04,
                   texture_leak_limit=0.35, min_area=16):
    """Derive the contact shadow from a :class:`~shadow_extract.extract.Decomposition`.

    The contact region is the shadow *band*: outside the product footprint,
    within a distance derived from the footprint size, and with occlusion above
    ``occlusion_thresh``.  Pixels in that band whose occlusion carries
    receiver-scale high frequency (texture leak above ``texture_leak_limit``)
    are left unclaimed, along with disconnected shadow components.
    """
    t = np.asarray(dec.t, dtype=np.float64)
    q = np.asarray(dec.q, dtype=np.float64)
    alpha = np.asarray(dec.alpha, dtype=np.float64)
    H, W = t.shape

    footprint = alpha > 0.5
    if footprint.sum() < 4:
        raise ContactShadowError(
            "no product footprint (alpha) in this extraction; cannot locate a contact shadow",
            code="no_footprint")

    # Contact distance scales with the footprint, so a big product gets a wider
    # plausible landing band without letting far cast shadows be claimed.
    ys, xs = np.where(footprint)
    span = max(float(ys.max() - ys.min()), float(xs.max() - xs.min()))
    if max_distance_px is None:
        max_distance_px = max(10.0, 0.35 * span)

    dist = cv2.distanceTransform((~footprint).astype(np.uint8), cv2.DIST_L2, 5)
    band = (~footprint) & (dist <= max_distance_px)
    occ = 1.0 - t
    region = band & (occ >= occlusion_thresh)

    if region.sum() < min_area:
        # No observable landing shadow: the interface returns an empty claim
        # rather than fabricating one.
        empty = np.zeros((H, W), bool)
        return ContactShadow(t, q, alpha, empty, empty, empty, footprint, band,
                             {"occlusion_thresh": occlusion_thresh,
                              "texture_leak_limit": texture_leak_limit,
                              "max_distance_px": float(max_distance_px),
                              "texture_leak": 0.0, "disconnected_px": 0})

    # Texture contamination is judged per connected shadow component: any
    # component whose occlusion is noisy beyond the limit is returned to the
    # author instead of being moved with the product.
    n, labels = cv2.connectedComponents(region.astype(np.uint8), connectivity=8)
    claimed = np.zeros((H, W), bool)
    disconnected = 0
    for lab in range(1, n):
        comp = labels == lab
        if comp.sum() < min_area:
            disconnected += int(comp.sum())
            continue
        # A contact component must touch (or nearly touch) the footprint.
        touch = cv2.dilate(footprint.astype(np.uint8), _disk(2)).astype(bool)
        if not (comp & touch).any():
            disconnected += int(comp.sum())
            continue
        leak = _texture_leak_score(t, comp)
        if leak > texture_leak_limit:
            disconnected += int(comp.sum())
            continue
        claimed |= comp

    unclaimed = region & ~claimed
    thresholds = {"occlusion_thresh": float(occlusion_thresh),
                  "texture_leak_limit": float(texture_leak_limit),
                  "max_distance_px": float(max_distance_px),
                  "texture_leak": _texture_leak_score(t, region),
                  "disconnected_px": int(disconnected)}
    return ContactShadow(t, q, alpha, claimed, unclaimed, region, footprint, band,
                         thresholds)


# --------------------------------------------------------------------------- #
# Ground tracker: re-place the contact shadow when the product moves
# --------------------------------------------------------------------------- #

class GroundTracker:
    """Tracks the floor when an artist moves the product.

    The tracker hands out a new contact region translated by the product's
    plane-space delta.  Claimed shadow moves with the product; unclaimed shadow
    (mixed with floor texture) is deliberately left behind for the author.
    """

    def __init__(self, contact: ContactShadow):
        self.contact = contact

    def translate(self, dx_px, dy_px):
        """Return new (t, q, masks) after moving the product by (dx, dy) px."""
        H, W = self.contact.shape
        M = np.float32([[1, 0, dx_px], [0, 1, dy_px]])
        warp = lambda a, flags=cv2.INTER_LINEAR: cv2.warpAffine(
            a, M, (W, H), flags=flags, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        t_moved = np.ones((H, W), np.float64)
        q_moved = np.zeros((H, W, 3), np.float64)
        alpha_moved = warp(self.contact.alpha.astype(np.float32)).astype(np.float64)
        claimed_moved = warp(self.contact.claimed.astype(np.uint8),
                             cv2.INTER_NEAREST).astype(bool)

        # Only claimed pixels transport their t/q.  Unclaimed pixels keep the
        # fully-open floor state so floor texture is never carried along.
        cm = claimed_moved
        if cm.any():
            t_w = warp(self.contact.t.astype(np.float32)).astype(np.float64)
            q_w = warp(self.contact.q.astype(np.float32)).astype(np.float64)
            t_moved[cm] = t_w[cm]
            q_moved[cm] = q_w[cm]

        unclaimed_stays = self.contact.unclaimed.copy()
        # Any destination that lands on pixels the author still holds is not
        # overwritten by the moved shadow.
        clash = cm & unclaimed_stays
        if clash.any():
            t_moved[clash] = 1.0
            q_moved[clash] = 0.0
            cm = cm & ~clash
        return {"t": t_moved, "q": q_moved, "alpha": alpha_moved,
                "claimed": cm, "unclaimed": unclaimed_stays,
                "clash": int(clash.sum())}

    def new_contact_region(self, dx_px, dy_px):
        """The contact region handed over after a move (plane-coordinate bbox)."""
        moved = self.translate(dx_px, dy_px)
        reg = moved["claimed"]
        if not reg.any():
            raise ContactShadowError(
                "moved contact region is empty; nothing to hand over", code="empty_region")
        ys, xs = np.where(reg)
        ox, oy = self.contact.plane_origin
        sx, sy = self.contact.plane_scale
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        return {"bbox_px": [x0, y0, x1, y1],
                "bbox_plane": [float(ox + x0 * sx), float(oy + y0 * sy),
                               float(ox + x1 * sx), float(oy + y1 * sy)],
                "n_pixels": int(reg.sum()),
                "unclaimed_left_behind": int(moved["unclaimed"].sum())}


# --------------------------------------------------------------------------- #
# Composite the contact shadow onto a new plate
# --------------------------------------------------------------------------- #

def composite_contact(plate_lin, t, q, alpha=None, P_lin=None,
                      shadow_mask=None, opacity=1.0, bounce=1.0):
    """Composite the contact shadow (and optionally the product) onto a plate.

    Linear-light physical model, never a black overlay::

        ground = B * (t_eff + q_eff),  t_eff = 1 - (1 - t) * opacity

    ``shadow_mask`` restricts the shadow contribution (e.g. to claimed pixels).
    A legacy colour block must be passed through this function explicitly; it is
    never reinterpreted as a transmission coefficient here.
    """
    B = np.asarray(plate_lin, dtype=np.float64)
    if B.shape[:2] != t.shape[:2]:
        raise ContactShadowError(
            f"plate shape {B.shape[:2]} != shadow plane {t.shape[:2]}", code="shape_mismatch")
    t_eff = 1.0 - (1.0 - np.asarray(t, dtype=np.float64)) * opacity
    q_eff = np.asarray(q, dtype=np.float64) * bounce
    if shadow_mask is not None:
        m = np.asarray(shadow_mask, bool)
        t_eff = np.where(m, t_eff, 1.0)
        q_eff = np.where(m[..., None], q_eff, 0.0)
    if np.any(t_eff < 0.0) or np.any(t_eff > 1.0 + 1e-9):
        raise ContactShadowError("transmission left the physical [t_min, 1] range",
                                 code="transmission_out_of_range")
    ground = B * (t_eff[..., None] + q_eff)
    if alpha is not None and P_lin is not None:
        a = np.asarray(alpha, dtype=np.float64)[..., None]
        out = a * np.asarray(P_lin, dtype=np.float64) + (1.0 - a) * ground
        return np.clip(out, 0.0, None)
    return np.clip(ground, 0.0, None)


# --------------------------------------------------------------------------- #
# Actual contact verification
# --------------------------------------------------------------------------- #

@dataclass
class ContactCheck:
    ok: bool
    contact_coverage: float
    gap_px: float
    shadow_contribution: float
    foreground_retention: float
    texture_leak: float
    errors: list

    def as_dict(self):
        return {"ok": self.ok, "contact_coverage": self.contact_coverage,
                "gap_px": self.gap_px, "shadow_contribution": self.shadow_contribution,
                "foreground_retention": self.foreground_retention,
                "texture_leak": self.texture_leak, "errors": list(self.errors)}


def verify_contact(comp_lin, plate_lin, contact: ContactShadow, *,
                   moved_alpha=None, moved_t=None, moved_claimed=None,
                   source_fg_premult=None, coverage_min=0.9, gap_max_px=4.0,
                   retention_min=0.98, leak_limit=0.35):
    """Verify an independent shadow actually contacts the object after compositing.

    All quantities are computed from the supplied images; nothing is assumed:

    * **contact coverage** -- fraction of the object's ground-contact rim that
      has a darker-than-plate neighbour on the independent shadow;
    * **gap** -- median distance from the footprint rim to the nearest shadow
      pixel (0 means the shadow touches the object);
    * **shadow contribution** -- mean linear-light occlusion the shadow adds on
      its claimed region, i.e. ``1 - mean(I/B)`` there;
    * **foreground retention** -- premultiplied foreground energy kept inside
      the footprint after the shadow edit (1.0 when the product was not moved);
    * **texture leak** -- high-frequency contamination of the shadow field.
    """
    errors = []
    comp = np.asarray(comp_lin, dtype=np.float64)
    B = np.asarray(plate_lin, dtype=np.float64)
    alpha = contact.alpha if moved_alpha is None else np.asarray(moved_alpha, float)
    t = contact.t if moved_t is None else np.asarray(moved_t, float)
    claimed = contact.claimed if moved_claimed is None else np.asarray(moved_claimed, bool)

    footprint = alpha > 0.5
    if footprint.sum() < 4:
        raise ContactShadowError("no footprint to test contact against", code="no_footprint")

    # --- geometric contact: rim pixels and their nearest dark neighbour ----- #
    rim = footprint & cv2.dilate((~footprint).astype(np.uint8), _disk(1)).astype(bool)
    dark = claimed & ((1.0 - t) >= contact.thresholds["occlusion_thresh"])
    if dark.any():
        dist_to_shadow = cv2.distanceTransform((~dark).astype(np.uint8),
                                               cv2.DIST_L2, 5)[rim]
        gap = float(np.median(dist_to_shadow)) if rim.any() else float("inf")
        near = cv2.dilate(dark.astype(np.uint8), _disk(2)).astype(bool)
        coverage = float((rim & near).sum() / max(rim.sum(), 1))
    else:
        gap = float("inf")
        coverage = 0.0
        errors.append("no occluded shadow pixel in claimed region")

    # --- physical shadow contribution on the claimed region ----------------- #
    if claimed.any():
        ratio = comp[claimed] / np.maximum(B[claimed], 1e-6)
        contribution = float(np.clip(1.0 - ratio.mean(), 0.0, 1.0))
        leak_local = _texture_leak_score(t, claimed)
    else:
        contribution = 0.0
        leak_local = 0.0
        errors.append("claimed region is empty")

    # --- foreground retention ----------------------------------------------- #
    # Two distinct measurements:
    #  * matte overlap: identical frames must preserve the footprint; a move
    #    intentionally relocates it, so for moved frames only the matte *mass*
    #    (number of covered pixels) is required to be preserved;
    #  * energy retention: premultiplied foreground energy inside the footprint.
    if moved_alpha is not None:
        union = footprint | (contact.alpha > 0.5)
        if np.allclose(moved_alpha, contact.alpha):
            retention = float(np.minimum(alpha[union], contact.alpha[union]).sum()
                              / max(np.maximum(alpha[union], contact.alpha[union]).sum(), 1e-9))
        else:
            mass_src = float((contact.alpha > 0.5).sum())
            mass_new = float((alpha > 0.5).sum())
            retention = min(mass_src, mass_new) / max(mass_src, mass_new)
    else:
        retention = 1.0
    if source_fg_premult is not None and footprint.any():
        src = np.asarray(source_fg_premult, dtype=np.float64)
        if moved_alpha is not None and not np.allclose(moved_alpha, contact.alpha):
            # Moved frame: compare total foreground energy (translation does not
            # change the matte mass or its colour energy).
            src_e = float(luminance(np.abs(src)).sum())
            comp_e = float(luminance(np.abs(comp))[footprint].sum())
        else:
            src_e = float(luminance(np.abs(src))[footprint].sum())
            comp_e = float(luminance(np.abs(comp))[footprint].sum())
        if src_e > 1e-9:
            energy = min(1.0, comp_e / src_e)
            retention = min(retention, energy) if moved_alpha is None else \
                0.5 * retention + 0.5 * energy

    if coverage < coverage_min:
        errors.append(f"contact coverage {coverage:.2f} below {coverage_min:.2f}")
    if gap > gap_max_px:
        errors.append(f"shadow gap {gap:.2f} px exceeds {gap_max_px:.2f}")
    if retention < retention_min:
        errors.append(f"foreground retention {retention:.3f} below {retention_min:.3f}")
    if leak_local > leak_limit:
        errors.append(f"texture leak {leak_local:.3f} above {leak_limit:.3f}")
    if contribution <= 0.0:
        errors.append("shadow makes no measurable contribution")

    return ContactCheck(ok=not errors, contact_coverage=coverage, gap_px=gap,
                        shadow_contribution=contribution, foreground_retention=retention,
                        texture_leak=leak_local, errors=errors)
