"""Tests for the contact-shadow contract (new path).

Covers contact derivation, the ground tracker hand-over, claimed vs returned
(to-author) pixels, compositing, and the actual contact/foreground checks.
All data is synthesised in the test; nothing depends on hand-edited files.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from shadow_extract.asset import premultiply
from shadow_extract.contact import (ContactShadowError, GroundTracker,
                                    composite_contact, derive_contact,
                                    verify_contact)


class _Dec:
    pass


def _scene(H=80, W=100, textured=False, gap=False, seed=1, occ_peak=0.55):
    rng = np.random.default_rng(seed)
    t = np.ones((H, W))
    q = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    alpha[30:50, 40:60] = 1.0
    P = np.full((H, W, 3), 0.7)
    yy, xx = np.mgrid[0:H, 0:W]
    if gap:
        # A detached shadow ring well outside any plausible contact band.
        ring = ((np.abs(xx - 50) >= 24) & (np.abs(xx - 50) <= 30)) | \
               ((np.abs(yy - 40) >= 22) & (np.abs(yy - 40) <= 28))
        occ = np.where(ring, 0.5, 0.0)
    else:
        d = np.maximum(np.abs(xx - 50) - 10, 0) + np.maximum(np.abs(yy - 40) - 10, 0)
        occ = np.clip(occ_peak * np.exp(-d / 6.0), 0.0, 1.0)
    if textured:
        occ = np.clip(occ + 0.3 * (rng.random((H, W)) - 0.5), 0.0, 1.0)
    t = 1.0 - occ
    d = _Dec()
    d.t, d.q, d.alpha, d.P_lin = t, q, alpha, P
    d.method, d.info = "joint", {}
    return d, rng


def test_derive_claims_smooth_contact_and_bounds():
    dec, _ = _scene()
    c = derive_contact(dec)
    assert c.claimed.sum() > 100
    assert c.unclaimed.sum() == 0
    assert c.claimed[30:50, 40:60].sum() == 0  # never claims the footprint
    # Claimed occlusion is real.
    assert c.occlusion()[c.claimed].mean() > 0.05
    assert c.summary()["version"] == "contact/2"


def test_textured_shadow_is_returned_to_author():
    dec, _ = _scene(textured=True)
    c = derive_contact(dec, texture_leak_limit=0.2)
    # Noisy components fail the texture test and are returned, not moved.
    assert c.unclaimed.sum() > 0
    assert (c.claimed & c.unclaimed).sum() == 0


def test_disconnected_shadow_not_claimed():
    dec, _ = _scene(gap=True)
    c = derive_contact(dec, max_distance_px=12.0)
    # The detached ring sits outside the contact band and is not claimed.
    assert c.claimed.sum() == 0
    assert (c.claimed & c.footprint).sum() == 0


def test_no_footprint_raises_interface_error():
    dec, _ = _scene()
    dec.alpha = np.zeros_like(dec.alpha)
    with pytest.raises(ContactShadowError, match="footprint"):
        derive_contact(dec)


def test_ground_tracker_hands_over_new_region_and_leaves_unclaimed():
    dec, _ = _scene()
    c = derive_contact(dec)
    tracker = GroundTracker(c)
    region = tracker.new_contact_region(12, 0)
    assert region["n_pixels"] == int(c.claimed.sum())
    moved = tracker.translate(12, 0)
    # The shadow physically shifted.
    ys, xs = np.where(moved["claimed"])
    ys0, xs0 = np.where(c.claimed)
    assert xs.mean() - xs0.mean() == pytest.approx(12, abs=1.5)
    # Transmission outside the moved claim stays fully open (no floor carried).
    assert np.allclose(moved["t"][~moved["claimed"]], 1.0)
    assert np.allclose(moved["q"][~moved["claimed"]], 0.0)


def test_moved_textured_part_stays_behind():
    dec, _ = _scene(textured=True)
    c = derive_contact(dec, texture_leak_limit=0.2)
    moved = GroundTracker(c).translate(8, 0)
    # Unclaimed pixels are never transported; the left-behind set is intact.
    assert moved["unclaimed"].sum() == c.unclaimed.sum()
    # The moved claim contains none of the left-behind pixels.
    assert (moved["claimed"] & c.unclaimed).sum() == 0


def test_composite_uses_physical_model_not_black_overlay():
    dec, _ = _scene()
    c = derive_contact(dec)
    plate = np.full(dec.t.shape + (3,), 0.5)
    ground = composite_contact(plate, dec.t, dec.q, shadow_mask=c.claimed)
    # On a 0.5 plate the shadowed ground must equal 0.5 * t (darker, not black).
    expected = 0.5 * dec.t[..., None]
    assert np.allclose(ground[c.claimed], expected[c.claimed], atol=1e-6)
    assert ground[c.claimed].min() > 0.0  # never a black rectangle


def test_legacy_color_is_not_used_as_transmission():
    dec, _ = _scene()
    c = derive_contact(dec)
    plate = np.full(dec.t.shape + (3,), 0.5)
    # A bogus "shadow colour" passed as q must not be silently turned into t.
    bad_q = np.full_like(dec.q, 0.3)
    ground = composite_contact(plate, dec.t, bad_q, shadow_mask=c.claimed)
    # t drives the multiplicative term unchanged; q only adds.
    assert np.allclose(ground[c.claimed, 0],
                       0.5 * (dec.t[c.claimed] + 0.3), atol=1e-6)


def test_shape_mismatch_raises():
    dec, _ = _scene()
    c = derive_contact(dec)
    with pytest.raises(ContactShadowError, match="shape"):
        composite_contact(np.zeros((40, 40, 3)), dec.t, dec.q, shadow_mask=c.claimed)


def test_verify_contact_passes_for_real_contact():
    dec, _ = _scene()
    c = derive_contact(dec)
    plate = np.full(dec.t.shape + (3,), 0.5)
    comp = composite_contact(plate, dec.t, dec.q, alpha=dec.alpha, P_lin=dec.P_lin,
                             shadow_mask=c.claimed)
    check = verify_contact(comp, plate, c,
                           source_fg_premult=premultiply(dec.P_lin, dec.alpha))
    assert check.ok, check.errors
    assert check.shadow_contribution > 0.05
    assert check.foreground_retention >= 0.98
    assert check.gap_px <= 2.0


def test_verify_detects_gap_when_shadow_removed():
    dec, _ = _scene()
    c = derive_contact(dec)
    plate = np.full(dec.t.shape + (3,), 0.5)
    empty_mask = np.zeros_like(c.claimed)
    comp = composite_contact(plate, np.ones_like(dec.t), np.zeros_like(dec.q),
                             alpha=dec.alpha, P_lin=dec.P_lin, shadow_mask=empty_mask)
    edited = c.__class__(np.ones_like(c.t), c.q, c.alpha, empty_mask, c.unclaimed,
                         c.region, c.footprint, c.band, c.thresholds)
    check = verify_contact(comp, plate, edited)
    assert not check.ok
    assert any("coverage" in e or "gap" in e or "contribution" in e for e in check.errors)


def test_foreground_retention_detects_matte_growth():
    dec, _ = _scene()
    c = derive_contact(dec)
    plate = np.full(dec.t.shape + (3,), 0.5)
    grown_alpha = dec.alpha.copy()
    grown_alpha[20:55, 30:65] = 1.0  # matte grew beyond the product
    comp = composite_contact(plate, dec.t, dec.q, alpha=grown_alpha, P_lin=dec.P_lin,
                             shadow_mask=c.claimed)
    check = verify_contact(comp, plate, c, moved_alpha=grown_alpha, moved_t=dec.t,
                           moved_claimed=c.claimed)
    assert not check.ok
    assert any("retention" in e for e in check.errors)


def test_transparent_edge_and_dark_floor_materials():
    # Soft (semi-transparent) matte edge and a very dark floor must still
    # produce bounded, finite contact fields.
    dec, _ = _scene(occ_peak=0.4)
    dec.alpha = dec.alpha * 0.6 + 0.4
    dec.alpha = np.clip(dec.alpha, 0, 1)
    c = derive_contact(dec)
    dark_plate = np.full(dec.t.shape + (3,), 0.02)
    comp = composite_contact(dark_plate, dec.t, dec.q, alpha=dec.alpha, P_lin=dec.P_lin,
                             shadow_mask=c.claimed)
    assert np.isfinite(comp).all()
    check = verify_contact(comp, dark_plate, c,
                           source_fg_premult=premultiply(dec.P_lin, dec.alpha),
                           retention_min=0.6)
    assert check.shadow_contribution >= 0.0
