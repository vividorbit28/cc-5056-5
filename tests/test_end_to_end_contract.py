"""End-to-end migration: old intermediates run the full new contact path.

These tests take the legacy/current intermediate contract (a ``gt.npz`` with
merged arrays, or a single-channel baked map) and drive it all the way through
migration -> contact derivation -> move -> composite -> verification, mapping
the shadow contribution and foreground retention to concrete numbers.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.asset import load_asset
from shadow_extract.contact import (ContactShadow, GroundTracker,
                                    composite_contact, verify_contact)
from shadow_extract.legacy import migrate_legacy, reopen_underdetermined
from shadow_extract.migrate import migrate_gt_npz, verify_build_lock, write_build_lock


def _fields(H=70, W=90):
    alpha = np.zeros((H, W))
    alpha[25:45, 35:55] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - 45) - 10, 0) + np.maximum(np.abs(yy - 35) - 10, 0)
    t = 1.0 - np.clip(0.55 * np.exp(-d / 6.0), 0, 1)
    q = np.zeros((H, W, 3))
    P = np.full((H, W, 3), 0.7)
    return alpha, t, q, P


def test_old_sample_runs_new_contact_path_to_completion(tmp_path):
    alpha, t, q, P = _fields()
    sdir = tmp_path / "sample"
    sdir.mkdir()
    np.savez(str(sdir / "gt.npz"), t=t, q=q, alpha=alpha, P=P,
             B1=np.full(t.shape + (3,), 0.5), I1=np.full(t.shape + (3,), 0.4))
    cv2.imwrite(str(sdir / "empty.png"), np.full(t.shape + (3,), 128, np.uint8))
    bundle = tmp_path / "assets" / "sample"
    loaded, contact = migrate_gt_npz(
        str(sdir / "gt.npz"), str(bundle), name="sample",
        source_files={"empty": str(sdir / "empty.png")})

    # Re-open from disk: the contract must survive a round trip.
    reopened = load_asset(str(bundle))
    t2, q2, alpha2 = (reopened["layers"]["t"], reopened["layers"]["q"],
                      reopened["layers"]["alpha"])
    claimed = reopened["layers"]["contact_unclaimed"] == 0
    claimed = claimed & ((1.0 - t2) >= 0.04) & (alpha2 < 0.5)

    # Ground tracker hands over a moved region.
    contact2 = ContactShadow(t2, q2, alpha2, claimed,
                             reopened["layers"]["contact_unclaimed"].astype(bool),
                             contact.region, contact.footprint, contact.band,
                             contact.thresholds)
    moved = GroundTracker(contact2).translate(10, 0)

    # Composite on two different floors and actually verify contact.
    for floor in (0.35, 0.75):
        plate = np.full(t.shape + (3,), floor)
        comp = composite_contact(plate, moved["t"], moved["q"],
                                 alpha=moved["alpha"], P_lin=P,
                                 shadow_mask=moved["claimed"])
        check = verify_contact(comp, plate, contact2,
                               moved_alpha=moved["alpha"], moved_t=moved["t"],
                               moved_claimed=moved["claimed"],
                               source_fg_premult=alpha[..., None] * P,
                               coverage_min=0.3, gap_max_px=6.0)
        # Evidence mapping: both required semantics carry numeric evidence.
        assert check.shadow_contribution > 0.05
        assert check.foreground_retention > 0.9
    # Lock the migration and confirm the clean-directory lock still verifies.
    lock_path, _ = write_build_lock(str(tmp_path / "assets"),
                                    [{"name": "sample", "bundle": str(bundle)}])
    ok, mismatches = verify_build_lock(str(tmp_path / "assets"))
    assert ok and not mismatches


def test_underdetermined_legacy_kept_editable_then_verified(tmp_path):
    alpha, t, q, P = _fields()
    # Merged-luma old file, no plate: underdetermined.
    legacy = tmp_path / "old.npz"
    np.savez(str(legacy), luma=t, alpha=alpha)
    src = tmp_path / "empty.png"
    cv2.imwrite(str(src), np.full(t.shape + (3,), 128, np.uint8))
    bundle = tmp_path / "legacy_asset"
    loaded, report = migrate_legacy(str(legacy), str(bundle), name="old",
                                    source_files={"empty": str(src)})
    assert report["paths"] == ["underdetermined:merged_luma"]

    # Author reopens, edits the preserved layer, and the bundle re-locks.
    reopened, luma = reopen_underdetermined(str(bundle))
    from shadow_extract.legacy import save_underdetermined_edit
    relocked = save_underdetermined_edit(str(bundle), np.clip(luma * 0.9, 0, 1))
    assert load_asset(str(bundle)) is not None
    # The transmission slot was never filled with a guessed colour.
    assert np.allclose(relocked["layers"]["t"], 1.0)
    assert relocked["manifest"]["layers"][0]["status"] == "pending_explanation"


def test_shadow_evidence_is_not_default_masked(tmp_path):
    """Contribution/retention numbers must come from pixels, not constants."""
    alpha, t, q, P = _fields()
    sdir = tmp_path / "s"
    sdir.mkdir()
    np.savez(str(sdir / "gt.npz"), t=t, q=q, alpha=alpha, P=P)
    cv2.imwrite(str(sdir / "empty.png"), np.full(t.shape + (3,), 128, np.uint8))
    bundle = tmp_path / "a"
    _, contact = migrate_gt_npz(str(sdir / "gt.npz"), str(bundle), name="s",
                                source_files={"empty": str(sdir / "empty.png")})
    plate = np.full(t.shape + (3,), 0.5)
    comp = composite_contact(plate, t, q, alpha=alpha, P_lin=P,
                             shadow_mask=contact.claimed)
    check = verify_contact(comp, plate, contact,
                           source_fg_premult=alpha[..., None] * P)
    # Removing the shadow must change the measured contribution (proves it is
    # computed, not a default).
    comp_flat = composite_contact(plate, np.ones_like(t), np.zeros_like(q),
                                  alpha=alpha, P_lin=P, shadow_mask=contact.claimed)
    flat = verify_contact(comp_flat, plate, contact,
                          source_fg_premult=alpha[..., None] * P)
    assert check.shadow_contribution > flat.shadow_contribution
    assert flat.shadow_contribution == pytest.approx(0.0, abs=1e-9)


def test_two_floor_export_distinguishes_carried_tile(tmp_path):
    """A re-lit shadow tracks each floor; a carried old tile does not."""
    from scipy import ndimage
    from shadow_extract.color import luminance
    from shadow_extract.plates import make_plate

    alpha, t, q, P = _fields()
    rng = np.random.default_rng(4)
    floor_a = make_plate(t.shape, rng, "wood", (0.5, 0.38, 0.28), 0.34)
    floor_b = make_plate(t.shape, np.random.default_rng(9), "marble",
                         (0.55, 0.55, 0.6), 0.30)

    class _C:
        pass

    dec = _C()
    dec.t, dec.q, dec.alpha, dec.P_lin = t, q, alpha, P
    from shadow_extract.contact import derive_contact
    contact = derive_contact(dec)
    m = contact.claimed

    def hf(img):
        l = luminance(img)
        return (l - ndimage.gaussian_filter(l, 2.0))[m]

    def corr(x, y):
        x, y = x - x.mean(), y - y.mean()
        return float((x * y).mean() / max(float(x.std() * y.std()), 1e-12))

    good_a = composite_contact(floor_a, t, q, shadow_mask=m)
    good_b = composite_contact(floor_b, t, q, shadow_mask=m)
    good_corr = min(corr(hf(good_a), hf(floor_a)),
                    corr(hf(good_b), hf(floor_b)))

    # Negative control: carry the *old floor's own high-frequency tile* into
    # floor B's shadow region (the failure mode where the tile moves with the
    # product).  The carried pattern is uncorrelated with floor B's texture.
    old_tile_hf = hf(floor_a)
    carried_b = floor_b.copy()
    lum_b = luminance(carried_b)
    flat = ndimage.gaussian_filter(lum_b, 2.0)
    vals = flat[m] + old_tile_hf  # replace floor B texture with floor A's
    lum_b[m] = vals
    carried_b = np.repeat(lum_b[..., None], 3, axis=2)
    bad_corr = corr(hf(carried_b), hf(floor_b))
    assert good_corr > bad_corr + 0.15
    assert bad_corr < 0.2
