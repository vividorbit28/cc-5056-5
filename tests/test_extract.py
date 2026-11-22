"""End-to-end checks for the landing-shadow extractor.

These use the generated samples (which carry true ``t``, ``q``, ``alpha`` and an
independent second plate) so the assertions are against ground truth, not
against the tool's own reconstruction.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.color import luminance, srgb_to_linear
from shadow_extract.extract import Params, extract, initial_product_mask, parse_support
from shadow_extract.evaluate import evaluate, evaluate_layers, evaluate_transfer
from shadow_extract.synth import SAMPLE_SPECS

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def _load(name):
    d = os.path.join(ROOT, name)
    e = cv2.cvtColor(cv2.imread(os.path.join(d, "empty.png")), cv2.COLOR_BGR2RGB)
    p = cv2.cvtColor(cv2.imread(os.path.join(d, "product.png")), cv2.COLOR_BGR2RGB)
    gt = dict(np.load(os.path.join(d, "gt.npz")))
    return e, p, gt


def _support(gt, pad=8):
    reg = (gt["alpha"] > 0.02) | (gt["t"] < 0.999)
    ys, xs = np.where(reg)
    m = np.zeros(reg.shape, np.uint8)
    cv2.rectangle(m, (xs.min() - pad, ys.min() - pad), (xs.max() + pad, ys.max() + pad), 1, -1)
    return parse_support(m, reg.shape)


@pytest.fixture(scope="module")
def cached():
    def get(name, method="joint"):
        key = (name, method)
        if key not in cached.store:
            e, p, gt = _load(name)
            dec = extract(e, p, support=_support(gt), method=method)
            cached.store[key] = (dec, gt)
        return cached.store[key]
    cached.store = {}
    return get


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

def test_diagnose_clean_pair(cached):
    from shadow_extract.diagnostics import diagnose
    from shadow_extract.extract import prepare_inputs
    e, p, gt = _load("soft_paper")
    el, pl = prepare_inputs(e, p)
    d = diagnose(el, pl, product_mask=gt["alpha"] > 0.5, support=_support(gt))
    assert abs(d.shift_px[0]) < 0.5 and abs(d.shift_px[1]) < 0.5
    assert max(abs(g - 1) for g in d.gain_rgb) < 0.06
    assert not d.background_changed


def test_diagnose_detects_perturbation():
    from shadow_extract.diagnostics import estimate_gain, estimate_shift
    d = os.path.join(ROOT, "soft_paper")
    e = cv2.cvtColor(cv2.imread(os.path.join(d, "empty.png")), cv2.COLOR_BGR2RGB)
    ep = cv2.cvtColor(cv2.imread(os.path.join(d, "empty_perturbed.png")), cv2.COLOR_BGR2RGB)
    p = cv2.cvtColor(cv2.imread(os.path.join(d, "product.png")), cv2.COLOR_BGR2RGB)
    e_lin = srgb_to_linear(e.astype(float) / 255)
    ep_lin = srgb_to_linear(ep.astype(float) / 255)
    p_lin = srgb_to_linear(p.astype(float) / 255)
    shift, _ = estimate_shift(ep_lin, p_lin)
    gain, _ = estimate_gain(ep_lin, p_lin)
    # The perturbed capture was shifted 1.3,-0.7 and gained (1.035,0.985,1.012)
    # relative to the clean one; the product matches the clean empty, so the
    # empty->product shift is the negative of the perturbation.
    assert abs(shift[0] + 1.3) < 0.9 or abs(shift[0] - 1.3) < 0.9
    assert abs(gain[0] - 1.035) < 0.05 or abs(gain[0] - 1 / 1.035) < 0.05


# --------------------------------------------------------------------------- #
# Segmentation
# --------------------------------------------------------------------------- #

def test_seed_rejects_neutral_product_shadow(cached):
    """The near-neutral mug's shadow must not be swallowed by the seed."""
    e, p, gt = _load("mug_light")
    from shadow_extract.extract import prepare_inputs
    el, pl = prepare_inputs(e, p)
    sup = _support(gt)
    seed = initial_product_mask(pl, el, sup)
    iou = (seed & (gt["alpha"] > 0.5)).sum() / max((seed | (gt["alpha"] > 0.5)).sum(), 1)
    assert iou > 0.85, f"seed IoU {iou:.3f}"


# --------------------------------------------------------------------------- #
# Extraction accuracy and layer validity
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("name", list(SAMPLE_SPECS))
def test_layers_accurate(cached, name):
    dec, gt = cached(name)
    L = evaluate_layers(dec, gt)
    assert L.t_mae < 0.08, (name, L.t_mae)
    assert L.alpha_iou > 0.88, (name, L.alpha_iou)
    assert L.q_mae < 0.04, (name, L.q_mae)


@pytest.mark.parametrize("name", list(SAMPLE_SPECS))
def test_transfer_beats_naive(cached, name):
    """On a *different* plate the layer must beat a copy-the-darkening baseline."""
    dec, gt = cached(name)
    T = evaluate_transfer(dec, gt)
    assert T.psnr > 35.0, (name, T.psnr)
    assert T.psnr >= T.psnr_alt - 0.5, (name, T.psnr, T.psnr_alt)


def test_shadow_is_bounded_and_not_black_overlay(cached):
    for name in SAMPLE_SPECS:
        dec, gt = cached(name)
        assert dec.t.min() >= dec.params.min_t - 1e-6
        assert dec.t.max() <= 1.0 + 1e-6
        assert np.isfinite(dec.q).all()
        assert dec.q.max() <= dec.params.q_max + 1e-5
        # A plain multiplicative layer would have q == 0 everywhere; require
        # real additive light while keeping it small.
        assert dec.q.max() > 1e-3


def test_no_texture_leak_into_shadow(cached):
    dec, gt = cached("textured_fabric")
    L = evaluate_layers(dec, gt)
    assert abs(L.texture_leak) < 0.35


def test_transfer_texture_is_plate_independent(cached):
    """Transferred shadow must not carry the *first* plate's texture."""
    dec, gt = cached("soft_paper")
    T = evaluate_transfer(dec, gt)
    assert T.artifact_ratio < 1.15


# --------------------------------------------------------------------------- #
# Method comparison
# --------------------------------------------------------------------------- #

def test_fit_only_matches_joint_on_t(cached):
    """The locked-base method is a valid alternative; both must recover t."""
    dec_j, gt = cached("soft_paper", "joint")
    dec_f, _ = cached("soft_paper", "fit_only")
    Lj = evaluate_layers(dec_j, gt)
    Lf = evaluate_layers(dec_f, gt)
    assert Lf.t_mae < 0.09, Lf.t_mae
    # The colour-identified joint solve should not be worse on the bounce term.
    assert Lj.q_mae <= Lf.q_mae + 0.01


# --------------------------------------------------------------------------- #
# Low-signal safety
# --------------------------------------------------------------------------- #

def test_dark_background_ratio_bounded():
    """I/B must not blow up: a near-black empty plate must not produce huge t/q."""
    e, p, gt = _load("dark_marble")
    e = (e.astype(np.float32) * 0.06).astype(np.uint8)  # push toward zero
    dec = extract(e, p, support=_support(gt))
    assert np.isfinite(dec.t).all() and np.isfinite(dec.q).all()
    assert dec.t.max() <= 1.0 + 1e-6
    # decomposed darkening is a fraction, never an unbounded ratio
    assert (1.0 - dec.t).max() <= 1.0 + 1e-6
