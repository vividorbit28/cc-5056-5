"""Flask application: upload a pair, mark a support region, extract and re-light.

The UI deliberately shows the three physical quantities the tool claims to
separate, each on its own panel:

* **foreground** -- the product's matte ``alpha`` and colour ``P`` over a
  checkerboard, so a too-large matte or a painted dark rim is obvious;
* **shadow strength** -- ``1 - t`` (occlusion) and the additive bounce ``q``
  separately, because one is a multiplicative attenuation and the other is
  added light, and they re-light differently;
* **unexplained** -- the residual that neither layer explains.  If this panel is
  not empty on ground, the pair is not consistent with a plain shadow and the
  tool must not pretend otherwise.

Re-backgrounding is parametric: the user picks a new plate (a generated one, an
upload, or a solid colour) plus an overall shadow-opacity and bounce scale, and
sees the layer re-lit on it.
"""
from __future__ import annotations

import io
import json
import os
import threading
import uuid

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_file

from .color import linear_to_srgb, srgb_to_linear
from .contact import (ContactShadowError, GroundTracker, composite_contact,
                      derive_contact, verify_contact)
from .extract import Params, extract
from .plates import make_plate
from .visualize import (checkerboard, composite_srgb, png_bytes, scalar_map,
                        to_u8_srgb)


class ContactApiError(Exception):
    def __init__(self, message, code, status=400):
        super().__init__(message)
        self.code = code
        self.status = status

app = Flask(__name__)

_SESSIONS = {}
_LOCK = threading.Lock()
_MAX_SESSIONS = 32


@app.errorhandler(ContactApiError)
def _handle_contact_api_error(err):
    return jsonify(error=str(err), code=err.code), err.status


# --------------------------------------------------------------------------- #
# Image helpers
# --------------------------------------------------------------------------- #

def _read_upload(file_storage):
    data = np.frombuffer(file_storage.read(), np.uint8)
    bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("could not decode image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _fit(a, shape):
    h, w = shape
    if a.shape[:2] == (h, w):
        return a
    return cv2.resize(a, (w, h), interpolation=cv2.INTER_AREA)


def _srgb_lin(img_u8):
    return srgb_to_linear(img_u8.astype(np.float64) / 255.0)


def _u8_lin(img_lin):
    return to_u8_srgb(img_lin)


def _alpha_over(a, rgb, bg):
    a = np.asarray(a, float)[..., None]
    return a * rgb + (1 - a) * bg


def _b64(u8_rgb):
    import base64
    return base64.b64encode(png_bytes(u8_rgb)).decode("ascii")


# --------------------------------------------------------------------------- #
# Panel rendering
# --------------------------------------------------------------------------- #

def _checker_like(B):
    h, w = B.shape[:2]
    return checkerboard(h, w, cell=max(8, h // 24))


def _panels(sess, opacity=1.0, bounce=1.0):
    """Render the diagnostic panels for the current extraction."""
    dec = sess["dec"]
    B = dec.B
    h, w = B.shape[:2]
    cb = _checker_like(B)

    # Foreground on checkerboard: shadows are deliberately omitted here so the
    # matte/colour can be judged alone.
    fg = _alpha_over(dec.alpha, dec.P_lin, cb)
    fg_u8 = _u8_lin(fg)

    # Shadow strength: 1-t occlusion, and q scaled by the opacity/bounce knobs.
    t_eff = 1.0 - (1.0 - dec.t) * opacity
    q_eff = dec.q * bounce
    occl_u8 = scalar_map(1.0 - t_eff, 0.0, 1.0, cv2.COLORMAP_INFERNO)
    # q is only defined on ground: inside the product silhouette it is a harmonic
    # interpolation, so showing it there would read as a shadow that is not
    # observed. Mask it and scale the colour ramp from the visible ground.
    ground = dec.alpha < 0.5
    qmag = np.linalg.norm(q_eff, axis=2) * ground
    q_ref = qmag[ground] if ground.any() else qmag
    qmax = max(float(np.percentile(q_ref, 99.5)) if q_ref.size else 0.0, 1e-4)
    q_u8 = scalar_map(qmag, 0.0, qmax, cv2.COLORMAP_TURBO)
    q_u8[~ground] = 0

    resid = dec.residual
    rmag = np.abs(resid).mean(axis=2)
    u_lin = (dec.unexplained.astype(float) * np.maximum(rmag, 0.0))
    u_u8 = scalar_map(u_lin, 0.0, max(float(np.percentile(u_lin, 99.5)), 1e-4),
                      cv2.COLORMAP_MAGMA)

    return fg_u8, occl_u8, q_u8, u_u8


def _composite(sess, plate_u8, opacity, bounce, use_product=True):
    """Parametric re-composite onto a new plate."""
    dec = sess["dec"]
    h, w = dec.B.shape[:2]
    B2 = srgb_to_linear(_fit(plate_u8, (h, w)).astype(np.float64) / 255.0)
    t_eff = 1.0 - (1.0 - dec.t) * opacity
    q_eff = dec.q * bounce
    ground2 = B2 * (t_eff[..., None] + q_eff)
    if use_product:
        out = dec.alpha[..., None] * dec.P_lin + (1 - dec.alpha[..., None]) * ground2
    else:
        out = ground2
    return _u8_lin(out)


# --------------------------------------------------------------------------- #
# Contact shadow (new path)
# --------------------------------------------------------------------------- #

def _contact(sess):
    """Return the session's contact shadow, deriving it once and caching it."""
    if sess.get("contact") is None:
        dec = sess["dec"]
        sess["contact"] = derive_contact(
            dec,
            occlusion_thresh=float(sess.get("contact_occlusion", 0.04)),
            texture_leak_limit=float(sess.get("contact_leak_limit", 0.35)))
    return sess["contact"]


def _contact_mask(sess):
    contact = _contact(sess)
    return contact.claimed


def _contact_composite(sess, plate_u8, opacity=1.0, bounce=1.0, use_product=True,
                       claimed_only=True):
    """Re-composite through the contact-shadow implementation."""
    dec = sess["dec"]
    h, w = dec.B.shape[:2]
    B2 = srgb_to_linear(_fit(plate_u8, (h, w)).astype(np.float64) / 255.0)
    mask = _contact_mask(sess) if claimed_only else None
    out = composite_contact(B2, dec.t, dec.q,
                            alpha=dec.alpha if use_product else None,
                            P_lin=dec.P_lin if use_product else None,
                            shadow_mask=mask, opacity=opacity, bounce=bounce)
    return _u8_lin(out)


def _plate_from_request(sess, spec):
    spec = spec or {}
    kind = spec.get("kind", "original")
    if kind == "solid":
        col = np.array(spec.get("color", [200, 200, 200]), float) / 255.0
        img = np.empty(sess["dec"].B.shape, np.float64)
        img[...] = col
        return to_u8_srgb(img)
    if kind == "generated":
        rng = np.random.default_rng(int(spec.get("seed", 0)))
        beta = make_plate(sess["dec"].B.shape[:2], rng,
                          kind=spec.get("style", "paper"),
                          base_rgb=tuple(spec.get("base", (0.62, 0.58, 0.52))),
                          contrast=float(spec.get("contrast", 0.22)))
        return to_u8_srgb(beta)
    return sess["empty"]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/extract", methods=["POST"])
def api_extract():
    empty = _read_upload(request.files["empty"])
    product = _read_upload(request.files["product"])
    if empty.shape[:2] != product.shape[:2]:
        product = cv2.resize(product, (empty.shape[1], empty.shape[0]),
                             interpolation=cv2.INTER_AREA)
    support = None
    if request.form.get("support"):
        pts = json.loads(request.form["support"])
        if len(pts) >= 3:
            m = np.zeros(empty.shape[:2], np.uint8)
            cv2.fillPoly(m, [np.asarray(pts, np.int32)], 1)
            support = m.astype(bool)
    method = request.form.get("method", "joint")
    params = Params()
    for key in ("t_smooth", "q_smooth", "min_t", "q_max"):
        val = request.form.get(key)
        if val not in (None, ""):
            setattr(params, key, float(val))

    dec = extract(empty, product, support=support, method=method, params=params)

    sid = uuid.uuid4().hex
    with _LOCK:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)))
        _SESSIONS[sid] = dict(dec=dec, empty=empty, product=product, contact=None)

    fg_u8, occl_u8, q_u8, u_u8 = _panels(_SESSIONS[sid])
    comp = _composite(_SESSIONS[sid], empty, 1.0, 1.0)

    d = dec.diag
    info = dict(
        session=sid,
        method=method,
        diag=d.as_dict(),
        info=dec.info,
        panels={
            "foreground": _b64(fg_u8),
            "occlusion": _b64(occl_u8),
            "bounce": _b64(q_u8),
            "unexplained": _b64(u_u8),
            "original": _b64(empty),
            "capture": _b64(product),
            "reconstruct": _b64(comp),
        },
    )
    return jsonify(info)


@app.route("/api/compose", methods=["POST"])
def api_compose():
    body = request.get_json(force=True)
    sid = body.get("session")
    sess = _SESSIONS.get(sid)
    if sess is None:
        return jsonify(error="unknown session"), 404
    opacity = float(body.get("opacity", 1.0))
    bounce = float(body.get("bounce", 1.0))
    use_product = bool(body.get("product", True))
    plate_u8 = _plate_from_request(sess, body.get("plate"))
    # The composition canvas always goes through the contact-shadow path.
    out = _contact_composite(sess, plate_u8, opacity, bounce, use_product,
                             claimed_only=bool(body.get("claimed_only", True)))
    return jsonify(image=_b64(out))


@app.route("/api/contact/<name>", methods=["POST"])
def api_contact(name):
    """Derive and report the contact shadow for a sample session."""
    sess = _get_sample_session(name)
    body = request.get_json(silent=True) or {}
    occlusion = float(body.get("occlusion", 0.04))
    leak_limit = float(body.get("leak_limit", 0.35))
    try:
        contact = derive_contact(sess["dec"], occlusion_thresh=occlusion,
                                 texture_leak_limit=leak_limit)
    except ContactShadowError as exc:
        return jsonify(error=str(exc), code=exc.code), 409
    sess["contact"] = contact
    return jsonify(contact=contact.summary(),
                   region=_b64(_contact_overlay(sess)))


def _get_sample_session(name):
    d = os.path.join(_SAMPLES_DIR, name)
    if not os.path.isfile(os.path.join(d, "empty.png")):
        raise ContactApiError("sample not found", "not_found", 404)
    sid = "__contact__" + name
    with _LOCK:
        sess = _SESSIONS.get(sid)
        if sess is None or sess.get("name") != name:
            empty = cv2.cvtColor(cv2.imread(os.path.join(d, "empty.png")),
                                 cv2.COLOR_BGR2RGB)
            product = cv2.cvtColor(cv2.imread(os.path.join(d, "product.png")),
                                   cv2.COLOR_BGR2RGB)
            dec = extract(empty, product, method="joint")
            sess = dict(dec=dec, empty=empty, product=product, name=name)
            _SESSIONS[sid] = sess
    return sess


def _contact_overlay(sess):
    """RGB overlay: claimed contact green, unclaimed (returned) amber."""
    contact = _contact(sess)
    base = sess["empty"].copy()
    overlay = base.astype(np.float32)
    if contact.claimed.any():
        overlay[contact.claimed] = 0.55 * overlay[contact.claimed] + 0.45 * np.array([60, 200, 120])
    if contact.unclaimed.any():
        overlay[contact.unclaimed] = 0.55 * overlay[contact.unclaimed] + 0.45 * np.array([220, 160, 60])
    return np.clip(overlay, 0, 255).astype(np.uint8)


@app.route("/api/contact/<name>/move", methods=["POST"])
def api_contact_move(name):
    """Hand over the new contact region after the product is moved."""
    sess = _get_sample_session(name)
    body = request.get_json(force=True)
    try:
        dx = float(body.get("dx", 0.0))
        dy = float(body.get("dy", 0.0))
        contact = _contact(sess)
        tracker = GroundTracker(contact)
        region = tracker.new_contact_region(dx, dy)
        moved = tracker.translate(dx, dy)
        # Re-composite on the requested plate using the moved shadow fields.
        plate_u8 = _plate_from_request(sess, body.get("plate"))
        B2 = srgb_to_linear(_fit(plate_u8, contact.shape[:2]).astype(np.float64) / 255.0)
        comp = composite_contact(B2, moved["t"], moved["q"], alpha=moved["alpha"],
                                 P_lin=sess["dec"].P_lin,
                                 shadow_mask=moved["claimed"])
    except ContactShadowError as exc:
        return jsonify(error=str(exc), code=exc.code), 409
    region["clash_px"] = moved["clash"]
    return jsonify(region=region, image=_b64(_u8_lin(comp)))


@app.route("/api/contact/<name>/check", methods=["POST"])
def api_contact_check(name):
    """Edit the shadow region, composite, then verify contact and retention."""
    sess = _get_sample_session(name)
    body = request.get_json(force=True)
    contact = _contact(sess)
    dec = sess["dec"]
    # Optional edit: polygon restricting the claimed region.
    claimed = contact.claimed
    if body.get("edit_polygon"):
        pts = np.asarray(body["edit_polygon"], np.int32)
        m = np.zeros(contact.shape, np.uint8)
        if len(pts) >= 3:
            cv2.fillPoly(m, [pts], 1)
        claimed = contact.claimed & m.astype(bool)
    if not claimed.any():
        return jsonify(error="edit removed the whole contact region; nothing left to "
                             "verify against the object",
                       code="empty_contact_region"), 409
    plate_u8 = _plate_from_request(sess, body.get("plate"))
    B2 = srgb_to_linear(_fit(plate_u8, contact.shape[:2]).astype(np.float64) / 255.0)
    try:
        comp = composite_contact(B2, dec.t, dec.q, alpha=dec.alpha, P_lin=dec.P_lin,
                                 shadow_mask=claimed)
        edited = contact.__class__(contact.t, contact.q, contact.alpha, claimed,
                                   contact.unclaimed & ~claimed, contact.region,
                                   contact.footprint, contact.band, contact.thresholds)
        check = verify_contact(comp, B2, edited,
                               source_fg_premult=dec.alpha[..., None] * dec.P_lin,
                               coverage_min=float(body.get("coverage_min", 0.9)),
                               gap_max_px=float(body.get("gap_max_px", 4.0)),
                               retention_min=float(body.get("retention_min", 0.98)),
                               leak_limit=float(body.get("leak_limit", 0.35)))
    except ContactShadowError as exc:
        return jsonify(error=str(exc), code=exc.code), 409
    return jsonify(check=check.as_dict(), image=_b64(_u8_lin(comp)))


@app.route("/api/contact/<name>/export", methods=["POST"])
def api_contact_export(name):
    """Export the composite on two different floor textures.

    Returns both images plus a tile-difference flag: if the shadow carried the
    original floor pattern with it, its high-frequency content would agree
    across the two exports, which the response reports.
    """
    sess = _get_sample_session(name)
    body = request.get_json(silent=True) or {}
    contact = _contact(sess)
    dec = sess["dec"]
    h, w = contact.shape
    rng = np.random.default_rng(int(body.get("seed", 11)))
    beta_a = make_plate((h, w), rng, kind="paper",
                        base_rgb=(0.72, 0.66, 0.56), contrast=0.30)
    beta_b = make_plate((h, w), np.random.default_rng(int(body.get("seed", 11)) + 77),
                        kind="wood", base_rgb=(0.42, 0.32, 0.24), contrast=0.34)
    out = {}
    shadows = {}
    floors = {}
    for label, beta in (("floor_a", beta_a), ("floor_b", beta_b)):
        comp = composite_contact(beta, dec.t, dec.q, alpha=dec.alpha, P_lin=dec.P_lin,
                                 shadow_mask=contact.claimed)
        shadow_only = beta * (dec.t[..., None] + dec.q)
        shadows[label] = shadow_only
        floors[label] = beta
        out[label] = _b64(_u8_lin(comp))
    # Tile-carry diagnostic: inside the claimed shadow, a physically re-lit
    # shadow keeps *each* floor's own high-frequency texture, so the shadow's
    # high frequencies correlate with the floor it sits on.  Carrying the old
    # tile would give a pattern uncorrelated with both new floors (near 0).
    from scipy import ndimage as _ndi
    from .color import luminance as _lum

    def _hf(img):
        l = _lum(img)
        return (l - _ndi.gaussian_filter(l, 2.0))[contact.claimed]

    def _corr(x, y):
        x = x - x.mean()
        y = y - y.mean()
        return float((x * y).mean() / max(float(x.std() * y.std()), 1e-12))

    floor_corr = {label: _corr(_hf(shadows[label]), _hf(floors[label]))
                  for label in shadows}
    carry = float(min(floor_corr.values()))
    return jsonify(images=out,
                   floor_texture_correlation=floor_corr,
                   texture_carry_ratio=carry,
                   # High correlation on both floors => the original tile was
                   # NOT carried along; low correlation flags a copied floor.
                   floor_patterns_differ=bool(carry > 0.2),
                   claimed_px=int(contact.claimed.sum()),
                   unclaimed_px=int(contact.unclaimed.sum()))


@app.route("/api/verify/<name>", methods=["POST"])
def api_verify(name):
    """Apply the extracted layers to the sample's *second* plate and score it.

    The target is the independently rendered/captured ``product2.png``, not a
    re-projection of the first capture.  That is the only honest test that the
    layers are physically transferable.
    """
    d = os.path.join(_SAMPLES_DIR, name)
    e2 = os.path.join(d, "empty2.png")
    p2 = os.path.join(d, "product2.png")
    if not (os.path.isfile(e2) and os.path.isfile(p2)):
        return jsonify(error="sample has no second plate"), 404
    body = request.get_json(silent=True) or {}
    sess = _SESSIONS.get(body.get("session"))
    if sess is None or "dec" not in sess:
        return jsonify(error="extract this sample first"), 400
    session_diff = sess.get("name") not in (None, name)
    if session_diff:
        return jsonify(error="session does not belong to this sample"), 400
    plate2_u8 = cv2.cvtColor(cv2.imread(e2), cv2.COLOR_BGR2RGB)
    target_u8 = cv2.cvtColor(cv2.imread(p2), cv2.COLOR_BGR2RGB)
    pred = _contact_composite(sess, plate2_u8, 1.0, 1.0, claimed_only=False)
    from .evaluate import _psnr, _ssim_global
    target_lin = _srgb_lin(_fit(target_u8, pred.shape[:2]))
    pred_lin = _srgb_lin(pred)
    ps = float(_psnr(pred_lin, target_lin))
    ss = float(_ssim_global(pred_lin, target_lin))
    return jsonify(image=_b64(pred), psnr=ps, ssim=ss,
                   width=pred.shape[1], height=pred.shape[0])


# --------------------------------------------------------------------------- #
# Sample library (lets the UI be exercised without uploading)
# --------------------------------------------------------------------------- #

_SAMPLES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "samples")


@app.route("/api/samples")
def api_samples():
    out = []
    if os.path.isdir(_SAMPLES_DIR):
        for name in sorted(os.listdir(_SAMPLES_DIR)):
            d = os.path.join(_SAMPLES_DIR, name)
            if os.path.isfile(os.path.join(d, "empty.png")):
                out.append(name)
    return jsonify(samples=out)


@app.route("/api/sample/<name>")
def api_sample(name):
    d = os.path.join(_SAMPLES_DIR, name)
    if not os.path.isfile(os.path.join(d, "empty.png")):
        return jsonify(error="not found"), 404
    with _LOCK:
        _SESSIONS["__sample__" + name] = dict(dir=d)
    b = {}
    for key, fn in (("empty", "empty.png"), ("product", "product.png"),
                    ("product2", "product2.png")):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            b[key] = _b64(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB))
    return jsonify(images=b)


@app.route("/api/sample_extract/<name>", methods=["POST"])
def api_sample_extract(name):
    d = os.path.join(_SAMPLES_DIR, name)
    p = os.path.join(d, "product.png")
    e = os.path.join(d, "empty.png")
    if not (os.path.isfile(p) and os.path.isfile(e)):
        return jsonify(error="not found"), 404
    empty = cv2.cvtColor(cv2.imread(e), cv2.COLOR_BGR2RGB)
    product = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
    body = request.get_json(silent=True) or {}
    support = None
    if body.get("support"):
        pts = np.asarray(body["support"], np.int32)
        if len(pts) >= 3:
            m = np.zeros(empty.shape[:2], np.uint8)
            cv2.fillPoly(m, [pts], 1)
            support = m.astype(bool)
    method = body.get("method", "joint")
    dec = extract(empty, product, support=support, method=method)
    sid = uuid.uuid4().hex
    with _LOCK:
        if len(_SESSIONS) >= _MAX_SESSIONS:
            _SESSIONS.pop(next(iter(_SESSIONS)))
        _SESSIONS[sid] = dict(dec=dec, empty=empty, product=product, name=name,
                              contact=None)
    fg_u8, occl_u8, q_u8, u_u8 = _panels(_SESSIONS[sid])
    comp = _composite(_SESSIONS[sid], empty, 1.0, 1.0)
    return jsonify(session=sid, method=method, diag=dec.diag.as_dict(), info=dec.info,
                   panels={"foreground": _b64(fg_u8), "occlusion": _b64(occl_u8),
                           "bounce": _b64(q_u8), "unexplained": _b64(u_u8),
                           "original": _b64(empty), "capture": _b64(product),
                           "reconstruct": _b64(comp)})


@app.route("/api/contact/<name>/asset", methods=["POST"])
def api_contact_asset(name):
    """Convert the sample's current extraction into a v2 asset bundle."""
    from .migrate import migration_for_decomposition
    sess = _get_sample_session(name)
    body = request.get_json(silent=True) or {}
    out_dir = body.get("out_dir") or os.path.join(_ASSETS_DIR, name)
    d = os.path.join(_SAMPLES_DIR, name)
    sources = {}
    for role, fn in (("empty", "empty.png"), ("product", "product.png")):
        p = os.path.join(d, fn)
        if os.path.isfile(p):
            sources[role] = p
    try:
        loaded, contact = migration_for_decomposition(
            sess["dec"], out_dir, name=name, source_files=sources,
            contact_kwargs=dict(occlusion_thresh=float(body.get("occlusion", 0.04)),
                                texture_leak_limit=float(body.get("leak_limit", 0.35))))
    except Exception as exc:  # asset integrity errors -> explicit interface failure
        return jsonify(error=str(exc), code="asset_write_failed"), 409
    return jsonify(bundle=out_dir, manifest=loaded["manifest"]["format_version"],
                   contact=contact.summary(),
                   statuses={e["name"]: e["status"]
                             for e in loaded["manifest"]["layers"]})


_ASSETS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "assets")


def main(host="127.0.0.1", port=7860, debug=False):
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
