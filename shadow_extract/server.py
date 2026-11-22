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
from .extract import Params, extract
from .plates import make_plate
from .visualize import (checkerboard, composite_srgb, png_bytes, scalar_map,
                        to_u8_srgb)

app = Flask(__name__)

_SESSIONS = {}
_LOCK = threading.Lock()
_MAX_SESSIONS = 32


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
        _SESSIONS[sid] = dict(dec=dec, empty=empty, product=product)

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
    plate = body.get("plate") or {}
    if plate.get("kind") == "solid":
        col = np.array(plate.get("color", [200, 200, 200]), float) / 255.0
        img = np.empty(sess["dec"].B.shape, np.float64)
        img[...] = col
        plate_u8 = to_u8_srgb(img)
    elif plate.get("kind") == "generated":
        rng = np.random.default_rng(int(plate.get("seed", 0)))
        beta = make_plate(sess["dec"].B.shape[:2], rng, kind=plate.get("style", "paper"),
                          base_rgb=tuple(plate.get("base", (0.62, 0.58, 0.52))),
                          contrast=float(plate.get("contrast", 0.22)))
        plate_u8 = to_u8_srgb(beta)
    else:  # original or uploaded
        plate_u8 = sess["empty"]
    out = _composite(sess, plate_u8, opacity, bounce, use_product)
    return jsonify(image=_b64(out))


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
    pred = _composite(sess, plate2_u8, 1.0, 1.0)
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
        _SESSIONS[sid] = dict(dec=dec, empty=empty, product=product, name=name)
    fg_u8, occl_u8, q_u8, u_u8 = _panels(_SESSIONS[sid])
    comp = _composite(_SESSIONS[sid], empty, 1.0, 1.0)
    return jsonify(session=sid, method=method, diag=dec.diag.as_dict(), info=dec.info,
                   panels={"foreground": _b64(fg_u8), "occlusion": _b64(occl_u8),
                           "bounce": _b64(q_u8), "unexplained": _b64(u_u8),
                           "original": _b64(empty), "capture": _b64(product),
                           "reconstruct": _b64(comp)})


def main(host="127.0.0.1", port=7860, debug=False):
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
