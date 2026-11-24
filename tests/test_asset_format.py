"""Tests for the v2 shadow/foreground asset format.

The fixtures are synthesised in a clean temporary directory from explicit
arrays -- no hand-edited data is required and the suite runs without the
generated sample tree.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.asset import (AssetFormatError, LayerSpec, Plane,
                                  is_premultiplied, layer_statuses, load_asset,
                                  premultiply, save_asset, sha256_file,
                                  unpremultiply)
from shadow_extract.contact import derive_contact


def _toy_decomposition(H=70, W=90, textured_shadow=False):
    rng = np.random.default_rng(3)
    t = np.ones((H, W))
    q = np.zeros((H, W, 3))
    alpha = np.zeros((H, W))
    alpha[25:45, 35:55] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - 45) - 10, 0) + np.maximum(np.abs(yy - 35) - 10, 0)
    occ = np.clip(0.55 * np.exp(-d / 6.0), 0.0, 1.0)
    if textured_shadow:
        occ += 0.25 * (rng.random((H, W)) - 0.5)
        occ = np.clip(occ, 0.0, 1.0)
    t = 1.0 - occ
    P = np.full((H, W, 3), 0.7)
    return t, q, alpha, P


class _Dec:
    pass


def _dec(t, q, alpha, P):
    d = _Dec()
    d.t, d.q, d.alpha, d.P_lin = t, q, alpha, P
    d.method = "joint"
    d.info = {}
    return d


@pytest.fixture()
def workspace(tmp_path):
    t, q, alpha, P = _toy_decomposition()
    src = tmp_path / "empty.png"
    cv2.imwrite(str(src), np.full(t.shape + (3,), 128, np.uint8))
    return tmp_path, t, q, alpha, P, str(src)


def _save(tmp_path, t, q, alpha, P, src, statuses=None):
    contact = derive_contact(_dec(t, q, alpha, P))
    fg = premultiply(P, alpha)
    layers = [
        LayerSpec("t", t, premultiplied=False, role="t"),
        LayerSpec("q", q, premultiplied=False, role="q"),
        LayerSpec("alpha", alpha, premultiplied=False, role="alpha"),
        LayerSpec("foreground_premult", fg, premultiplied=True),
        LayerSpec("contact_unclaimed", contact.unclaimed.astype(np.uint8),
                  premultiplied=False),
    ]
    bundle = tmp_path / "asset"
    loaded = save_asset(str(bundle), name="toy", layers=layers, plane=Plane(),
                        source_files={"empty": src}, contact=contact.to_json())
    return loaded, contact


def test_asset_declares_premult_plane_and_revision(workspace):
    tmp_path, t, q, alpha, P, src = workspace
    loaded, _ = _save(tmp_path, t, q, alpha, P, src)
    m = loaded["manifest"]
    assert m["format_version"] == "2.0"
    assert m["plane"]["frame"] == "image_pixel"
    assert m["plane"]["origin_px"] == [0.0, 0.0]
    # Revision is declared explicitly; in a git tree it is a commit hash.
    assert "git_revision" in m["source"]
    assert m["source"]["files"][0]["sha256"] == sha256_file(src)
    assert is_premultiplied(loaded, "foreground_premult") is True
    assert is_premultiplied(loaded, "t") is False
    # Premultiplication round-trips.
    fg = loaded["layers"]["foreground_premult"]
    back = unpremultiply(fg, alpha)
    assert np.allclose(back[alpha > 0.5], P[alpha > 0.5])


def test_asset_hash_tampering_is_rejected(workspace):
    tmp_path, t, q, alpha, P, src = workspace
    loaded, _ = _save(tmp_path, t, q, alpha, P, src)
    bad = np.zeros_like(t)
    np.save(str(tmp_path / "asset" / "layer_t.npy"), bad)
    with pytest.raises(AssetFormatError, match="hash mismatch"):
        load_asset(str(tmp_path / "asset"))


def test_missing_premult_flag_rejected(tmp_path, workspace):
    _, t, q, alpha, P, src = workspace
    bundle = tmp_path / "a"
    layers = [LayerSpec("t", t, premultiplied=False)]
    save_asset(str(bundle), name="a", layers=layers, plane=Plane(),
               source_files={"empty": src})
    manifest = bundle / "shadow_asset.json"
    doc = json.loads(manifest.read_text())
    del doc["layers"][0]["premultiplied"]
    manifest.write_text(json.dumps(doc))
    with pytest.raises(AssetFormatError, match="premultiplied"):
        load_asset(str(bundle))


def test_missing_plane_block_rejected(tmp_path, workspace):
    _, t, q, alpha, P, src = workspace
    bundle = tmp_path / "a"
    save_asset(str(bundle), name="a", layers=[LayerSpec("t", t, premultiplied=False)],
               plane=Plane(), source_files={"empty": src})
    manifest = bundle / "shadow_asset.json"
    doc = json.loads(manifest.read_text())
    del doc["plane"]
    manifest.write_text(json.dumps(doc))
    with pytest.raises(AssetFormatError, match="plane"):
        load_asset(str(bundle))


def test_null_revision_requires_note(tmp_path, workspace):
    _, t, _, _, _, src = workspace
    bundle = tmp_path / "a"
    save_asset(str(bundle), name="a", layers=[LayerSpec("t", t, premultiplied=False)],
               plane=Plane(), source_files={"empty": src})
    manifest = bundle / "shadow_asset.json"
    doc = json.loads(manifest.read_text())
    doc["source"]["git_revision"] = None
    doc["source"]["revision_note"] = None
    manifest.write_text(json.dumps(doc))
    with pytest.raises(AssetFormatError, match="revision"):
        load_asset(str(bundle))


def test_layer_statuses_include_pending_and_claim(tmp_path, workspace):
    _, t, q, alpha, P, src = workspace
    from shadow_extract.asset import STATUS_NEEDS_CLAIM, STATUS_PENDING
    bundle = tmp_path / "a"
    save_asset(str(bundle), name="a", plane=Plane(), source_files={"empty": src},
               layers=[LayerSpec("t", t, False, status=STATUS_PENDING),
                       LayerSpec("legacy", t, False, status=STATUS_NEEDS_CLAIM)])
    loaded = load_asset(str(bundle))
    assert layer_statuses(loaded) == {"t": STATUS_PENDING,
                                      "legacy": STATUS_NEEDS_CLAIM}
