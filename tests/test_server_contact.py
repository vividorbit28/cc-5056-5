"""Server integration tests for the contact-shadow routes and composition canvas.

Uses the Flask test client against a small generated sample written into the
samples directory at runtime, so no manual data preparation is needed.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
import pytest

from shadow_extract.server import app
from shadow_extract.synth import (Light, build_product, make_plate, render)
from shadow_extract.geometry import Camera
from shadow_extract.visualize import to_u8_srgb


@pytest.fixture()
def client(tmp_path, monkeypatch):
    samples = tmp_path / "samples"
    samples.mkdir()
    name = "server_case"
    sdir = samples / name
    sdir.mkdir()
    cam = Camera(width=200, height=160)
    rng = np.random.default_rng(0)
    scene, albedo = build_product("ball", scale=0.55)
    light = Light(key_dir=(0.05, 0.99, 0.05), angular_radius=0.05)
    beta = make_plate((160, 200), rng, "flat", (0.6, 0.58, 0.52), 0.10)
    out = render(cam, scene, albedo, beta, light, ss=1, rng=rng)
    cv2.imwrite(str(sdir / "empty.png"),
                cv2.cvtColor(to_u8_srgb(out["B"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sdir / "product.png"),
                cv2.cvtColor(to_u8_srgb(out["I"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sdir / "empty2.png"),
                cv2.cvtColor(to_u8_srgb(out["B"]), cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(sdir / "product2.png"),
                cv2.cvtColor(to_u8_srgb(out["I"]), cv2.COLOR_RGB2BGR))
    monkeypatch.setattr("shadow_extract.server._SAMPLES_DIR", str(samples))
    monkeypatch.setattr("shadow_extract.server._ASSETS_DIR", str(tmp_path / "assets"))
    app.config.update(TESTING=True)
    return app.test_client(), name, tmp_path


def test_contact_derive_route(client):
    c, name, _ = client
    r = c.post(f"/api/contact/{name}", json={"occlusion": 0.04})
    assert r.status_code == 200
    d = r.get_json()
    assert d["contact"]["version"] == "contact/2"
    assert d["contact"]["n_claimed"] > 0
    assert "region" in d


def test_contact_move_hands_over_region(client):
    c, name, _ = client
    c.post(f"/api/contact/{name}", json={})
    r = c.post(f"/api/contact/{name}/move", json={"dx": 10, "dy": 0})
    assert r.status_code == 200
    region = r.get_json()["region"]
    assert region["n_pixels"] > 0
    assert "bbox_plane" in region


def test_contact_check_route_reports_metrics(client):
    c, name, _ = client
    r = c.post(f"/api/contact/{name}/check",
               json={"plate": {"kind": "solid", "color": [180, 170, 160]},
                     "coverage_min": 0.35, "gap_max_px": 12.0})
    assert r.status_code == 200
    check = r.get_json()["check"]
    assert {"contact_coverage", "gap_px", "shadow_contribution",
            "foreground_retention"} <= set(check)
    assert check["shadow_contribution"] > 0.0
    assert check["foreground_retention"] >= 0.98


def test_contact_check_interface_error_on_bad_edit(client):
    c, name, _ = client
    # A polygon wholly outside the image leaves no claimed region; the
    # interface must explain instead of silently compositing an empty shadow.
    r = c.post(f"/api/contact/{name}/check",
               json={"edit_polygon": [[900, 900], [905, 900], [900, 905]]})
    d = r.get_json()
    assert r.status_code == 409
    assert "code" in d


def test_export_two_floors_flags_texture_carry(client):
    c, name, _ = client
    r = c.post(f"/api/contact/{name}/export", json={"seed": 11})
    assert r.status_code == 200
    d = r.get_json()
    assert set(d["images"]) == {"floor_a", "floor_b"}
    assert set(d["floor_texture_correlation"]) == {"floor_a", "floor_b"}
    assert isinstance(d["floor_patterns_differ"], bool)
    assert d["claimed_px"] > 0


def test_compose_canvas_uses_contact_path(client):
    c, name, tmp = client
    # Extract first to create a normal session.
    with open(str(tmp / "samples" / name / "empty.png"), "rb") as fe, \
         open(str(tmp / "samples" / name / "product.png"), "rb") as fp:
        r = c.post("/api/extract",
                   data={"empty": (fe, "empty.png"), "product": (fp, "product.png")},
                   content_type="multipart/form-data")
    sid = r.get_json()["session"]
    r2 = c.post("/api/compose", json={"session": sid,
                                      "plate": {"kind": "solid", "color": [200, 200, 200]}})
    assert r2.status_code == 200
    assert "image" in r2.get_json()


def test_asset_conversion_route(client):
    c, name, tmp = client
    r = c.post(f"/api/contact/{name}/asset", json={})
    assert r.status_code == 200
    d = r.get_json()
    assert d["manifest"] == "2.0"
    bundle = tmp / "assets" / name
    assert (bundle / "shadow_asset.json").is_file()
    assert (bundle / "contact.json").is_file()


def test_unknown_sample_returns_404(client):
    c, _, _ = client
    r = c.post("/api/contact/nope", json={})
    assert r.status_code == 404


def test_index_documents_graphical_entries(client):
    c, _, _ = client
    html = c.get("/").get_data(as_text=True)
    assert "Derive contact shadow" in html
    assert "Export two floor textures" in html
    assert "Convert to v2 asset" in html
