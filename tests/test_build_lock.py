"""Build-lock and sample-tree migration tests (clean-directory reproducible)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.asset import AssetFormatError, load_asset, sha256_file
from shadow_extract.migrate import (LOCK_NAME, migrate_gt_npz,
                                    migrate_sample_tree, verify_build_lock,
                                    write_build_lock)


def _make_sample(root, name, H=64, W=80):
    sdir = root / name
    sdir.mkdir(parents=True)
    alpha = np.zeros((H, W))
    alpha[20:42, 30:52] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - 41) - 11, 0) + np.maximum(np.abs(yy - 31) - 11, 0)
    t = 1.0 - np.clip(0.5 * np.exp(-d / 5.0), 0, 1)
    q = np.zeros((H, W, 3))
    P = np.full((H, W, 3), 0.7)
    np.savez(str(sdir / "gt.npz"), t=t, q=q, alpha=alpha, P=P,
             B1=np.full((H, W, 3), 0.5), I1=np.full((H, W, 3), 0.4),
             beta1=np.full((H, W, 3), 0.5))
    cv2.imwrite(str(sdir / "empty.png"), np.full((H, W, 3), 128, np.uint8))
    cv2.imwrite(str(sdir / "product.png"), np.full((H, W, 3), 110, np.uint8))
    (sdir / "meta.json").write_text(json.dumps({"name": name, "cam": {
        "eye": [0, 2, 3], "target": [0, 0, 0], "up": [0, 1, 0],
        "f": 560.0, "width": W, "height": H}}))
    return sdir


def test_gt_npz_migration_carries_contract_and_revision(tmp_path):
    sdir = _make_sample(tmp_path, "sample_a")
    bundle = tmp_path / "assets" / "sample_a"
    loaded, contact = migrate_gt_npz(
        str(sdir / "gt.npz"), str(bundle), name="sample_a",
        meta=json.loads((sdir / "meta.json").read_text()),
        source_files={"empty": str(sdir / "empty.png"),
                      "product": str(sdir / "product.png")})
    m = loaded["manifest"]
    assert m["format_version"] == "2.0"
    assert m["source"]["git_revision"] or m["source"]["revision_note"]
    assert loaded["contact"]["version"] == "contact/2"
    assert "foreground_premult" in loaded["layers"]
    # The camera plane is preserved from meta.
    assert loaded["plane"].camera["f"] == 560.0


def test_build_lock_detects_drift(tmp_path):
    sdir = _make_sample(tmp_path, "sample_a")
    assets = tmp_path / "assets"
    bundle = assets / "sample_a"
    migrate_gt_npz(str(sdir / "gt.npz"), str(bundle), name="sample_a",
                   source_files={"empty": str(sdir / "empty.png")})
    lock_path, _ = write_build_lock(str(assets),
                                    [{"name": "sample_a", "bundle": str(bundle)}])
    ok, mm = verify_build_lock(str(assets))
    assert ok and mm == []
    # Tamper with a layer after locking.
    np.save(str(bundle / "layer_t.npy"), np.zeros_like(np.load(str(bundle / "layer_t.npy"))))
    ok, mm = verify_build_lock(str(assets))
    assert not ok and mm and "drift" in mm[0]["reason"]


def test_sample_tree_migration_runs_in_clean_dir(tmp_path):
    samples = tmp_path / "samples"
    _make_sample(samples, "one")
    _make_sample(samples, "two")
    # Legacy file exercising the underdetermined path.
    legacy = samples / "one" / "old_merged.npz"
    with np.load(str(samples / "one" / "gt.npz")) as z:
        np.savez(str(legacy), luma=1.0 - z["t"], alpha=z["alpha"])
    assets = tmp_path / "assets"
    result = migrate_sample_tree(
        str(samples), str(assets),
        legacy_files={"one": [str(legacy)]})
    assert result["n_assets"] == 3
    assert os.path.isfile(result["lock"])
    ok, _ = verify_build_lock(str(assets))
    assert ok
    # Every bundle independently re-verifies and declares its migration path.
    for name in ("one", "two"):
        loaded = load_asset(str(assets / name))
        paths = loaded["manifest"]["extra"]["migration"]["paths"]
        assert paths == ["reversible:full_decomposition"]
    legacy_asset = load_asset(str(assets / "one_legacy_old_merged"))
    assert legacy_asset["manifest"]["extra"]["migration"]["paths"] == \
        ["underdetermined:merged_luma"]


def test_lock_covers_source_hashes(tmp_path):
    sdir = _make_sample(tmp_path, "sample_a")
    assets = tmp_path / "assets"
    bundle = assets / "sample_a"
    migrate_gt_npz(str(sdir / "gt.npz"), str(bundle), name="sample_a",
                   source_files={"empty": str(sdir / "empty.png")})
    write_build_lock(str(assets), [{"name": "sample_a", "bundle": str(bundle)}])
    lock = json.loads((assets / LOCK_NAME).read_text())
    entry = lock["assets"][0]
    src = next(f for f in entry["source_files"] if f["role"] == "empty")
    assert src["sha256"] == sha256_file(str(sdir / "empty.png"))
    # Premultiplication flags are locked per layer.
    premult = {lf["file"]: lf["premultiplied"] for lf in entry["layer_files"]}
    assert premult["layer_foreground_premult.npy"] is True
    assert premult["layer_t.npy"] is False


def test_missing_lock_is_an_error(tmp_path):
    with pytest.raises(AssetFormatError, match="lock"):
        verify_build_lock(str(tmp_path))
