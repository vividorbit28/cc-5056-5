"""Tests for v1 -> v2 layer migration: reversible vs underdetermined paths.

Every legacy file is generated inside the test from explicit arrays, so the
suite needs no hand-edited data and runs in a clean directory.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.asset import (STATUS_NEEDS_CLAIM,
                                  STATUS_PENDING, load_asset)
from shadow_extract.legacy import (LegacyFormatError, inspect_legacy,
                                   migrate_legacy, reopen_underdetermined,
                                   save_underdetermined_edit)


def _arrays(H=60, W=80):
    rng = np.random.default_rng(5)
    alpha = np.zeros((H, W))
    alpha[20:40, 30:50] = 1.0
    t = np.ones((H, W))
    t[15:45, 25:55] = 0.6
    q = np.zeros((H, W, 3))
    q[15:45, 25:55] = (0.02, 0.01, 0.005)
    P = np.full((H, W, 3), 0.7)
    plate = rng.uniform(0.2, 0.8, (H, W, 3))
    luma = t.copy()  # legacy single channel stored as the linear darkening scale
    return alpha, t, q, P, plate, luma


@pytest.fixture()
def legacy_tree(tmp_path):
    alpha, t, q, P, plate, luma = _arrays()
    src = tmp_path / "empty.png"
    cv2.imwrite(str(src), np.full(t.shape + (3,), 128, np.uint8))
    full = tmp_path / "full.npz"
    np.savez(str(full), t=t, q=q, alpha=alpha, P=P)
    baked = tmp_path / "baked.npz"
    np.savez(str(baked), scale=t, plate_lin=plate, alpha=alpha, P=P)
    baked_noplate = tmp_path / "baked_noplate.npz"
    np.savez(str(baked_noplate), scale=t, alpha=alpha, P=P)
    merged = tmp_path / "merged.npz"
    np.savez(str(merged), luma=luma, alpha=alpha, P=P)
    merged_plate = tmp_path / "merged_plate.npz"
    np.savez(str(merged_plate), luma=luma, plate_lin=plate, alpha=alpha, P=P)
    return tmp_path, src


def test_inspect_detects_three_kinds(legacy_tree):
    tmp, _ = legacy_tree
    assert inspect_legacy(str(tmp / "full.npz"))["kind"] == "full_decomposition"
    assert inspect_legacy(str(tmp / "baked.npz"))["kind"] == "baked_plane"
    assert inspect_legacy(str(tmp / "merged.npz"))["kind"] == "merged_luma"
    with pytest.raises(LegacyFormatError):
        inspect_legacy(str(tmp / "empty.png"))


def test_full_decomposition_migrates_losslessly(legacy_tree):
    tmp, src = legacy_tree
    alpha, t, q, P, _, _ = _arrays()
    loaded, report = migrate_legacy(str(tmp / "full.npz"), str(tmp / "b_full"),
                                    source_files={"empty": str(src)})
    assert report["paths"] == ["reversible:full_decomposition"]
    assert np.allclose(loaded["layers"]["t"], t)
    assert np.allclose(loaded["layers"]["q"], q)
    assert np.allclose(loaded["layers"]["alpha"], alpha)
    assert load_asset(str(tmp / "b_full")) is not None  # reopens


def test_baked_plate_reversible_keeps_original_and_splits(legacy_tree):
    tmp, src = legacy_tree
    _, t, _, _, _, _ = _arrays()
    loaded, report = migrate_legacy(str(tmp / "baked.npz"), str(tmp / "b_baked"),
                                    source_files={"empty": str(src)})
    assert "reversible" in report["paths"][0]
    # Achromatic split lands in t; q stays zero because bounce is unobservable.
    assert np.allclose(loaded["layers"]["t"], t)
    assert np.allclose(loaded["layers"]["q"], 0.0)
    # Original baked map preserved byte-for-byte in the legacy block.
    legacy_arr = np.load(str(tmp / "b_baked" / "legacy" / "v1_scale.npy"))
    assert np.allclose(legacy_arr, t)
    # A legacy colour/darkening never occupies t via a colour guess: t is
    # explicitly the linear achromatic transmission.
    assert loaded["manifest"]["legacy"]["evaluable"] is True


def test_baked_plane_without_plate_is_underdetermined(legacy_tree):
    tmp, src = legacy_tree
    loaded, report = migrate_legacy(str(tmp / "baked_noplate.npz"),
                                    str(tmp / "b_baked2"),
                                    source_files={"empty": str(src)})
    assert report["paths"] == ["underdetermined:baked_plane"]
    statuses = {e["name"]: e["status"] for e in loaded["manifest"]["layers"]}
    assert statuses["t"] == STATUS_NEEDS_CLAIM
    assert statuses["legacy_shadow_luma"] == STATUS_NEEDS_CLAIM
    # t must not have been guessed: fully open until the author claims.
    assert np.allclose(loaded["layers"]["t"], 1.0)
    assert any("plate" in w for w in report["warnings"])


def test_merged_luma_default_is_underdetermined_and_preserved(legacy_tree):
    tmp, src = legacy_tree
    _, t, _, _, _, luma = _arrays()
    loaded, report = migrate_legacy(str(tmp / "merged.npz"), str(tmp / "b_merged"),
                                    source_files={"empty": str(src)})
    assert report["paths"] == ["underdetermined:merged_luma"]
    statuses = {e["name"]: e["status"] for e in loaded["manifest"]["layers"]}
    assert statuses["t"] == STATUS_PENDING
    assert statuses["legacy_shadow_luma"] == STATUS_NEEDS_CLAIM
    # The single channel is preserved for continued editing.
    assert np.allclose(loaded["layers"]["legacy_shadow_luma"], luma)


def test_merged_luma_reversible_achromatic_path(legacy_tree):
    tmp, src = legacy_tree
    _, t, _, _, _, _ = _arrays()
    loaded, report = migrate_legacy(str(tmp / "merged_plate.npz"),
                                    str(tmp / "b_merged_rev"),
                                    source_files={"empty": str(src)},
                                    prefer="reversible")
    assert report["paths"] == ["reversible:merged_luma_achromatic"]
    assert np.allclose(loaded["layers"]["t"], t)
    assert np.allclose(loaded["layers"]["q"], 0.0)
    assert loaded["manifest"]["legacy"]["evaluable"] is True


def test_underdetermined_layer_reopens_edits_and_relocks(legacy_tree):
    tmp, src = legacy_tree
    bundle = str(tmp / "b_edit")
    migrate_legacy(str(tmp / "merged.npz"), bundle,
                   source_files={"empty": str(src)})
    loaded, luma = reopen_underdetermined(bundle)
    edited = np.clip(luma * 0.8, 0.0, 1.0)
    save_underdetermined_edit(bundle, edited)
    # Reopened after the edit: the asset still verifies and keeps the edit.
    again = load_asset(bundle)
    assert np.allclose(again["layers"]["legacy_shadow_luma"], edited)
    assert {e["name"]: e["status"] for e in again["manifest"]["layers"]}[
        "legacy_shadow_luma"] == STATUS_NEEDS_CLAIM
    # Hash in the manifest matches the rewritten file (no stale lock).
    entry = next(e for e in again["manifest"]["layers"]
                 if e["name"] == "legacy_shadow_luma")
    from shadow_extract.asset import sha256_file
    assert entry["sha256"] == sha256_file(str(tmp / "b_edit" / entry["file"]))


def test_legacy_shadow_color_never_becomes_transmission(legacy_tree):
    tmp, src = legacy_tree
    # A chromatic legacy map must not be split into per-channel t.
    chroma = np.zeros(_arrays()[1].shape + (3,))
    chroma[..., 0] = 0.4
    path = tmp / "chroma.npz"
    np.savez(str(path), luma=chroma, alpha=np.zeros(chroma.shape[:2]))
    loaded, _ = migrate_legacy(str(path), str(tmp / "b_chroma"),
                               source_files={"empty": str(src)})
    # Underdetermined: t stays open; the chroma is held in the luma layer only.
    assert np.allclose(loaded["layers"]["t"], 1.0)
    assert loaded["layers"]["legacy_shadow_luma"].ndim == 2


def test_json_legacy_container(legacy_tree):
    tmp, src = legacy_tree
    _, t, q, alpha, _, _ = _arrays()
    doc = {"layers": {"t": t.tolist(), "q": q.tolist(), "alpha": alpha.tolist()}}
    p = tmp / "doc.json"
    p.write_text(json.dumps(doc))
    loaded, report = migrate_legacy(str(p), str(tmp / "b_json"),
                                    source_files={"empty": str(src)})
    assert report["kind"] == "full_decomposition"
    assert np.allclose(loaded["layers"]["t"], t, atol=1e-9)
