"""Convert current shadow/foreground intermediates into v2 assets and lock the build.

Two entry points:

* :func:`migration_for_decomposition` -- a fresh in-memory extraction (the
  normal path) becomes a full v2 bundle with a derived contact shadow;
* :func:`migrate_sample_tree` -- walk a samples directory, migrate every
  current ``gt.npz`` intermediate and any legacy v1 layer files, and write a
  ``build-lock.json`` recording every input/output hash and the git revision.

The lock is the evidence that the conversion is reproducible in a clean
directory: re-running the migration recomputes the hashes and fails loudly on
any drift instead of silently accepting changed data.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import __version__ as TOOL_VERSION
from .asset import (AssetFormatError, LayerSpec, Plane, STATUS_CONFIRMED,
                    git_revision, load_asset, save_asset, sha256_file)
from .contact import derive_contact
from .legacy import migrate_legacy

LOCK_NAME = "build-lock.json"


def _camera_plane(meta):
    if not meta or "cam" not in meta:
        return Plane()
    cam = meta["cam"]
    return Plane(origin_px=(0.0, 0.0), scale_px=(1.0, 1.0),
                 frame="image_pixel", camera=cam)


def migration_for_decomposition(dec, bundle_dir, *, name, source_files,
                                repo_root=None, notes=None, plane=None,
                                contact_kwargs=None):
    """Build a v2 asset (with contact shadow) from an extraction result."""
    contact = derive_contact(dec, **(contact_kwargs or {}))
    fg_premult = dec.alpha[..., None] * dec.P_lin
    layers = [
        LayerSpec("t", dec.t, premultiplied=False, role="occlusion_transmission"),
        LayerSpec("q", dec.q, premultiplied=False, role="additive_bounce"),
        LayerSpec("alpha", dec.alpha, premultiplied=False, role="foreground_matte"),
        LayerSpec("foreground_premult", fg_premult, premultiplied=True,
                  role="foreground_color"),
    ]
    # Pixels the contact derivation could not claim are recorded as their own
    # status on a mask layer, so authors can find the returned-to-claim region.
    unclaimed = contact.unclaimed.astype(np.uint8)
    layers.append(LayerSpec("contact_unclaimed", unclaimed, premultiplied=False,
                            status=STATUS_CONFIRMED, role="returned_to_author_mask"))
    plane = plane or Plane()
    loaded = save_asset(
        bundle_dir, name=name, layers=layers, plane=plane,
        source_files=source_files, contact=contact.to_json(),
        repo_root=repo_root, notes=notes,
        extra_meta={"method": dec.method, "info": dec.info,
                    "contact_summary": contact.summary()})
    return loaded, contact


def migrate_gt_npz(gt_path, bundle_dir, *, name, source_files, meta=None,
                   repo_root=None):
    """Migrate a current ``gt.npz`` intermediate (full decomposition) to v2."""
    with np.load(gt_path, allow_pickle=False) as z:
        keys = set(z.files)
        if not {"t", "q", "alpha"} <= keys:
            raise AssetFormatError(f"{gt_path} is not a full decomposition",
                                   code="not_decomposable")
        t = np.clip(z["t"].astype(np.float64), 0.0, 1.0)
        q = np.asarray(z["q"], dtype=np.float64)
        alpha = np.clip(z["alpha"].astype(np.float64), 0.0, 1.0)
        P = np.asarray(z["P"] if "P" in keys else np.zeros_like(q), dtype=np.float64)
    fg_premult = alpha[..., None] * P

    class _Dec:  # minimal structural stand-in for derive_contact
        pass

    dec = _Dec()
    dec.t, dec.q, dec.alpha = t, q, alpha
    contact = derive_contact(dec)

    layers = [
        LayerSpec("t", t, premultiplied=False, role="occlusion_transmission"),
        LayerSpec("q", q, premultiplied=False, role="additive_bounce"),
        LayerSpec("alpha", alpha, premultiplied=False, role="foreground_matte"),
        LayerSpec("foreground_premult", fg_premult, premultiplied=True,
                  role="foreground_color"),
        LayerSpec("contact_unclaimed", contact.unclaimed.astype(np.uint8),
                  premultiplied=False, role="returned_to_author_mask"),
    ]
    sources = dict(source_files)
    sources.setdefault("gt_npz", os.path.abspath(gt_path))
    plane = _camera_plane(meta)
    loaded = save_asset(
        bundle_dir, name=name, layers=layers, plane=plane,
        source_files=sources, contact=contact.to_json(), repo_root=repo_root,
        notes=["migrated from current gt.npz intermediate"],
        extra_meta={"migration": {"from_version": "1", "kind": "full_decomposition",
                                  "paths": ["reversible:full_decomposition"]},
                    "contact_summary": contact.summary()})
    return loaded, contact


def write_build_lock(assets_root, entries, repo_root=None):
    """Write (or verify) the build lock covering all migrated assets."""
    rev, note = git_revision(repo_root)
    lock = {"tool_version": TOOL_VERSION, "git_revision": rev,
            "revision_note": note, "n_assets": len(entries), "assets": []}
    for entry in entries:
        bundle = entry["bundle"]
        manifest_path = os.path.join(bundle, "shadow_asset.json")
        asset = load_asset(bundle)  # strict re-verification before locking
        lock["assets"].append({
            "name": entry.get("name", os.path.basename(bundle)),
            "bundle": os.path.relpath(bundle, assets_root),
            "manifest_sha256": sha256_file(manifest_path),
            "layer_files": [{"file": l["file"], "sha256": l["sha256"],
                             "status": l["status"], "premultiplied": l["premultiplied"]}
                            for l in asset["manifest"]["layers"]],
            "source_files": asset["manifest"]["source"]["files"],
            "migration_paths": asset["manifest"].get("extra", {})
                                          .get("migration", {}).get("paths", []),
        })
    lock_path = os.path.join(assets_root, LOCK_NAME)
    with open(lock_path, "w") as f:
        json.dump(lock, f, indent=2)
    return lock_path, lock


def verify_build_lock(assets_root):
    """Re-hash every locked bundle and return (ok, mismatches)."""
    lock_path = os.path.join(assets_root, LOCK_NAME)
    if not os.path.isfile(lock_path):
        raise AssetFormatError(f"no build lock in {assets_root}", code="lock_missing")
    with open(lock_path) as f:
        lock = json.load(f)
    mismatches = []
    for entry in lock["assets"]:
        bundle = os.path.join(assets_root, entry["bundle"])
        manifest = os.path.join(bundle, "shadow_asset.json")
        if not os.path.isfile(manifest) or sha256_file(manifest) != entry["manifest_sha256"]:
            mismatches.append({"name": entry["name"], "reason": "manifest drift"})
            continue
        for lf in entry["layer_files"]:
            p = os.path.join(bundle, lf["file"])
            if not os.path.isfile(p) or sha256_file(p) != lf["sha256"]:
                mismatches.append({"name": entry["name"], "reason": f"layer drift: {lf['file']}"})
    return not mismatches, mismatches


def migrate_sample_tree(samples_root, assets_root, *, repo_root=None,
                        legacy_files=None, verify=True):
    """Migrate every sample intermediate plus explicit legacy v1 files.

    ``legacy_files`` maps ``sample_name -> [paths]`` for v1 layer files that
    exercise the reversible/underdetermined paths.  The output layout mirrors
    the samples tree under ``assets_root``.
    """
    os.makedirs(assets_root, exist_ok=True)
    entries = []
    reports = []
    for name in sorted(os.listdir(samples_root)):
        sdir = os.path.join(samples_root, name)
        gt = os.path.join(sdir, "gt.npz")
        if not os.path.isfile(gt):
            continue
        meta = None
        meta_path = os.path.join(sdir, "meta.json")
        if os.path.isfile(meta_path):
            with open(meta_path) as f:
                meta = json.load(f)
        sources = {}
        for role, fn in (("empty", "empty.png"), ("product", "product.png")):
            p = os.path.join(sdir, fn)
            if os.path.isfile(p):
                sources[role] = p
        bundle = os.path.join(assets_root, name)
        _, _ = migrate_gt_npz(gt, bundle, name=name, source_files=sources,
                              meta=meta, repo_root=repo_root)
        entries.append({"name": name, "bundle": bundle})

        for legacy_path in (legacy_files or {}).get(name, []):
            lbundle = os.path.join(assets_root, name + "_legacy_"
                                   + os.path.splitext(os.path.basename(legacy_path))[0])
            _, report = migrate_legacy(legacy_path, lbundle, name=name + "_legacy",
                                       source_files=sources, repo_root=repo_root)
            reports.append(report)
            entries.append({"name": os.path.basename(lbundle), "bundle": lbundle})

    lock_path, _ = write_build_lock(assets_root, entries, repo_root=repo_root)
    if verify:
        ok, mismatches = verify_build_lock(assets_root)
        if not ok:
            raise AssetFormatError(f"build lock verification failed: {mismatches}",
                                   code="lock_drift")
    return {"assets_root": assets_root, "lock": lock_path,
            "n_assets": len(entries), "legacy_reports": reports}
