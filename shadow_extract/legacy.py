"""Legacy shadow/foreground layer files (layer format v1) and their migration.

Old intermediates come in three shapes, detected from the file itself rather
than from its extension:

* ``full_decomposition`` -- ``t``, ``q`` and ``alpha`` arrays: the physical
  decomposition, migrated losslessly into the v2 format;
* ``baked_plane`` -- a single darkening map ``s(x)`` already baked against the
  original plate.  When the source plate accompanies it the **reversible** path
  rebuilds ``t, q`` in linear light (achromatic split); otherwise the map is
  **underdetermined** and the layer is kept separately with a
  ``needs_author_claim`` status;
* ``merged_luma`` -- one luminance-only channel.  This is always underdetermined
  with respect to the (t, q) colour split; both conversion paths are offered and
  the undetermined layer is preserved so an author can keep editing it.

A critical invariant: a legacy shadow *colour* (or a display-space darkening
ratio) is never written into the transmission slot ``t``.  Only a linear-light
achromatic transmission derived from the reversible path may become ``t``.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .asset import (LayerSpec, Plane, STATUS_CONFIRMED,
                    STATUS_NEEDS_CLAIM, STATUS_PENDING, save_asset)
from .color import luminance, srgb_to_linear

LEGACY_VERSION = "1"


class LegacyFormatError(Exception):
    def __init__(self, message, code="legacy_format_error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Detection
# --------------------------------------------------------------------------- #

def inspect_legacy(path):
    """Describe a v1 layer file without converting it."""
    if not os.path.isfile(path):
        raise LegacyFormatError(f"legacy file not found: {path}", code="file_missing")
    if path.endswith(".npz"):
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            arrays = {k: z[k] for k in z.files}
    elif path.endswith(".json"):
        with open(path) as f:
            doc = json.load(f)
        if "layers" not in doc or not isinstance(doc["layers"], dict):
            raise LegacyFormatError("json legacy file needs a 'layers' object", code="bad_json")
        arrays = {k: np.asarray(v, dtype=np.float64) for k, v in doc["layers"].items()}
        keys = set(arrays)
    else:
        raise LegacyFormatError(f"unsupported legacy container: {path}", code="bad_container")

    # Normalise field ranks: scalar layers are (H, W), colour layers (H, W, 3).
    colour_keys = ("q", "P", "plate_lin", "B")
    for k in list(arrays):
        arr = arrays[k]
        if k in colour_keys:
            if arr.ndim == 2:
                arr = np.repeat(arr[..., None], 3, axis=2)
        elif arr.ndim == 3:
            arr = arr.mean(axis=2)
        arrays[k] = arr

    if {"t", "q", "alpha"} <= keys:
        kind = "full_decomposition"
    elif "scale" in keys:
        kind = "baked_plane"
    elif "luma" in keys or "merged" in keys:
        kind = "merged_luma"
    else:
        raise LegacyFormatError(f"unrecognised legacy layer keys: {sorted(keys)}",
                                code="unrecognised")

    shape = next(iter(arrays.values())).shape[:2]
    return {"path": path, "kind": kind, "keys": sorted(keys), "shape": shape,
            "arrays": arrays,
            "has_plate": "plate_lin" in keys or "B" in keys,
            "declared_version": (1 if kind == "full_decomposition" else 1)}


# --------------------------------------------------------------------------- #
# Linear-light conversions
# --------------------------------------------------------------------------- #

def _as_linear(x):
    x = np.asarray(x, dtype=np.float64)
    # Heuristic-free rule: values <= 1.5 are treated as already linear only when
    # the caller passes linear arrays; legacy sRGB maps are converted explicitly.
    return x


def reversible_split(scale_lin, plate_lin=None):
    """Reversible path: recover an achromatic ``t`` from a linear darkening map.

    ``scale_lin`` is the multiplicative darkening ``I_ground / B`` in *linear*
    light.  The bounce term cannot be separated without colour evidence, so the
    honest split is ``t = clip(scale)`` and ``q = 0``; the original map is kept
    byte-for-byte in the legacy block so the conversion is reversible.
    """
    s = np.asarray(scale_lin, dtype=np.float64)
    if s.ndim == 3:
        s = luminance(s)
    t = np.clip(s, 0.0, 1.0)
    q = np.zeros(s.shape + (3,), dtype=np.float64)
    return t, q


def underdetermined_layer(luma_map):
    """Preserve a merged-luminance shadow as its own author-claimable layer."""
    m = np.asarray(luma_map, dtype=np.float64)
    if m.ndim == 3:
        m = luminance(m)
    return m


def _to_linear_map(arr, encoded_srgb):
    arr = np.asarray(arr, dtype=np.float64)
    if encoded_srgb:
        return srgb_to_linear(np.clip(arr, 0.0, 1.0))
    return arr


# --------------------------------------------------------------------------- #
# Migration
# --------------------------------------------------------------------------- #

def migrate_legacy(path, bundle_dir, *, name=None, encoded_srgb=False,
                   plane=None, source_files=None, repo_root=None, notes=None,
                   camera=None, prefer=None):
    """Convert one v1 layer file into a v2 asset bundle.

    Returns the loaded asset plus a migration report describing which path was
    taken.  ``prefer`` for ``merged_luma`` selects ``"reversible"`` (only valid
    when a plate is present) or leaves the default underdetermined preservation.
    """
    info = inspect_legacy(path)
    kind = info["kind"]
    arrays = info["arrays"]
    shape = info["shape"]
    name = name or os.path.splitext(os.path.basename(path))[0]
    plane = plane or Plane()
    sources = dict(source_files or {})
    sources.setdefault("legacy_layer", os.path.abspath(path))
    report = {"source": path, "kind": kind, "paths": [], "warnings": []}

    if kind == "full_decomposition":
        t = np.clip(arrays["t"].astype(np.float64), 0.0, 1.0)
        q = np.asarray(arrays["q"], dtype=np.float64)
        alpha = np.clip(arrays["alpha"].astype(np.float64), 0.0, 1.0)
        P = arrays["P"] if "P" in arrays else np.zeros(shape + (3,))
        fg_premult = alpha[..., None] * np.asarray(P, dtype=np.float64)
        layers = [
            LayerSpec("t", t, premultiplied=False, role="occlusion_transmission"),
            LayerSpec("q", q, premultiplied=False, role="additive_bounce"),
            LayerSpec("alpha", alpha, premultiplied=False, role="foreground_matte"),
            LayerSpec("foreground_premult", fg_premult, premultiplied=True,
                      role="foreground_color"),
        ]
        legacy = {"representation": "full_decomposition", "evaluable": True,
                  "note": "v1 physical decomposition migrated losslessly",
                  "arrays": {"v1_t": t, "v1_q": q, "v1_alpha": alpha},
                  "params": {"layer_format_version": LEGACY_VERSION}}
        report["paths"].append("reversible:full_decomposition")

    elif kind == "baked_plane":
        scale = arrays["scale"]
        plate = arrays.get("plate_lin") if "plate_lin" in arrays else arrays.get("B")
        if plate is not None and prefer != "underdetermined":
            # Reversible path: the baked map is a linear multiplicative scale.
            scale_lin = _to_linear_map(scale, encoded_srgb)
            t, q = reversible_split(scale_lin)
            status = STATUS_CONFIRMED
            legacy = {
                "representation": "baked_plane", "evaluable": True,
                "note": "baked darkening kept; reversible achromatic split "
                        "(q unobservable, stored as 0)",
                "arrays": {"v1_scale": np.asarray(scale, float),
                           "v1_plate_lin": np.asarray(plate, float)},
                "params": {"encoded_srgb": bool(encoded_srgb),
                           "transform": "I = B * scale_lin"},
            }
            report["paths"].append("reversible:baked_plane")
        else:
            # Underdetermined: no plate, so the map cannot be un-baked.
            t = np.ones(shape, dtype=np.float64)
            q = np.zeros(shape + (3,), dtype=np.float64)
            status = STATUS_NEEDS_CLAIM
            legacy = {
                "representation": "baked_plane", "evaluable": False,
                "note": "baked darkening without source plate; kept for author claim",
                "arrays": {"v1_scale": np.asarray(scale, float)},
                "params": {"encoded_srgb": bool(encoded_srgb)},
            }
            report["paths"].append("underdetermined:baked_plane")
            report["warnings"].append("source plate absent; t/q cannot be recovered")
        alpha = np.clip(arrays["alpha"], 0.0, 1.0) if "alpha" in arrays \
            else np.zeros(shape)
        fg_premult = alpha[..., None] * (arrays["P"] if "P" in arrays
                                         else np.zeros(shape + (3,)))
        layers = [
            LayerSpec("t", t, premultiplied=False, status=status,
                      role="occlusion_transmission"),
            LayerSpec("q", q, premultiplied=False, status=status,
                      role="additive_bounce"),
            LayerSpec("alpha", alpha, premultiplied=False, role="foreground_matte"),
            LayerSpec("foreground_premult", fg_premult, premultiplied=True,
                      role="foreground_color"),
            LayerSpec("legacy_shadow_luma", underdetermined_layer(scale),
                      premultiplied=False, status=STATUS_NEEDS_CLAIM,
                      role="undetermined_legacy_shadow"),
        ]

    else:  # merged_luma -- always underdetermined wrt the (t, q) colour split
        luma = arrays.get("luma", arrays.get("merged"))
        plate = arrays.get("plate_lin") if "plate_lin" in arrays else arrays.get("B")
        alpha = np.clip(arrays["alpha"], 0.0, 1.0) if "alpha" in arrays \
            else np.zeros(shape)
        fg_premult = alpha[..., None] * (arrays["P"] if "P" in arrays
                                         else np.zeros(shape + (3,)))
        luma_lin = _to_linear_map(luma, encoded_srgb)
        if plate is not None and prefer == "reversible":
            # Reversible *achromatic* path only: luma darkening -> t, q = 0.
            t, q = reversible_split(luma_lin)
            t_status = STATUS_CONFIRMED
            report["paths"].append("reversible:merged_luma_achromatic")
        else:
            # Default: do not invent t from a colour-ambiguous map.
            t = np.ones(shape, dtype=np.float64)
            q = np.zeros(shape + (3,), dtype=np.float64)
            t_status = STATUS_PENDING
            report["paths"].append("underdetermined:merged_luma")
            report["warnings"].append(
                "merged luminance cannot determine the (t, q) colour split; "
                "layer preserved separately for author claim")
        layers = [
            LayerSpec("t", t, premultiplied=False, status=t_status,
                      role="occlusion_transmission"),
            LayerSpec("q", q, premultiplied=False, status=t_status,
                      role="additive_bounce"),
            LayerSpec("alpha", alpha, premultiplied=False, role="foreground_matte"),
            LayerSpec("foreground_premult", fg_premult, premultiplied=True,
                      role="foreground_color"),
            LayerSpec("legacy_shadow_luma", underdetermined_layer(luma_lin),
                      premultiplied=False, status=STATUS_NEEDS_CLAIM,
                      role="undetermined_legacy_shadow"),
        ]
        legacy_arrays = {"v1_luma": np.asarray(luma_lin, float)}
        if plate is not None:
            legacy_arrays["v1_plate_lin"] = np.asarray(plate, float)
        legacy = {"representation": "merged_luma",
                  "evaluable": plate is not None and prefer == "reversible",
                  "note": "single-channel luminance; underdetermined colour split "
                          "preserved for continued authoring",
                  "arrays": legacy_arrays,
                  "params": {"encoded_srgb": bool(encoded_srgb),
                             "available_paths": ["reversible:achromatic_with_plate",
                                                 "underdetermined:preserve"]}}

    if camera is not None and isinstance(plane, Plane):
        plane = Plane(origin_px=plane.origin_px, scale_px=plane.scale_px,
                      frame=plane.frame, camera=camera)

    loaded = save_asset(bundle_dir, name=name, layers=layers, plane=plane,
                        source_files=sources, legacy=legacy,
                        notes=(notes or []) + report["warnings"],
                        repo_root=repo_root,
                        extra_meta={"migration": {"from_version": LEGACY_VERSION,
                                                  "kind": kind,
                                                  "paths": report["paths"]}})
    report["bundle"] = bundle_dir
    report["statuses"] = {e["name"]: e["status"] for e in loaded["manifest"]["layers"]}
    return loaded, report


def reopen_underdetermined(bundle_dir):
    """Re-open a migrated asset and return the layer awaiting author claim.

    Raises if the asset has no such layer; the author can edit the returned
    array and re-save it through :func:`save_underdetermined_edit`.
    """
    from .asset import load_asset
    loaded = load_asset(bundle_dir)
    statuses = {e["name"]: e["status"] for e in loaded["manifest"]["layers"]}
    if "legacy_shadow_luma" not in statuses:
        raise LegacyFormatError("asset contains no underdetermined legacy layer",
                                code="no_underdetermined")
    return loaded, loaded["layers"]["legacy_shadow_luma"]


def save_underdetermined_edit(bundle_dir, edited_luma, status=STATUS_NEEDS_CLAIM):
    """Persist an author edit of the underdetermined layer and re-lock hashes."""
    from .asset import load_asset, _layer_filename, sha256_file  # noqa: PLC2701
    import json as _json
    loaded = load_asset(bundle_dir)
    name = "legacy_shadow_luma"
    fpath = os.path.join(bundle_dir, _layer_filename(name))
    np.save(fpath, np.ascontiguousarray(edited_luma, dtype=np.float64))
    for entry in loaded["manifest"]["layers"]:
        if entry["name"] == name:
            entry["sha256"] = sha256_file(fpath)
            entry["status"] = status
            entry["shape"] = list(np.asarray(edited_luma).shape)
    with open(os.path.join(bundle_dir, "shadow_asset.json"), "w") as f:
        _json.dump(loaded["manifest"], f, indent=2)
    return load_asset(bundle_dir)
