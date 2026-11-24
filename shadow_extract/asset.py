"""Versioned shadow/foreground asset format (layer format v2).

A v2 asset bundle is a directory containing a ``shadow_asset.json`` manifest
plus one ``.npy`` file per layer.  Every quantity that old intermediates left
implicit is declared explicitly in the manifest:

* **premultiplied state** -- each colour layer carries a boolean
  ``premultiplied`` flag; the foreground is stored premultiplied (``alpha*P``)
  and the flag is checked on load, never guessed;
* **plane coordinates** -- a ``plane`` block declares the coordinate frame,
  origin and pixel scale the layer arrays live in, with the camera projection
  when one is known, so a moved product can be placed geometrically;
* **source revision** -- the git revision of the converting code plus sha256
  of every source file, so an asset is traceable to exactly what produced it.

Layer files are content-hashed in the manifest; loading verifies the hashes.
A missing declaration or a hash mismatch raises :class:`AssetFormatError`
instead of being papered over with a default value.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from . import __version__ as TOOL_VERSION

FORMAT_NAME = "shadow-asset"
LAYER_FORMAT_VERSION = "2.0"
LEGACY_LAYER_FORMAT_VERSION = "1"
COLOR_SPACE = "linear_rgb"

# Layer lifecycle statuses.
STATUS_CONFIRMED = "confirmed"
STATUS_NEEDS_CLAIM = "needs_author_claim"
STATUS_PENDING = "pending_explanation"
KNOWN_STATUSES = {STATUS_CONFIRMED, STATUS_NEEDS_CLAIM, STATUS_PENDING}


class AssetFormatError(Exception):
    """Raised when an asset bundle is incomplete or fails verification."""

    def __init__(self, message, code="asset_format_error"):
        super().__init__(message)
        self.code = code


# --------------------------------------------------------------------------- #
# Hashing / revision
# --------------------------------------------------------------------------- #

def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git_revision(repo_root=None):
    """Return (revision, note).  A missing git repo is reported, not faked."""
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo_root or os.getcwd(),
                             capture_output=True, text=True, timeout=10)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None, "git not available"
    if out.returncode != 0:
        return None, "not a git working tree"
    rev = out.stdout.strip()
    if not rev:
        return None, "git returned no revision"
    dirty = subprocess.run(["git", "status", "--porcelain"], cwd=repo_root or os.getcwd(),
                           capture_output=True, text=True, timeout=10)
    note = "working tree dirty at conversion" if dirty.stdout.strip() else None
    return rev, note


# --------------------------------------------------------------------------- #
# Plane coordinate frame
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Plane:
    """The 2-D plane the layer arrays are sampled on.

    ``origin_px`` is the plane coordinate of pixel (0, 0) and ``scale_px`` the
    plane-unit size of one pixel along (row, col).  The default frame is plain
    image pixels; a renderer-supplied camera is recorded alongside for assets
    that know their world projection.
    """

    origin_px: tuple = (0.0, 0.0)
    scale_px: tuple = (1.0, 1.0)
    frame: str = "image_pixel"
    camera: dict | None = None

    def to_dict(self):
        return {"frame": self.frame, "origin_px": list(self.origin_px),
                "scale_px": list(self.scale_px),
                "camera": self.camera if self.camera is not None else None}

    @classmethod
    def from_dict(cls, d):
        if not isinstance(d, dict) or "origin_px" not in d or "scale_px" not in d:
            raise AssetFormatError("plane block must declare frame, origin_px and scale_px",
                                   code="plane_undeclared")
        return cls(origin_px=tuple(d["origin_px"]), scale_px=tuple(d["scale_px"]),
                   frame=d.get("frame", "image_pixel"), camera=d.get("camera"))

    def pixel_to_plane(self, row, col):
        return (self.origin_px[0] + row * self.scale_px[0],
                self.origin_px[1] + col * self.scale_px[1])

    def plane_to_pixel(self, prow, pcol):
        return ((prow - self.origin_px[0]) / self.scale_px[0],
                (pcol - self.origin_px[1]) / self.scale_px[1])


# --------------------------------------------------------------------------- #
# Premultiplication helpers
# --------------------------------------------------------------------------- #

def premultiply(P_lin, alpha):
    """Straight linear colour -> premultiplied foreground ``alpha*P``."""
    a = np.asarray(alpha, dtype=np.float64)[..., None]
    return np.asarray(P_lin, dtype=np.float64) * a


def unpremultiply(F_premult, alpha):
    """Premultiplied foreground -> straight colour (zero where alpha is zero)."""
    a = np.asarray(alpha, dtype=np.float64)[..., None]
    return np.where(a > 1e-6, np.asarray(F_premult, dtype=np.float64) / np.maximum(a, 1e-6), 0.0)


# --------------------------------------------------------------------------- #
# Layer declaration
# --------------------------------------------------------------------------- #

@dataclass
class LayerSpec:
    name: str
    array: np.ndarray
    premultiplied: bool
    status: str = STATUS_CONFIRMED
    role: str = ""

    def validate(self):
        if not isinstance(self.premultiplied, bool):
            raise AssetFormatError(f"layer {self.name!r}: premultiplied must be an explicit bool",
                                   code="premult_undeclared")
        if self.status not in KNOWN_STATUSES:
            raise AssetFormatError(f"layer {self.name!r}: unknown status {self.status!r}",
                                   code="unknown_status")
        if not isinstance(self.array, np.ndarray) or self.array.size == 0:
            raise AssetFormatError(f"layer {self.name!r}: empty or non-ndarray data",
                                   code="empty_layer")


# --------------------------------------------------------------------------- #
# Save / load
# --------------------------------------------------------------------------- #

def _layer_filename(name):
    return f"layer_{name}.npy"


def save_asset(bundle_dir, *, name, layers, plane, source_files,
               contact=None, legacy=None, notes=None, repo_root=None,
               extra_meta=None):
    """Write a v2 asset bundle and return its verified manifest.

    Parameters
    ----------
    layers : list[LayerSpec]
    source_files : dict[role, path] -- hashed into the manifest's source block.
    contact : optional dict produced by :mod:`shadow_extract.contact`.
    legacy : optional legacy-provenance block (see :mod:`shadow_extract.legacy`).
    """
    if not layers:
        raise AssetFormatError("an asset requires at least one layer", code="empty_asset")
    os.makedirs(bundle_dir, exist_ok=True)
    for spec in layers:
        spec.validate()
    shape = layers[0].array.shape[:2]
    for spec in layers:
        if spec.array.shape[:2] != shape:
            raise AssetFormatError(
                f"layer {spec.name!r} shape {spec.array.shape[:2]} != asset shape {shape}",
                code="shape_mismatch")

    rev, rev_note = git_revision(repo_root)
    src = []
    for role, path in sorted(source_files.items()):
        if not os.path.isfile(path):
            raise AssetFormatError(f"source file for {role!r} not found: {path}",
                                   code="source_missing")
        src.append({"role": role, "path": os.path.basename(path),
                    "sha256": sha256_file(path)})

    layer_entries = []
    for spec in layers:
        fname = _layer_filename(spec.name)
        fpath = os.path.join(bundle_dir, fname)
        np.save(fpath, np.ascontiguousarray(spec.array))
        layer_entries.append({
            "name": spec.name, "file": fname,
            "sha256": sha256_file(fpath),
            "dtype": str(spec.array.dtype), "shape": list(spec.array.shape),
            "premultiplied": spec.premultiplied, "status": spec.status,
            "role": spec.role,
        })

    contact_block = None
    if contact is not None:
        contact_path = os.path.join(bundle_dir, "contact.json")
        with open(contact_path, "w") as f:
            json.dump(contact, f, indent=2)
        contact_block = {"file": "contact.json", "sha256": sha256_file(contact_path)}

    legacy_dir = None
    if legacy is not None:
        legacy_dir = _write_legacy_block(bundle_dir, legacy)

    manifest = {
        "format": FORMAT_NAME,
        "format_version": LAYER_FORMAT_VERSION,
        "name": name,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "color_space": COLOR_SPACE,
        "shape": list(shape),
        "tool": {"name": "shadow_extract", "version": TOOL_VERSION,
                 "layer_format_upgrade": {"from": LEGACY_LAYER_FORMAT_VERSION,
                                          "to": LAYER_FORMAT_VERSION}},
        "plane": plane.to_dict() if isinstance(plane, Plane) else Plane(**plane).to_dict(),
        "source": {"git_revision": rev, "revision_note": rev_note, "files": src},
        "layers": layer_entries,
        "contact": contact_block,
        "legacy": legacy_dir,
        "notes": list(notes or []),
    }
    if extra_meta:
        manifest["extra"] = extra_meta
    manifest_path = os.path.join(bundle_dir, "shadow_asset.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    # Self-verify immediately: a bundle we cannot re-read is not delivered.
    loaded = load_asset(bundle_dir)
    return loaded


def _write_legacy_block(bundle_dir, legacy):
    """Persist auxiliary legacy arrays and return a JSON-serialisable block."""
    legacy_dir = os.path.join(bundle_dir, "legacy")
    os.makedirs(legacy_dir, exist_ok=True)
    block = {"representation": legacy.get("representation", "baked_plane"),
             "path": legacy.get("path"), "evaluable": bool(legacy.get("evaluable", False)),
             "note": legacy.get("note", ""), "arrays": [], "params": legacy.get("params", {})}
    for key, arr in legacy.get("arrays", {}).items():
        fname = f"{key}.npy"
        fpath = os.path.join(legacy_dir, fname)
        np.save(fpath, np.ascontiguousarray(arr))
        block["arrays"].append({"name": key, "file": fname,
                                "sha256": sha256_file(fpath),
                                "shape": list(np.asarray(arr).shape)})
    return block


def load_asset(bundle_dir):
    """Read and strictly verify a v2 asset bundle."""
    manifest_path = os.path.join(bundle_dir, "shadow_asset.json")
    if not os.path.isfile(manifest_path):
        raise AssetFormatError(f"no shadow_asset.json in {bundle_dir}", code="manifest_missing")
    with open(manifest_path) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT_NAME:
        raise AssetFormatError(f"not a {FORMAT_NAME} bundle: {manifest.get('format')!r}",
                               code="wrong_format")
    if str(manifest.get("format_version")) != LAYER_FORMAT_VERSION:
        raise AssetFormatError(
            f"unsupported layer format version {manifest.get('format_version')!r}; "
            f"this build reads {LAYER_FORMAT_VERSION}", code="version_unsupported")
    if "plane" not in manifest:
        raise AssetFormatError("asset declares no plane coordinate frame",
                               code="plane_undeclared")
    plane = Plane.from_dict(manifest["plane"])
    if "source" not in manifest or "files" not in manifest["source"]:
        raise AssetFormatError("asset declares no source revision block",
                               code="source_undeclared")
    src = manifest["source"]
    if "git_revision" not in src:
        raise AssetFormatError("source.git_revision must be declared (null is allowed "
                               "only with a revision_note)", code="revision_undeclared")
    if src["git_revision"] is None and not src.get("revision_note"):
        raise AssetFormatError("null git revision requires an explanatory revision_note",
                               code="revision_undeclared")

    layers = {}
    for entry in manifest["layers"]:
        for required in ("name", "file", "sha256", "premultiplied", "status"):
            if required not in entry:
                raise AssetFormatError(f"layer entry missing {required!r}",
                                       code="layer_declaration_incomplete")
        if not isinstance(entry["premultiplied"], bool):
            raise AssetFormatError(f"layer {entry['name']!r}: premultiplied flag not bool",
                                   code="premult_undeclared")
        fpath = os.path.join(bundle_dir, entry["file"])
        if not os.path.isfile(fpath):
            raise AssetFormatError(f"layer file missing: {entry['file']}", code="layer_file_missing")
        actual = sha256_file(fpath)
        if actual != entry["sha256"]:
            raise AssetFormatError(
                f"layer {entry['name']!r} hash mismatch: manifest {entry['sha256'][:12]} "
                f"!= file {actual[:12]}", code="layer_hash_mismatch")
        arr = np.load(fpath, mmap_mode=None)
        if list(arr.shape) != list(entry["shape"]):
            raise AssetFormatError(f"layer {entry['name']!r} shape mismatch",
                                   code="shape_mismatch")
        layers[entry["name"]] = arr

    contact = None
    if manifest.get("contact"):
        cpath = os.path.join(bundle_dir, manifest["contact"]["file"])
        if sha256_file(cpath) != manifest["contact"]["sha256"]:
            raise AssetFormatError("contact block hash mismatch", code="contact_hash_mismatch")
        with open(cpath) as f:
            contact = json.load(f)

    return {"manifest": manifest, "plane": plane, "layers": layers, "contact": contact,
            "dir": bundle_dir}


def layer_statuses(loaded):
    return {e["name"]: e["status"] for e in loaded["manifest"]["layers"]}


def is_premultiplied(loaded, layer_name):
    for e in loaded["manifest"]["layers"]:
        if e["name"] == layer_name:
            return e["premultiplied"]
    raise AssetFormatError(f"no layer named {layer_name!r}", code="layer_missing")
