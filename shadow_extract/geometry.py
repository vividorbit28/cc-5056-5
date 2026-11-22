"""Ray-traced SDF geometry for synthetic tabletop scenes.

Rendering is deliberately simple but physically meaningful: a small area light
above the product, a Lambertian table, and a product cast as a union of smooth
solids.  The ground plane is the only receiver; the product is a single
foreground object.  Visibility/occlusion are computed with shadow rays so the
resulting transmission field has real penumbrae rather than a painted blur.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def sphere(p, c, r):
    return np.linalg.norm(p - np.asarray(c, dtype=np.float64), axis=1) - r


def capsule(p, a, b, r):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    pa = p - a
    ba = b - a
    denom = float(ba @ ba)
    h = np.clip(pa @ ba / denom, 0.0, 1.0) if denom > 0 else np.zeros(len(p))
    return np.linalg.norm(pa - h[:, None] * ba, axis=1) - r


def rounded_box(p, c, half, r):
    q = np.abs(p - np.asarray(c, dtype=np.float64)) - np.asarray(half, dtype=np.float64)
    return (np.linalg.norm(np.maximum(q, 0.0), axis=1)
            + np.minimum(np.max(q, axis=1), 0.0) - r)


def vertical_capsule(p, x, z, y0, y1, r):
    return capsule(p, (x, y0, z), (x, y1, z), r)


@dataclass
class Primitive:
    kind: str
    params: dict

    def distance(self, p):
        if self.kind == "sphere":
            return sphere(p, **self.params)
        if self.kind == "capsule":
            return capsule(p, **self.params)
        if self.kind == "vcap":
            return vertical_capsule(p, **self.params)
        if self.kind == "rbox":
            return rounded_box(p, **self.params)
        raise ValueError(self.kind)


class Scene:
    """A product composed of primitives plus an unobstructed table plane at y=0."""

    def __init__(self, primitives: Sequence[Primitive]):
        self.primitives = list(primitives)

    def sdf(self, p):
        if not self.primitives:
            return np.full(len(p), 1e3)
        d = self.primitives[0].distance(p)
        for prim in self.primitives[1:]:
            d = np.minimum(d, prim.distance(p))
        return d

    # ---- camera / projection -------------------------------------------- #

    def project(self, p, cam):
        """World (X,Y,Z) -> pixel (col,row). Table plane is y=0."""
        X, Y, Z = p[:, 0], p[:, 1], p[:, 2]
        depth = cam.eye_y - Y
        eps = 1e-6
        x = cam.cx + cam.f * (X - cam.eye_x) / np.maximum(depth, eps)
        z = cam.cz + cam.f * (Z - cam.eye_z) / np.maximum(depth, eps)
        return x, z

    def ray_dirs(self, xs, ys, cam):
        """Pixel coordinates -> world-space directions from the camera eye."""
        d = np.stack([(xs - cam.cx) / cam.f, -np.ones_like(xs), (ys - cam.cz) / cam.f], axis=1)
        n = np.linalg.norm(d, axis=1, keepdims=True)
        return d / n

    def intersect_ground(self, xs, ys, cam):
        """Intersect camera rays with y=0; return hit points and validity."""
        d = self.ray_dirs(xs, ys, cam)
        eye = np.array([cam.eye_x, cam.eye_y, cam.eye_z])
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (0.0 - eye[1]) / d[:, 1]
        hit = eye + t[:, None] * d
        ok = t > 0
        return hit, ok

    # ---- visibility ------------------------------------------------------ #

    def visibility(self, origin, direction, max_dist, steps=96):
        """Fraction of the segment [origin, origin+direction*max_dist] that is free.

        Averaged over many samples along the ray this turns a binary shadow ray
        into a smooth soft-shadow estimate (an area-light surrogate).
        """
        s = np.linspace(0.0, 1.0, steps, dtype=np.float64)
        fracs = np.zeros(len(origin))
        chunk = 4096
        for start in range(0, len(origin), chunk):
            sl = slice(start, start + chunk)
            o = origin[sl]
            dvec = direction[sl]
            pts = o[:, None, :] + s[None, :, None] * max_dist[sl, None, None] * dvec[:, None, :]
            flat = pts.reshape(-1, 3)
            d = self.sdf(flat).reshape(len(o), steps)
            # A ray is blocked where the SDF is negative (inside a solid).  Use a
            # small band to suppress grazing artefacts.
            blocked = (d < -0.002).mean(axis=1)
            fracs[sl] = 1.0 - blocked
        return np.clip(fracs, 0.0, 1.0)

    def ambient_occlusion(self, points, samples=48, radius=0.30, seed=0):
        """Cosine-weighted AO from points on the ground plane (normal +Y)."""
        rng = np.random.default_rng(seed)
        n = len(points)
        ao = np.zeros(n)
        chunk = 2048
        for start in range(0, n, chunk):
            sl = slice(start, start + chunk)
            pts = points[sl]
            m = len(pts)
            # Cosine-weighted hemisphere directions around +Y.
            u1 = rng.random((m, samples))
            u2 = rng.random((m, samples))
            r = np.sqrt(u1)
            phi = 2 * np.pi * u2
            dirs = np.stack([r * np.cos(phi), np.sqrt(1 - u1), r * np.sin(phi)], axis=2)
            dists = np.full((m, samples), radius)
            origins = np.repeat(pts[:, None, :], samples, axis=1)
            hit = self._ray_march(origins, dirs, dists)
            ao[sl] = np.clip(1.0 - hit, 0.0, 1.0).mean(axis=1)
        return ao

    def _ray_march(self, origins, dirs, max_dist, steps=40):
        """Return 1 where the ray hits a solid within max_dist, else 0."""
        m, k, _ = dirs.shape
        t = np.zeros((m, k))
        alive = np.ones((m, k), dtype=bool)
        origins = origins.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)
        dists = max_dist.reshape(-1)
        for _ in range(steps):
            pts = origins + t.reshape(-1, 1) * dirs
            d = self.sdf(pts).reshape(m, k)
            alive &= d > 0.004
            t = np.where(alive, t + np.clip(d, 0.004, 0.08), t)
            alive &= t < dists
            if not alive.any():
                break
        return (~alive).astype(np.float64)


@dataclass
class Camera:
    """General pinhole camera looking at the table.

    World axes: X right, Y up, Z away from the viewer.  The table is y=0.
    Image coordinates: (u, v) with v increasing downward.
    """
    eye: tuple = (0.0, 2.05, 3.15)
    target: tuple = (0.0, 0.34, 0.0)
    up: tuple = (0.0, 1.0, 0.0)
    f: float = 560.0
    width: int = 640
    height: int = 480

    def __post_init__(self):
        self.eye = np.asarray(self.eye, dtype=np.float64)
        self.target = np.asarray(self.target, dtype=np.float64)
        self.up = np.asarray(self.up, dtype=np.float64)
        fwd = self.target - self.eye
        self.forward = fwd / np.linalg.norm(fwd)
        right = np.cross(self.forward, self.up)
        self.right = right / np.linalg.norm(right)
        self.up_cam = np.cross(self.right, self.forward)
        self.cx = (self.width - 1) * 0.5
        self.cy = (self.height - 1) * 0.5

    def ray_dirs(self, xs, ys):
        """Pixel coordinates (u, v) -> unit world directions (N, 3)."""
        xs = np.asarray(xs, dtype=np.float64)
        ys = np.asarray(ys, dtype=np.float64)
        d = (self.forward[None, :]
             + ((xs - self.cx) / self.f)[:, None] * self.right[None, :]
             + ((self.cy - ys) / self.f)[:, None] * self.up_cam[None, :])
        return d / np.linalg.norm(d, axis=1, keepdims=True)

    def project(self, p):
        """World points (N,3) -> (u, v, depth) in pixels."""
        rel = np.asarray(p, dtype=np.float64) - self.eye[None, :]
        z = rel @ self.forward
        x = rel @ self.right
        y = rel @ self.up_cam
        eps = 1e-9
        u = self.cx + self.f * x / np.maximum(z, eps)
        v = self.cy - self.f * y / np.maximum(z, eps)
        return u, v, z

    def intersect_ground(self, xs=None, ys=None):
        """Intersect camera rays with the plane y=0.

        Returns hit points (N,3) and a validity mask (in front, above table).
        """
        if xs is None:
            xs, ys = np.meshgrid(np.arange(self.width), np.arange(self.height))
            xs = xs.ravel().astype(np.float64)
            ys = ys.ravel().astype(np.float64)
        d = self.ray_dirs(xs, ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (0.0 - self.eye[1]) / d[:, 1]
        hit = self.eye[None, :] + t[:, None] * d
        ok = (t > 0.0) & np.isfinite(t) & (np.abs(hit[:, 2]) < 50)
        return hit, ok
