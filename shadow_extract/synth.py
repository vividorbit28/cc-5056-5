"""Physically-based synthetic tabletop renderer with ground-truth layers.

Forward model (linear light), matching what the estimator inverts::

    background radiance      B  = beta * E0                (empty plate)
    ground under product     Ig = t * B + A
    total transmission       t  = (Ea*AO + Ed*Vy*V) / (Ea + Ed*Vy)  in [0,1]
    additive product light   A  = beta * G                 (rgb bounce)
    product pixel            I  = alpha * P + (1-alpha) * Ig

``t`` is achromatic occlusion transmission derived from real shadow rays through
an SDF; ``V`` is sampled over a finite angular light disc, which produces
genuine penumbrae.  ``A`` is bounce light from the product.  Because the table
is Lambertian, ``A`` is proportional to local reflectance ``beta`` and therefore
to ``B``; the plate-independent field is ``q = A / B = G / E0``.  A second
plate ``B2`` is lit by ``B2*(t + q)`` with the *same* ``t`` and ``q``, which is
the transfer property the extractor is tested against.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict

import cv2
import numpy as np

from .color import linear_to_srgb
from .geometry import Camera, Primitive, Scene
from .plates import make_plate


# --------------------------------------------------------------------------- #
# Product catalogue
# --------------------------------------------------------------------------- #

PRODUCTS = {
    "bottle": {
        "prims": [
            ("sphere", {"c": (0.0, 0.52, 0.0), "r": 0.44}),
            ("capsule", {"a": (0.0, 0.85, 0.0), "b": (0.0, 1.24, 0.0), "r": 0.15}),
            ("sphere", {"c": (0.0, 1.28, 0.0), "r": 0.18}),
        ],
        "albedo": (0.78, 0.20, 0.14),
    },
    "can": {
        "prims": [
            ("capsule", {"a": (0.0, 0.40, 0.0), "b": (0.0, 1.05, 0.0), "r": 0.34}),
        ],
        "albedo": (0.22, 0.42, 0.72),
    },
    "ball": {
        "prims": [("sphere", {"c": (0.0, 0.42, 0.0), "r": 0.42})],
        "albedo": (0.86, 0.55, 0.18),
    },
    "box": {
        "prims": [("rbox", {"c": (0.0, 0.33, 0.0), "half": (0.40, 0.33, 0.30), "r": 0.05})],
        "albedo": (0.34, 0.62, 0.30),
    },
    "mug": {
        "prims": [
            ("rbox", {"c": (0.0, 0.36, 0.0), "half": (0.30, 0.36, 0.30), "r": 0.08}),
            ("capsule", {"a": (0.32, 0.50, 0.0), "b": (0.54, 0.50, 0.0), "r": 0.06}),
            ("capsule", {"a": (0.54, 0.50, 0.0), "b": (0.54, 0.24, 0.0), "r": 0.06}),
            ("capsule", {"a": (0.54, 0.24, 0.0), "b": (0.30, 0.20, 0.0), "r": 0.06}),
        ],
        "albedo": (0.82, 0.78, 0.68),
    },
    "jar": {
        "prims": [
            ("capsule", {"a": (0.0, 0.22, 0.0), "b": (0.0, 0.78, 0.0), "r": 0.36}),
            ("capsule", {"a": (0.0, 0.80, 0.0), "b": (0.0, 0.94, 0.0), "r": 0.30}),
            ("capsule", {"a": (0.0, 0.96, 0.0), "b": (0.0, 1.02, 0.0), "r": 0.33}),
        ],
        "albedo": (0.55, 0.30, 0.62),
    },
}


def build_product(kind, scale=1.0):
    spec = PRODUCTS[kind]
    prims = []
    for pk, pp in spec["prims"]:
        params = {}
        for key, val in pp.items():
            if isinstance(val, tuple):
                params[key] = tuple(float(v) * scale for v in val)
            else:
                params[key] = float(val) * scale
        prims.append(Primitive(pk, params))
    return Scene(prims), np.asarray(spec["albedo"], dtype=np.float64)


# --------------------------------------------------------------------------- #
# Lighting helpers
# --------------------------------------------------------------------------- #

@dataclass
class Light:
    # Near-vertical key so the ground shadow stays contact-local, plus a
    # moderate ambient term that gives the unoccluded ground a healthy level.
    key_dir: tuple = (0.30, 0.905, 0.30)
    key_color: tuple = (1.0, 0.97, 0.92)
    ambient: float = 0.42
    sky_color: tuple = (0.90, 0.95, 1.0)
    angular_radius: float = 0.045   # radians; controls penumbra width
    n_samples: int = 12
    bounce_gain: float = 0.055     # additive bounce strength
    bounce_falloff: float = 0.55   # world units

    def key_unit(self):
        v = np.asarray(self.key_dir, dtype=np.float64)
        return v / np.linalg.norm(v)

    def e0(self):
        """Plate irradiance vector (folded with 1/pi)."""
        l = self.key_unit()
        return self.ambient * np.asarray(self.sky_color) + l[1] * np.asarray(self.key_color)

    def e0_scalar(self):
        return float(self.ambient + self.key_unit()[1])


def _basis(d):
    a = np.array([0.0, 1.0, 0.0]) if abs(d[1]) > 0.8 else np.array([1.0, 0.0, 0.0])
    u = np.cross(d, a)
    u /= np.linalg.norm(u)
    v = np.cross(d, u)
    return u, v


def _light_dirs(light, rng, k):
    l = light.key_unit()
    u, v = _basis(l)
    # Uniform-ish disc directions (slightly smaller than the tangent disc).
    r = light.angular_radius * np.sqrt(rng.random(k))
    phi = 2 * np.pi * rng.random(k)
    return l[None, :] + (r * np.cos(phi))[:, None] * u[None, :] + (r * np.sin(phi))[:, None] * v[None, :], l


def _march_blocked(scene, origins, dirs, max_dist, steps=22, t0=0.01):
    """Return blocked mask for (N,K) rays.  origins (N,3), dirs (N,K,3).

    A ray is "blocked" only if it actually enters a solid.  Rays that exhaust
    ``max_dist`` without a hit are unoccluded, which is the common case for
    ground points far from the product.
    """
    N, K, _ = dirs.shape
    o = np.repeat(origins[:, None, :], K, axis=1).reshape(-1, 3)
    d = dirs.reshape(-1, 3)
    t = np.full(N * K, t0)
    hit = np.zeros(N * K, dtype=bool)
    alive = np.ones(N * K, dtype=bool)
    far = np.full(N * K, float(max_dist))
    step_max = max_dist / steps
    for _ in range(steps):
        pts = o + t[:, None] * d
        sdf = scene.sdf(pts)
        newly_hit = alive & (sdf < 1e-4)
        hit |= newly_hit
        alive &= ~newly_hit
        t = np.where(alive, t + np.clip(sdf, 0.005, step_max), t)
        alive &= t < far
        if not alive.any():
            break
    return hit.reshape(N, K)


def _march_aperture(scene, origins, dirs, radius, steps=22):
    """Hemisphere AO: blocked count within ``radius`` (dirs (N,K,3))."""
    return _march_blocked(scene, origins, dirs, radius, steps=steps)


def _cosine_hemisphere(rng, n, k):
    u1 = rng.random((n, k))
    u2 = rng.random((n, k))
    r = np.sqrt(u1)
    phi = 2 * np.pi * u2
    return np.stack([r * np.cos(phi), np.sqrt(np.maximum(0.0, 1 - u1)), r * np.sin(phi)], axis=2)


# --------------------------------------------------------------------------- #
# Ground fields (t, q) on a coarse grid, upsampled
# --------------------------------------------------------------------------- #

def _ground_fields(cam, scene, light, beta_mean, H, W, coarse=2, rng=None):
    """Compute t (H,W) and q (H,W,3).

    The camera-ray ground hit is cheap, so it is evaluated at full resolution.
    Occlusion, however, is only relevant near the product: a coarse SDF pass
    locates that band and the expensive shadow rays are cast only there, at
    full pixel resolution.
    """
    if rng is None:
        rng = np.random.default_rng(7)
    xs, ys = np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))
    xs = xs.ravel()
    ys = ys.ravel()
    hit, ok = cam.intersect_ground(xs, ys)

    t = np.ones(H * W)
    q = np.zeros((H * W, 3))
    dist = np.full(H * W, np.inf)
    dist[ok] = np.maximum(scene.sdf(hit[ok]), 0.0)

    band = ok & (dist < 2.4)
    if band.any():
        pb = hit[band]
        l = light.key_unit()
        ldirs, _ = _light_dirs(light, rng, light.n_samples)
        dirs = np.broadcast_to(ldirs[None, :, :], (len(pb), light.n_samples, 3)).copy()
        blocked = _march_blocked(scene, pb, dirs, max_dist=8.0, steps=26)
        V = 1.0 - blocked.mean(axis=1)

        adirs = _cosine_hemisphere(rng, len(pb), 20)
        ablocked = _march_aperture(scene, pb, adirs, radius=0.55, steps=20)
        AO = 1.0 - ablocked.mean(axis=1)

        e0s = light.e0_scalar()
        t[band] = (light.ambient * AO + light.key_unit()[1] * V) / e0s

        e0v = light.e0()
        prod_rad = beta_mean * (light.ambient + light.key_unit()[1])
        d = np.maximum(dist[band], 0.0)
        G = light.bounce_gain * prod_rad[None, :] * np.exp(-d / light.bounce_falloff)[:, None]
        q[band] = G / np.maximum(e0v[None, :], 1e-4)

    t = np.clip(t.reshape(H, W), 0.0, 1.0)
    q = np.maximum(q.reshape(H, W, 3), 0.0)
    return t, q


# --------------------------------------------------------------------------- #
# Product rendering
# --------------------------------------------------------------------------- #

def _normals(scene, p, eps=0.004):
    n = np.zeros_like(p)
    for axis in range(3):
        d = np.zeros(3)
        d[axis] = eps
        n[:, axis] = scene.sdf(p + d[None, :]) - scene.sdf(p - d[None, :])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    return n


def _scene_bounds(scene):
    lo = np.array([np.inf, np.inf, np.inf])
    hi = np.array([-np.inf, -np.inf, -np.inf])
    for prim in scene.primitives:
        kind, pp = prim.kind, prim.params
        if kind == "sphere":
            c = np.asarray(pp["c"], float)
            r = pp["r"]
            lo = np.minimum(lo, c - r); hi = np.maximum(hi, c + r)
        elif kind == "capsule":
            a = np.asarray(pp["a"], float); b = np.asarray(pp["b"], float); r = pp["r"]
            lo = np.minimum(lo, np.minimum(a, b) - r); hi = np.maximum(hi, np.maximum(a, b) + r)
        elif kind == "vcap":
            r = pp["r"]
            a = np.array([pp["x"], pp["y0"], pp["z"]]); b = np.array([pp["x"], pp["y1"], pp["z"]])
            lo = np.minimum(lo, np.minimum(a, b) - r); hi = np.maximum(hi, np.maximum(a, b) + r)
        elif kind == "rbox":
            c = np.asarray(pp["c"], float); hhalf = np.asarray(pp["half"], float) + pp["r"]
            lo = np.minimum(lo, c - hhalf); hi = np.maximum(hi, c + hhalf)
    return lo, hi


def _project_roi(cam, lo, hi, pad_frac=0.10):
    corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                        for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    u, v, z = cam.project(corners)
    if np.any(z <= 1e-3):
        return 0, cam.width, 0, cam.height
    u0 = int(np.floor(u.min())); u1 = int(np.ceil(u.max()))
    v0 = int(np.floor(v.min())); v1 = int(np.ceil(v.max()))
    pad = int(pad_frac * max(u1 - u0, v1 - v0)) + 12
    return (max(0, u0 - pad), min(cam.width, u1 + pad),
            max(0, v0 - pad), min(cam.height, v1 + pad))


def _render_product(cam, scene, albedo, light, ss=2, max_dist=10.0, feather=0.0):
    """Primary-visibility render of the product with supersampled coverage."""
    H, W = cam.height, cam.width
    alpha = np.zeros((H, W))
    Pacc = np.zeros((H, W, 3))
    Pcnt = np.zeros((H, W))
    lo, hi = _scene_bounds(scene)
    x0, x1, y0, y1 = _project_roi(cam, lo, hi)
    if x1 <= x0 or y1 <= y0:
        return alpha, Pacc
    offs = (np.arange(ss) + 0.5) / ss
    l = light.key_unit()
    hvec = l + np.array([0.0, 1.0, 0.0])
    hvec /= np.linalg.norm(hvec)
    spec_w, spec_p = 0.10, 45.0
    for sy in range(ss):
        for sx in range(ss):
            xs, ys = np.meshgrid(np.arange(x0, x1) + offs[sx], np.arange(y0, y1) + offs[sy])
            xs = xs.ravel().astype(np.float64)
            ys = ys.ravel().astype(np.float64)
            dirs = cam.ray_dirs(xs, ys)
            t = np.zeros(len(xs))
            sdf_min = np.full(len(xs), np.inf)
            alive = np.ones(len(xs), dtype=bool)
            for _ in range(120):
                pts = cam.eye[None, :] + t[:, None] * dirs
                sdf = scene.sdf(pts)
                sdf_min = np.minimum(sdf_min, sdf)
                alive &= sdf > 1e-3
                t = np.where(alive, t + np.clip(sdf, 0.004, 0.20), t)
                alive &= t < max_dist
                if not alive.any():
                    break
            if feather > 0:
                cov = np.clip((feather - np.maximum(sdf_min, 0.0)) / feather, 0.0, 1.0)
                cov[sdf_min <= 0] = 1.0
            else:
                cov = (sdf_min <= 2e-3).astype(np.float64)
            sub = cov > 1e-3
            if not sub.any():
                continue
            # Shade at the closest approach (also works for the feather shell).
            ph = cam.eye[None, :] + (t[:, None] + np.maximum(sdf_min[:, None], 0.0)) * dirs
            nh = _normals(scene, ph[sub])
            ndl = np.clip(nh @ l, 0.0, 1.0)
            ndh = np.clip(nh @ hvec, 0.0, 1.0)
            amb = 0.55 + 0.45 * np.clip(ph[sub, 1] / 0.8, 0.0, 1.0) ** 0.5
            shade = (albedo[None, :] * (light.ambient * amb[:, None] * np.asarray(light.sky_color)[None, :]
                                        + ndl[:, None] * np.asarray(light.key_color)[None, :])
                     + spec_w * (ndh ** spec_p)[:, None] * np.asarray(light.key_color)[None, :])
            yy = ys[sub].astype(int)
            xx = xs[sub].astype(int)
            c = cov[sub]
            np.add.at(Pacc, (yy, xx), shade * c[:, None])
            np.add.at(Pcnt, (yy, xx), c)
            np.add.at(alpha, (yy, xx), c / (ss * ss))
    alpha = np.clip(alpha, 0.0, 1.0)
    Pm = np.zeros_like(Pacc)
    nz = Pcnt > 0
    Pm[nz] = Pacc[nz] / Pcnt[nz][:, None]
    # Fill uncovered-but-adjacent pixels with the nearest product colour so that
    # matte edges do not blend toward black.
    if np.any(nz) and np.any(~nz):
        _, idx = _nearest_fill(nz)
        Pm[~nz] = Pm[tuple(idx)]
    return alpha, Pm


def _nearest_fill(mask):
    """Reliable nearest-neighbour fill using distance transform labels."""
    ys, xs = np.where(mask)
    if len(ys) == 0:
        return None
    # Build index image and use cv2.distanceTransformWithLabels on the *inverse*
    idx_img = np.full(mask.shape, -1, dtype=np.int32)
    flat_idx = np.arange(len(ys))
    idx_img[ys, xs] = flat_idx
    # Use scipy cKDTree for clarity/correctness.
    from scipy.spatial import cKDTree
    pts = np.stack([ys, xs], axis=1)
    tree = cKDTree(pts)
    qy, qx = np.where(~mask)
    _, nn = tree.query(np.stack([qy, qx], axis=1), k=1)
    return (qy, qx), (ys[nn], xs[nn])


def render(cam, scene, albedo, beta, light, ss=2, field_scale=2, rng=None, feather=0.0):
    H, W = cam.height, cam.width
    if rng is None:
        rng = np.random.default_rng(7)
    t, q = _ground_fields(cam, scene, light, albedo, H, W, rng=rng)
    alpha, P = _render_product(cam, scene, albedo, light, ss=ss, feather=feather)
    e0 = light.e0()
    B = beta * e0[None, None, :]
    ground = B * t[..., None] + B * q
    I = alpha[..., None] * P + (1.0 - alpha[..., None]) * ground
    return dict(alpha=alpha, P=P, t=t, q=q, B=B, ground=ground, I=I,
                beta=beta, e0=e0)


# --------------------------------------------------------------------------- #
# Sample builder
# --------------------------------------------------------------------------- #

SAMPLE_SPECS = {
    "soft_paper": dict(product="bottle", plate="paper", base=(0.62, 0.58, 0.50),
                       contrast=0.20, light=dict(angular_radius=0.16, bounce_gain=0.06)),
    "hard_wood": dict(product="can", plate="wood", base=(0.55, 0.42, 0.30),
                      contrast=0.30, light=dict(angular_radius=0.035, bounce_gain=0.04)),
    "textured_fabric": dict(product="ball", plate="fabric", base=(0.55, 0.52, 0.48),
                            contrast=0.30, light=dict(angular_radius=0.11, bounce_gain=0.07)),
    "dark_marble": dict(product="box", plate="marble", base=(0.13, 0.13, 0.15),
                        contrast=0.35, light=dict(angular_radius=0.09, bounce_gain=0.05)),
    "mug_light": dict(product="mug", plate="paper", base=(0.72, 0.70, 0.66),
                      contrast=0.16, light=dict(angular_radius=0.13, bounce_gain=0.05)),
    "vase_fabric": dict(product="jar", plate="fabric", base=(0.62, 0.60, 0.57),
                        contrast=0.26, light=dict(angular_radius=0.13, bounce_gain=0.06)),
    "soft_edge_ball": dict(product="ball", plate="paper", base=(0.66, 0.63, 0.58),
                           contrast=0.18, feather=0.06,
                           light=dict(angular_radius=0.10, bounce_gain=0.06)),
    "penumbra_can": dict(product="can", plate="paper", base=(0.70, 0.66, 0.60),
                         contrast=0.17,
                         light=dict(angular_radius=0.16, bounce_gain=0.05)),
}


def make_scene_sample(name, seed=0, cam=None, spec=None, second_plate=True):
    spec = spec or SAMPLE_SPECS[name]
    rng = np.random.default_rng(seed)
    cam = cam or Camera()
    scene, albedo = build_product(spec["product"])
    light = Light(**spec.get("light", {}))
    shape = (cam.height, cam.width)
    beta1 = make_plate(shape, rng, spec["plate"], spec["base"], spec["contrast"])
    r1 = render(cam, scene, albedo, beta1, light, rng=rng, feather=spec.get("feather", 0.0))

    out = dict(name=name, alpha=r1["alpha"], P=r1["P"], t=r1["t"], q=r1["q"],
               B1=r1["B"], I1=r1["I"], beta1=beta1,
               cam=dict(eye=[float(v) for v in cam.eye], target=[float(v) for v in cam.target],
                        up=[float(v) for v in cam.up], f=float(cam.f),
                        width=int(cam.width), height=int(cam.height)),
               light=asdict(light), product=spec["product"])
    if second_plate:
        # Same physical shade, different table surface.
        beta2 = make_plate(shape, np.random.default_rng(seed + 991), "wood" if spec["plate"] != "wood" else "marble",
                           (0.46, 0.50, 0.58), 0.30)
        B2 = beta2 * r1["e0"][None, None, :]
        ground2 = B2 * r1["t"][..., None] + B2 * r1["q"]
        I2 = r1["alpha"][..., None] * r1["P"] + (1.0 - r1["alpha"][..., None]) * ground2
        out.update(B2=B2, I2=I2, beta2=beta2, ground2=ground2)
    return out


def apply_measurement(rgb_lin, gain=(1.0, 1.0, 1.0), shift=(0.0, 0.0), blur=0.0,
                      noise=0.0, rng=None, seed=0):
    """Simulate capture differences and encode to 8-bit sRGB."""
    rng = rng or np.random.default_rng(seed)
    img = rgb_lin * np.asarray(gain)[None, None, :]
    if blur > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur)
    if shift != (0.0, 0.0):
        M = np.float32([[1, 0, shift[0]], [0, 1, shift[1]]])
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT)
    srgb = linear_to_srgb(img)
    if noise > 0:
        srgb = srgb + rng.normal(0, noise, srgb.shape)
        # Noise is applied in display space; that is how a real capture behaves.
    srgb = np.clip(srgb, 0.0, 1.0)
    return np.round(srgb * 255.0).astype(np.uint8)


def save_sample(out, out_dir, name, perturb=True, seed=0):
    os.makedirs(out_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    empty = apply_measurement(out["B1"], rng=rng, seed=seed)
    prod = apply_measurement(out["I1"], rng=rng, seed=seed + 1)
    cv2.imwrite(os.path.join(out_dir, "empty.png"), cv2.cvtColor(empty, cv2.COLOR_RGB2BGR))
    cv2.imwrite(os.path.join(out_dir, "product.png"), cv2.cvtColor(prod, cv2.COLOR_RGB2BGR))
    arrs = {k: out[k].astype(np.float32) for k in
            ("alpha", "P", "t", "q", "B1", "I1", "beta1")}
    meta = dict(name=name, cam=out["cam"], light=out["light"], product=out["product"],
                has_second="B2" in out)
    if "B2" in out:
        b2 = out["B2"]
        empty2 = apply_measurement(b2, rng=rng, seed=seed + 2)
        prod2 = apply_measurement(out["I2"], rng=rng, seed=seed + 3)
        cv2.imwrite(os.path.join(out_dir, "empty2.png"), cv2.cvtColor(empty2, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "product2.png"), cv2.cvtColor(prod2, cv2.COLOR_RGB2BGR))
        arrs.update(B2=b2.astype(np.float32), I2=out["I2"].astype(np.float32),
                    beta2=out["beta2"].astype(np.float32))
    if perturb:
        # A deliberately mismatched capture pair for the diagnostics path:
        # the empty plate was shot with slightly different framing, exposure,
        # focus and noise than the product shot.
        emp_p = apply_measurement(out["B1"], gain=(1.035, 0.985, 1.012), shift=(1.3, -0.7),
                                  blur=0.45, noise=0.008,
                                  rng=np.random.default_rng(seed + 10), seed=seed + 10)
        prd_p = apply_measurement(out["I1"], gain=(1.0, 1.0, 1.0), shift=(0.0, 0.0),
                                  blur=0.0, noise=0.006,
                                  rng=np.random.default_rng(seed + 11), seed=seed + 11)
        cv2.imwrite(os.path.join(out_dir, "empty_perturbed.png"), cv2.cvtColor(emp_p, cv2.COLOR_RGB2BGR))
        cv2.imwrite(os.path.join(out_dir, "product_perturbed.png"), cv2.cvtColor(prd_p, cv2.COLOR_RGB2BGR))
        meta["perturbation"] = dict(empty_gain=[1.035, 0.985, 1.012], empty_shift=[1.3, -0.7],
                                    empty_blur=0.45, empty_noise=0.008, product_noise=0.006)
    np.savez_compressed(os.path.join(out_dir, "gt.npz"), **arrs)
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def build_all_samples(root, seeds=None, names=None):
    names = names or list(SAMPLE_SPECS)
    seeds = seeds or {n: i for i, n in enumerate(names)}
    made = []
    for n in names:
        out = make_scene_sample(n, seed=seeds.get(n, 0))
        save_sample(out, os.path.join(root, n), n, seed=seeds.get(n, 0))
        made.append(n)
    return made
