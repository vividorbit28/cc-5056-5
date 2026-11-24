"""Command-line assistance for the contact-shadow workflow.

Scope is deliberately limited to contact-shadow operations (derive, move,
verify and export the contact shadow of an extraction).  Asset authoring and
layer migration are graphical/pipeline actions documented for the web UI; this
CLI only helps automate the contact part.
"""
from __future__ import annotations

import argparse
import json
import sys

import cv2
import numpy as np

from .asset import load_asset
from .color import srgb_to_linear
from .contact import (ContactShadowError, GroundTracker, composite_contact,
                      derive_contact, verify_contact)
from .extract import extract, parse_support
from .visualize import to_u8_srgb


def _read_rgb(path):
    bgr = cv2.imread(path)
    if bgr is None:
        raise SystemExit(f"cannot read image: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _support_from_gt(gt_path, pad=8):
    with np.load(gt_path) as z:
        reg = (z["alpha"] > 0.02) | (z["t"] < 0.999)
    ys, xs = np.where(reg)
    m = np.zeros(reg.shape, np.uint8)
    cv2.rectangle(m, (xs.min() - pad, ys.min() - pad),
                  (xs.max() + pad, ys.max() + pad), 1, -1)
    return parse_support(m, reg.shape)


def _cmd_derive(args):
    empty = _read_rgb(args.empty)
    product = _read_rgb(args.product)
    support = _support_from_gt(args.gt) if args.gt else None
    dec = extract(empty, product, support=support, method=args.method)
    contact = derive_contact(dec, occlusion_thresh=args.occlusion,
                             texture_leak_limit=args.leak_limit)
    out = contact.summary()
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


def _cmd_move(args):
    asset = load_asset(args.asset)
    t = asset["layers"]["t"]
    q = asset["layers"]["q"]
    alpha = asset["layers"]["alpha"]
    unclaimed = asset["layers"]["contact_unclaimed"].astype(bool)
    claimed = ((1.0 - t) >= args.occlusion) & ~unclaimed & (alpha < 0.5)
    contact = _contact_from_asset(asset, t, q, alpha, claimed, unclaimed, args)
    tracker = GroundTracker(contact)
    region = tracker.new_contact_region(args.dx, args.dy)
    print(json.dumps(region, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(region, f, indent=2)


def _cmd_verify(args):
    asset = load_asset(args.asset)
    t = asset["layers"]["t"]
    q = asset["layers"]["q"]
    alpha = asset["layers"]["alpha"]
    unclaimed = asset["layers"]["contact_unclaimed"].astype(bool)
    claimed = ((1.0 - t) >= args.occlusion) & ~unclaimed & (alpha < 0.5)
    contact = _contact_from_asset(asset, t, q, alpha, claimed, unclaimed, args)
    plate = srgb_to_linear(_read_rgb(args.plate).astype(np.float64) / 255.0)
    comp = composite_contact(plate, t, q, alpha=alpha, shadow_mask=claimed)
    check = verify_contact(comp, plate, contact, retention_min=args.retention_min,
                           leak_limit=args.leak_limit)
    print(json.dumps(check.as_dict(), indent=2))
    if not check.ok:
        raise SystemExit(2)


def _cmd_export_composite(args):
    asset = load_asset(args.asset)
    t = asset["layers"]["t"]
    q = asset["layers"]["q"]
    alpha = asset["layers"]["alpha"]
    unclaimed = asset["layers"]["contact_unclaimed"].astype(bool)
    claimed = ((1.0 - t) >= args.occlusion) & ~unclaimed & (alpha < 0.5)
    plate = srgb_to_linear(_read_rgb(args.plate).astype(np.float64) / 255.0)
    P = asset["layers"]["foreground_premult"] / np.maximum(alpha[..., None], 1e-6)
    P = np.where(alpha[..., None] > 1e-6, P, 0.0)
    comp = composite_contact(plate, t, q, alpha=alpha, P_lin=P,
                             shadow_mask=claimed, opacity=args.opacity,
                             bounce=args.bounce)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(to_u8_srgb(comp), cv2.COLOR_RGB2BGR))
    if not ok:
        raise SystemExit("png encode failed")
    with open(args.out, "wb") as f:
        f.write(buf.tobytes())
    print(json.dumps({"wrote": args.out, "claimed_px": int(claimed.sum())}, indent=2))


def _contact_from_asset(asset, t, q, alpha, claimed, unclaimed, args):
    from .contact import ContactShadow
    region = claimed | unclaimed
    footprint = alpha > 0.5
    dist = cv2.distanceTransform((~footprint).astype(np.uint8), cv2.DIST_L2, 5)
    band = (~footprint) & (dist <= args.max_distance)
    thresholds = {"occlusion_thresh": args.occlusion,
                  "texture_leak_limit": args.leak_limit,
                  "max_distance_px": float(args.max_distance),
                  "texture_leak": 0.0, "disconnected_px": int(unclaimed.sum())}
    return ContactShadow(t, q, alpha, claimed, unclaimed, region, footprint, band,
                         thresholds)


def build_parser():
    p = argparse.ArgumentParser(
        prog="shadow-contact",
        description="Contact-shadow assistance (derive / move / verify / export).")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("derive", help="derive the contact shadow of an extraction")
    d.add_argument("--empty", required=True)
    d.add_argument("--product", required=True)
    d.add_argument("--gt", help="optional gt.npz used to build the support region")
    d.add_argument("--method", default="joint", choices=("joint", "fit_only"))
    d.add_argument("--occlusion", type=float, default=0.04)
    d.add_argument("--leak-limit", type=float, default=0.35)
    d.add_argument("--out", help="optional JSON output path")
    d.set_defaults(func=_cmd_derive)

    m = sub.add_parser("move", help="hand over a new contact region after a move")
    m.add_argument("--asset", required=True, help="v2 asset bundle directory")
    m.add_argument("--dx", type=float, required=True)
    m.add_argument("--dy", type=float, required=True)
    m.add_argument("--occlusion", type=float, default=0.04)
    m.add_argument("--leak-limit", type=float, default=0.35)
    m.add_argument("--max-distance", type=float, default=60.0)
    m.add_argument("--out")
    m.set_defaults(func=_cmd_move)

    v = sub.add_parser("verify", help="verify independent shadow contacts the object")
    v.add_argument("--asset", required=True)
    v.add_argument("--plate", required=True)
    v.add_argument("--occlusion", type=float, default=0.04)
    v.add_argument("--leak-limit", type=float, default=0.35)
    v.add_argument("--retention-min", type=float, default=0.98)
    v.add_argument("--max-distance", type=float, default=60.0)
    v.set_defaults(func=_cmd_verify)

    e = sub.add_parser("export-composite", help="composite the contact shadow on a plate")
    e.add_argument("--asset", required=True)
    e.add_argument("--plate", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--occlusion", type=float, default=0.04)
    e.add_argument("--opacity", type=float, default=1.0)
    e.add_argument("--bounce", type=float, default=1.0)
    e.set_defaults(func=_cmd_export_composite)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except ContactShadowError as exc:
        print(json.dumps({"error": str(exc), "code": exc.code}), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
