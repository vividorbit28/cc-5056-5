"""End-to-end demonstration: reconstruction, re-background and second-plate transfer.

Writes samples/_demo.png, one row per sample:
  capture | reconstruction | re-background (new plate) | second-plate transfer | |unexplained|
The re-background row is what the deliverable claims: the same extracted layers,
re-lit on a plate the estimator never saw.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from shadow_extract.color import luminance, srgb_to_linear
from shadow_extract.evaluate import evaluate_transfer, texture_leak
from shadow_extract.extract import extract, parse_support
from shadow_extract.plates import make_plate
from shadow_extract.synth import SAMPLE_SPECS
from shadow_extract.visualize import contact_sheet, scalar_map, to_u8_srgb

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def _load(name):
    d = os.path.join(ROOT, name)
    e = cv2.cvtColor(cv2.imread(os.path.join(d, "empty.png")), cv2.COLOR_BGR2RGB)
    p = cv2.cvtColor(cv2.imread(os.path.join(d, "product.png")), cv2.COLOR_BGR2RGB)
    gt = dict(np.load(os.path.join(d, "gt.npz")))
    return e, p, gt


def _support(gt, pad=8):
    reg = (gt["alpha"] > 0.02) | (gt["t"] < 0.999)
    ys, xs = np.where(reg)
    m = np.zeros(reg.shape, np.uint8)
    cv2.rectangle(m, (xs.min() - pad, ys.min() - pad), (xs.max() + pad, ys.max() + pad), 1, -1)
    return parse_support(m, reg.shape)


def _u8(x):
    return to_u8_srgb(np.clip(x, 0, 1))


def main(method="joint"):
    tiles = []
    for i, name in enumerate(SAMPLE_SPECS):
        e, p, gt = _load(name)
        dec = extract(e, p, support=_support(gt), method=method)
        h, w = gt["alpha"].shape

        recon = _u8(dec.recon())

        # New plate never seen by the estimator, in a warm paper tone.
        rng = np.random.default_rng(100 + i)
        beta_new = make_plate((h, w), rng, kind="paper",
                              base_rgb=(0.70, 0.62, 0.50), contrast=0.28)
        B_new = beta_new  # E0 = 1 for a synthetic plate of this scale
        ground_new = B_new * (dec.t[..., None] + dec.q)
        comp_new = _u8(dec.alpha[..., None] * dec.P_lin
                       + (1 - dec.alpha[..., None]) * ground_new)

        # Transfer onto the sample's independently captured second plate.
        B2 = gt["B2"]
        pred2 = dec.transfer(B2)
        pred2_full = dec.alpha[..., None] * gt["P"] + (1 - dec.alpha[..., None]) * pred2
        T = evaluate_transfer(dec, gt)
        transfer = _u8(pred2_full)
        gt2 = _u8(gt["I2"])

        u = np.abs(luminance(dec.residual))
        u = u * dec.unexplained
        unexp = scalar_map(u, 0.0, max(float(np.percentile(u, 99.5)), 1e-4), cv2.COLORMAP_MAGMA)

        tiles.append((f"{name} · capture", p))
        tiles.append((f"reconstruction  α·P+(1−α)(t+q)B", recon))
        tiles.append(("re-background · unseen plate β_new", comp_new))
        tiles.append(("second-plate truth I₂", gt2))
        tiles.append((f"second-plate transfer  {T.psnr:.1f}dB", transfer))
        tiles.append(("unexplained (not shadow)", unexp))

    sheet = contact_sheet(tiles, cols=6, cell_scale=0.5)
    out = os.path.join(ROOT, "_demo.png")
    cv2.imwrite(out, cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("wrote", out, sheet.shape)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "joint")
