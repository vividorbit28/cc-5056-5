"""Run extraction on every sample and print layer + transfer metrics."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from shadow_extract.extract import extract
from shadow_extract.evaluate import evaluate
from shadow_extract.synth import SAMPLE_SPECS

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")


def load(name):
    d = os.path.join(ROOT, name)
    bgr = cv2.imread(os.path.join(d, "empty.png"))
    pgr = cv2.imread(os.path.join(d, "product.png"))
    gt = dict(np.load(os.path.join(d, "gt.npz")))
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), cv2.cvtColor(pgr, cv2.COLOR_BGR2RGB), gt


def drawn_support(gt, pad=8):
    """The bbox a user would draw around product + its visible shadow."""
    reg = (gt["alpha"] > 0.02) | (gt["t"] < 0.999)
    ys, xs = np.where(reg)
    m = np.zeros(reg.shape, np.uint8)
    cv2.rectangle(m, (xs.min() - pad, ys.min() - pad), (xs.max() + pad, ys.max() + pad), 1, -1)
    return m  # bool via parse_support accepting nonzero


def main(method="joint"):
    header = ("sample            t_MAE  t_bias t_deep  q_MAE  q_cos  IoU   "
              "tMAE_a recon  xfer  alt  leak  unexp")
    print(header)
    rows = []
    for name in SAMPLE_SPECS:
        empty, prod, gt = load(name)
        sup = drawn_support(gt)
        dec = extract(empty, prod, support=sup, method=method)
        L, T = evaluate(dec, gt, second_plate=True)
        unexp = int(dec.unexplained.sum())
        print(f"{name:<17} {L.t_mae:6.3f} {L.t_bias:+.3f} {L.t_deep_mae:6.3f} "
              f"{L.q_mae:6.4f} {L.q_cos:5.3f} {L.alpha_iou:5.3f} "
              f"{L.alpha_mae:6.3f} {L.recon_psnr:5.1f} {T.psnr:5.1f} "
              f"{T.psnr_alt:5.1f} {T.artifact_ratio:5.2f} {unexp:5d}")
        rows.append((name, L, T))
    return rows


if __name__ == "__main__":
    meth = sys.argv[1] if len(sys.argv) > 1 else "joint"
    main(meth)
