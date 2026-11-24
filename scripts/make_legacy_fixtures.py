"""Generate legacy v1 layer fixtures under ``samples/legacy_v1``.

The fixtures represent the three old intermediate shapes the migrator must
handle.  They are synthesised from the existing sample ground truth, so the
legacy path can be exercised without keeping any hand-edited binary in the
tree.  Files use the ``<sample>__<kind>`` naming expected by build_assets.py.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "samples", "legacy_v1")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="soft_paper")
    args = parser.parse_args(argv)
    os.makedirs(OUT, exist_ok=True)
    gt = dict(np.load(os.path.join(ROOT, "samples", args.sample, "gt.npz")))
    t, q, alpha, P, B = gt["t"], gt["q"], gt["alpha"], gt["P"], gt["B1"]

    # Baked darkening map with its source plate: reversible path.
    np.savez(os.path.join(OUT, f"{args.sample}__baked.npz"),
             scale=t, plate_lin=B, alpha=alpha, P=P)
    # Merged luminance only: underdetermined (no plate, no colour split).
    np.savez(os.path.join(OUT, f"{args.sample}__merged.npz"),
             luma=t, alpha=alpha, P=P)
    # Full decomposition in the v1 npz layout.
    np.savez(os.path.join(OUT, f"{args.sample}__full.npz"),
             t=t, q=q, alpha=alpha, P=P)
    print("wrote legacy v1 fixtures for", args.sample, "to", OUT)


if __name__ == "__main__":
    main()
