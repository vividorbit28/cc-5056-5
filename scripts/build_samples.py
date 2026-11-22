"""Build all synthetic samples and a diagnostic contact sheet."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np

from shadow_extract.synth import SAMPLE_SPECS, make_scene_sample, save_sample
from shadow_extract.visualize import contact_sheet, to_u8_srgb, scalar_map


def sample_tiles(path):
    gt = np.load(os.path.join(path, "gt.npz"))
    empty = cv2.cvtColor(cv2.imread(os.path.join(path, "empty.png")), cv2.COLOR_BGR2RGB)
    prod = cv2.cvtColor(cv2.imread(os.path.join(path, "product.png")), cv2.COLOR_BGR2RGB)
    t, q = gt["t"], gt["q"]
    tvis = scalar_map(1.0 - t, 0.0, 1.0, cv2.COLORMAP_INFERNO)
    qvis = scalar_map(q.mean(axis=2), 0.0, max(float(np.percentile(q, 99.5)), 0.01), cv2.COLORMAP_TURBO)
    a = gt["alpha"]
    avis = scalar_map(a, 0.0, 1.0, cv2.COLORMAP_VIRIDIS)
    return [("empty plate", empty), ("product", prod),
            ("1 - t (occlusion)", tvis), ("q = A/B", qvis), ("alpha", avis)]


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "samples")
    names = list(SAMPLE_SPECS)
    for i, n in enumerate(names):
        out = make_scene_sample(n, seed=i)
        save_sample(out, os.path.join(root, n), n, seed=i)
        print("built", n, flush=True)
    tiles = []
    for n in names:
        for lab, im in sample_tiles(os.path.join(root, n))[:5]:
            tiles.append((f"{n} | {lab}", im))
    sheet = contact_sheet(tiles, cols=5, cell_scale=0.5)
    cv2.imwrite(os.path.join(root, "_contact_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
    print("sheet", sheet.shape)
