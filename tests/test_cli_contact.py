"""CLI scope tests: the command line only assists contact-shadow operations."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import pytest

from shadow_extract.asset import load_asset
from shadow_extract.cli import build_parser, main
from shadow_extract.migrate import migrate_gt_npz


def _sample(tmp_path, H=70, W=90):
    sdir = tmp_path / "s"
    sdir.mkdir()
    alpha = np.zeros((H, W))
    alpha[25:45, 35:55] = 1.0
    yy, xx = np.mgrid[0:H, 0:W]
    d = np.maximum(np.abs(xx - 45) - 10, 0) + np.maximum(np.abs(yy - 35) - 10, 0)
    t = 1.0 - np.clip(0.55 * np.exp(-d / 6.0), 0, 1)
    q = np.zeros((H, W, 3))
    P = np.full((H, W, 3), 0.7)
    np.savez(str(sdir / "gt.npz"), t=t, q=q, alpha=alpha, P=P)
    cv2.imwrite(str(sdir / "empty.png"), np.full((H, W, 3), 128, np.uint8))
    cv2.imwrite(str(sdir / "product.png"), np.full((H, W, 3), 110, np.uint8))
    bundle = tmp_path / "asset"
    migrate_gt_npz(str(sdir / "gt.npz"), str(bundle), name="s",
                   source_files={"empty": str(sdir / "empty.png")})
    return sdir, bundle


def test_parser_exposes_only_contact_commands():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {"derive", "move", "verify", "export-composite"}


def test_cli_move_and_export(tmp_path):
    sdir, bundle = _sample(tmp_path)
    out_json = tmp_path / "region.json"
    main(["move", "--asset", str(bundle), "--dx", "8", "--dy", "0",
          "--out", str(out_json)])
    region = json.loads(out_json.read_text())
    assert region["n_pixels"] > 0
    out_png = tmp_path / "comp.png"
    main(["export-composite", "--asset", str(bundle),
          "--plate", str(sdir / "empty.png"), "--out", str(out_png)])
    img = cv2.imread(str(out_png))
    assert img is not None and img.ndim == 3


def test_cli_verify_exit_code_on_broken_contact(tmp_path):
    sdir, bundle = _sample(tmp_path)
    # Raise the occlusion threshold above the shadow's strength so the claimed
    # region is empty and verification reports a failed contact (non-zero exit).
    asset = load_asset(str(bundle))
    peak = float((1.0 - asset["layers"]["t"]).max())
    with pytest.raises(SystemExit) as exc:
        main(["verify", "--asset", str(bundle),
              "--plate", str(sdir / "empty.png"),
              "--occlusion", str(peak + 0.1)])
    assert exc.value.code == 2
