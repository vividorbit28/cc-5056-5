"""Build the v2 shadow assets from the current sample intermediates.

Runs entirely from declared inputs (``samples/*``) so it is reproducible in a
clean checkout: every migrated bundle is content-hashed and the resulting
``assets/build-lock.json`` pins every manifest, layer, premultiplication flag,
status and source-file hash.  Legacy v1 layer files, when present under
``samples/legacy_v1/``, exercise both the reversible and the underdetermined
migration paths.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shadow_extract.migrate import migrate_sample_tree, verify_build_lock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES = os.path.join(ROOT, "samples")
ASSETS = os.path.join(ROOT, "assets")
LEGACY_DIR = os.path.join(SAMPLES, "legacy_v1")


def _legacy_files():
    if not os.path.isdir(LEGACY_DIR):
        return {}
    out = {}
    for fn in sorted(os.listdir(LEGACY_DIR)):
        if not (fn.endswith(".npz") or fn.endswith(".json")):
            continue
        sample = fn.split("__", 1)[0]
        out.setdefault(sample, []).append(os.path.join(LEGACY_DIR, fn))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", default=SAMPLES)
    parser.add_argument("--assets", default=ASSETS)
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args(argv)
    result = migrate_sample_tree(args.samples, args.assets,
                                 legacy_files=_legacy_files(),
                                 verify=not args.no_verify)
    print(json.dumps({"assets_root": result["assets_root"],
                      "lock": result["lock"], "n_assets": result["n_assets"],
                      "legacy_reports": [
                          {"source": os.path.basename(r["source"]),
                           "paths": r["paths"], "warnings": r["warnings"]}
                          for r in result["legacy_reports"]]}, indent=2))
    ok, mismatches = verify_build_lock(args.assets)
    if not ok:
        print("BUILD LOCK DRIFT:", mismatches, file=sys.stderr)
        raise SystemExit(1)
    print("build lock verified:", ok)


if __name__ == "__main__":
    main()
