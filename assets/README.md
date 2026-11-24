# Shadow asset bundles (layer format v2)

This directory holds the converted shadow/foreground assets.  The layer data
under `<name>/` is generated and content-hashed; it is intentionally not
committed.  The committed artefacts are:

- `build-lock.json` — pins every migrated bundle: manifest hash, per-layer
  hash, premultiplication flag, lifecycle status, and the sha256 of every
  source image/intermediate, plus the git revision of the converting code.

## Rebuild in a clean checkout

```bash
python3 scripts/make_legacy_fixtures.py --sample soft_paper   # legacy v1 inputs
python3 scripts/build_assets.py                                # migrate + lock
```

The build re-hashes every output and fails on drift.  Each bundle contains:

- `shadow_asset.json` — manifest: format `shadow-asset` version `2.0`, the
  plane coordinate frame (`plane`), the source revision block
  (`source.git_revision` + per-file sha256), and per-layer premultiplied
  flags/statuses;
- `layer_*.npy` — `t` (achromatic transmission, straight), `q` (additive
  bounce, straight), `alpha` (matte, straight), `foreground_premult`
  (premultiplied `alpha*P`, flag `premultiplied: true`), and
  `contact_unclaimed` (shadow pixels returned to the author);
- `contact.json` — the contact-shadow summary (claimed/unclaimed counts,
  thresholds, plane origin/scale);
- `legacy/` — for converted v1 files, the original arrays and the chosen
  conversion path (`reversible:*` or `underdetermined:*`).

## Lifecycle statuses

- `confirmed` — physically derived and usable in compositing;
- `needs_author_claim` — shadow mixed with receiver texture, or a baked map
  without its source plate; held for an author decision;
- `pending_explanation` — a merged-luminance layer whose `(t, q)` colour split
  is underdetermined; the transmission slot stays fully open until explained.

A legacy shadow colour or display-space darkening ratio is never written into
the transmission slot `t`; only a linear-light achromatic transmission from the
reversible path is.
