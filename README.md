# Landing-shadow extractor

From a fixed-camera **empty plate** and **product** capture, separate the product
foreground from its contact/cast shadow so the product can be re-composited onto
a new background with an editable landing shadow.

The shadow is not treated as a black transparent overlay. In linear light it is

```
B      = beta * E0            empty plate radiance
I_g    = t * B + A            ground under the product
q      = A / B                additive bounce per unit plate radiance
I      = alpha * P + (1-alpha) * I_g
```

i.e. a **multiplicative occlusion** `t ∈ [t_min, 1]` plus **added light** `q·B`.
Ratios are computed in linear light, clipped to a reliable band (plate above the
noise floor) so `I/B` never blows up as `B → 0`. Transfer to a new plate re-uses
the *same* `t` and `q`: `I2 = B2·(t + q)`, which is why `q` must carry the
product's bounce colour, not the original plate's texture.

## Run

```bash
python3 -m shadow_extract.server          # UI at http://127.0.0.1:7860
```

Rebuild the generated samples (re-renders all 8 with ground truth + second plate):

```bash
python3 scripts/build_samples.py
```

Run the evaluation table (layer accuracy + transfer to a second plate):

```bash
python3 scripts/eval_all.py joint         # or: fit_only
```

Tests:

```bash
python3 -m pytest tests/ -q
```

## UI

1. Load a built-in sample or upload your own empty/product pair.
2. Click points around the product **and its shadow**; Enter / double-click to
   close the support region. Outside it the plate is untouched.
3. **Extract.** The panels show, separately:
   - **foreground** `α·P` over a checkerboard (judge the matte, no shadow),
   - **occlusion** `1−t` (multiplicative shadow strength),
   - **bounce** `|q|` (additive shadow colour, ground only),
   - **unexplained** residual that neither layer explains.
4. **Capture diagnosis** reports registration shift, exposure gain, blur, noise
   and texture agreement, with a gate: if the background itself changed, the
   tool says so instead of calling the change a shadow.
5. **Parametric re-background**: pick a generated plate or solid colour and scale
   shadow opacity (occlusion) and bounce (added light) independently.
6. **Verify on second plate** applies the same `t`, `q` to an independently
   captured second plate and reports PSNR/SSIM against the real capture — the
   check that the layers are physically transferable, not just able to repaint
   the original.

## Two estimators

- **joint** (default): colour-identified. Per ground pixel `y = I/B = g·(t + s·d)`
  with achromatic `t`, a global bounce direction `d`, and per-pixel strength `s`.
  The colour split makes `t` and `q` identifiable instead of leaving it to a
  smoothness prior. Best `q` accuracy.
- **fit_only**: locks the known base surface from the *lit periphery* of the
  support (the physical premise: outer ground is fully lit), fits `t`, then a
  bounded residual `q`. Competitive on `t`, slightly worse on `q`.

## Guarantees enforced

- `t ∈ [t_min, 1]`; `q` bounded and `≤ 1−t` so the shadow never brightens.
- Support region: no shadow is claimed outside it.
- Anti-black-halo: matte from residual, product-colour fill at fringes.
- Background texture is never absorbed into `t` (smooth, ratio-weighted, and
  checked by the transfer texture-leak metric).

## Asset format (layer file upgrade v1 → v2.0)

Shadow/foreground intermediates are stored as versioned **shadow-asset**
bundles (see `shadow_extract/asset.py`).  Everything old intermediates left
implicit is now declared explicitly and verified on load:

- **premultiplied state** — every layer carries an explicit boolean
  `premultiplied`; the foreground colour is stored premultiplied (`alpha·P`).
  A missing flag or a content-hash mismatch raises an error instead of
  defaulting;
- **plane coordinates** — a `plane` block declares frame, origin, pixel scale
  and the camera projection, so a moved product is placed geometrically;
- **source revision** — the git revision of the converting code and the
  sha256 of every source image/intermediate.

### Migrating old layer files

`shadow_extract/legacy.py` detects three v1 shapes and offers two paths:

| old shape | reversible path | underdetermined path |
| --- | --- | --- |
| full `t, q, alpha` | lossless v2 migration | — |
| baked darkening + source plate | linear achromatic split (`q=0`, original map kept) | without the plate: layer kept, status `needs_author_claim` |
| merged luminance only | achromatic `t` only when a plate is present | colour split impossible: `t` stays open, status `pending_explanation`, single channel preserved separately |

The underdetermined layer re-opens for continued editing
(`reopen_underdetermined_edit` / `save_underdetermined_edit`) and re-locks its
hashes.  A legacy shadow colour is **never** treated as a transmission
coefficient.

```bash
python3 scripts/make_legacy_fixtures.py --sample soft_paper
python3 scripts/build_assets.py     # writes assets/ + assets/build-lock.json
```

The build lock pins every output hash; re-running it in a clean directory
recomputes the hashes and fails on drift rather than hiding it.

## Contact shadow (new path)

`shadow_extract/contact.py` derives the landing shadow that belongs to the
product: a smooth, strongly-occluding band touching the footprint.  Shadow
pixels mixed with floor texture, or disconnected from the footprint, are
**returned to the author** (`contact_unclaimed`) instead of being moved.

When the artist moves the product, the `GroundTracker` hands over a new
contact region: the claimed shadow translates with the product while the
unclaimed part stays with the floor.  Contact verification is an actual
computation (rim coverage, gap in pixels, measured shadow contribution
`1 − mean(I/B)`, and foreground retention); failures are surfaced through the
contact-shadow interface, never substituted with a painted ellipse.

### Graphical entry points (web UI, section 6)

- **Derive contact shadow** — claimed vs returned-to-author overlay;
- **Move product ◀ / ▶** — ground-tracker hand-over with the new region bbox;
- **Edit region & verify contact** — polygon edit, composite, then verify
  independent shadow/object contact and foreground retention;
- **Export two floor textures** — two different floors, with a
  texture-carry diagnostic that reveals whether the original tile was moved;
- **Convert to v2 asset** — write the manifest/layers/contact bundle.

The composition canvas always renders through the contact-shadow
implementation.  The CLI only *assists* these same contact operations:

```bash
python3 -m shadow_extract.cli derive --empty e.png --product p.png --gt gt.npz
python3 -m shadow_extract.cli move --asset assets/soft_paper --dx 12 --dy 0
python3 -m shadow_extract.cli verify --asset assets/soft_paper --plate e2.png
python3 -m shadow_extract.cli export-composite --asset assets/soft_paper \
        --plate floor.png --out comp.png
```
