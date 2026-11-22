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
