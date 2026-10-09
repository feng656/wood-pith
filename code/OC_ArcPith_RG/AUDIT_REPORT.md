# ArcPith-GT v4 implementation audit

## Input decision

The two PNG directories are derived products, not the primary algorithm input
specified by v4. `dataset` contains 2,371 1000x1000 B/D mask PNGs and
`dataset_grid` contains 1,393 800x800 grid PNGs. They do not carry the required
parent-ring/fragment lineage and curve coordinates. The package manifest is
the suitable starting point: `data/manifest.jsonl` has 5,182 crops, 102
sections, 4 tree IDs, ordered curves, crop origins, full-section pith GT and
outside-crop cases.

The manifest has no `mm_per_pixel` values. Stage 0 therefore freezes the run in
`PIXEL_ONLY` units and never fabricates physical millimetres. It also derives a
rectangular crop polygon and truncation flags from `image_size`,
`crop_origin_px`, `pith_full_px` and `pith_px`.

## Findings

1. **Stage 0 contract was missing.** Existing ingestion hashes rows but did not
   validate image paths, crop transforms, truncation, target-domain status or
   the full v4 hard invariants.
2. **Stage 1 was incomplete.** `preprocess.py` fits a cubic spline, but has no
   frozen smoothing selection, endpoint guard, measurement covariance,
   tangent-stability quantity, or explicit uncertainty provenance. Its
   uncertainty is not propagated into residuals.
3. **Stage 2 was incomplete.** `preflight.py` provides only a small candidate-
   free descriptor set. `candidates.py` is GT-dependent and does not implement
   the v4 CS2 continuous quadratic intersection, CS3 parent-group consensus,
   CS4 whole-arc circle gate, CS5 inverse-range ladder, or CS6 deterministic
   dual-chart mesh.
4. **The existing optimizer is not v4 Stage 3.** It uses a bounded
   `L-BFGS-B` far chart with a fixed `kappa_max`, no dual-chart seam replay,
   no search certificate, and a Student-t kernel rather than the frozen
   pseudo-Huber objective specified by v4.
5. **Physical units are not enforced.** `Sample.mm_per_pixel` is optional and
   existing metrics silently operate in normalized pixels.
6. **The package validation baseline was incomplete.** `validate_package.py`
   reported a missing `AUDIT_REPORT.md`; this report now supplies that missing
   artifact. The package's 20 unit tests pass when run with the package root on
   `PYTHONPATH`.

## Implemented in this pass

- `stages/stage0_contract.py`: normalized manifest, hashes, hard-field checks,
  target/truncation/scale status and failure ledger.
- `stages/stage1_continuous.py`: cubic B-spline arcs, arc-length nodes,
  tangent-stability fallback, endpoint guard, covariance proxy and fixed
  parent-ring weights.
- `stages/stage2_preflight_coldstart.py`: candidate-free descriptors plus all
  six cold-start families in an RP2-compatible registry, with deterministic
  deduplication.

These scripts write independently inspectable products under
`代码/最终产物/stage 0`, `stage 1` and `stage 2`. No script in this pass
selects a final pith coordinate or uses GT to choose a seed.
