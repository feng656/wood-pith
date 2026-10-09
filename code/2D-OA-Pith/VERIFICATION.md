# Verification record

Date: 2026-08-01

This repository is a research engineering core, not a claim of trained accuracy or
production readiness.  The following checks were completed in the delivery
environment:

- `python -m compileall -q src tests`: passed;
- full AST parsing: passed for 55 Python files (source plus tests);
- independent static mathematics review: near/far positive increments, far
  rationalized infinity limit, implicit-gradient chain, Schur block orientation,
  affine gauge lock and data-rank topology were checked;
- pure NumPy reference checks: out-of-frame coordinate round trip, far residual
  continuity at `rho=0`, implicit-function scale invariance, Schur/full-inverse
  identity, positive growth increments and tile-to-global uncertainty units passed;
- static engineering review: direct native-resolution source sampling, rotated
  crop bounds, checkpoint contracts, mode quotas, joint NULL decisions,
  stationarity rejection and tile second-moment variance blending were checked.

The runtime does **not** contain PyTorch, pytest or a CUDA stack.  Consequently the
provided PyTorch tests, backward/AMP execution, full CLI smoke test and GPU memory
profile could not be executed here.  CI installs the declared dependencies and is
configured to run the non-slow test suite; the slow synthetic optimizer test must
also be run before any release.

No real data were supplied in this task.  Therefore E0--E10, external baseline
reproduction, learned weights, calibration coverage, accuracy, throughput and
cross-domain claims remain experiments to perform, not results of this delivery.
The current documented gaps (ArcGraphNet training/top-K grouping, defect
supervision, explicit conformal-set inversion/rendering, strict group-equivariant
encoder and full-pipeline LOAO) are intentionally not represented as completed.

## Python-script conversion check (2026-08-07)

- `python -m compileall -q src scripts tests`: passed after adding the four
  programmatic experiment scripts and `oapith.workflows` facade;
- all 61 current Python files are included in the compile pass;
- direct-API regression tests for consensus and grouped conformal calibration were
  added in `tests/test_python_workflows.py`;
- this delivery runtime still has no PyTorch or pytest, so those new runtime tests
  and the pre-existing PyTorch suite could not be executed locally.  They must be
  run after `pip install -e ".[dev]"` in the target environment.
