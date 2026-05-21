# archive/

Files in this directory are no longer imported by any live code in this repo
but are kept for research-history value. Do not import from this directory.

## Contents

### `quadratic_estimator.py` (2,156 lines)

Phase 2 QE prototype (Hotinli & Johnson, "Reconstructing large scales at
cosmic dawn"). Substantially orphaned at archival time:

- One external caller (a 4-cell stub notebook at
  `noisy_reconstruction/notebooks/quadratic/01_noisy_pipeline_smoketest.ipynb`,
  itself archived in the same cleanup pass).
- Its `__main__` block calls `run_paper_consistent_analysis(redshifts)` —
  function not defined in the file. So `python quadratic_estimator.py`
  raises `NameError`. Left as-is in the archived copy; not worth fixing
  dead code.
- Ten internal `quadratic_estimator_*` variants (v1–v4 + flatsky + diagonal +
  velocity/v2 + proxy + linear) are an exploratory playground that the
  published paper does not depend on. Paper code lives in
  `functions/generate_all_plots.py`.

Retained as a reference if the QE direction is revived (e.g. for Phase 2
noise-regime comparisons against the linear continuity method).

Audit reference: `claude-workspace/outputs/2026-05-19-redundancy-audit.md` §3.1.

### `ksz_pipeline.py` (699 lines)

Standalone class-based kSZ reconstruction + cross-correlation pipeline,
rescued from the (now-removed) `claude/jovial-pascal` worktree. Was never
committed to `main`. Defines:

- `KSZPipeline` class with chunk-based redshift processing
- `ChunkResult` / `PipelineResult` dataclasses
- `compute_ksz_map(vz, xhi, density, axis=2)` — distinct signature from
  `functions/generate_all_plots.py:compute_ksz_maps(vz, xhi, den, z=None,
  physical_norm=False, use_optical_depth=False)`
- `run_from_files(...)` entry point for file-based runs

Genuinely different architecture from the script-style `generate_all_plots.py`.
Kept as a reference if a class-based refactor is revived.
