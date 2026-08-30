# CLAUDE.md

Guidance for AI assistants (and humans) working in this repository.

## What this is

`tribble-clustering` is a Python package of **optimized VAT/IVAT visual
cluster-tendency assessment** and **fuzzy c-means (FCM) clustering**. The
headline features are a priority-queue MST speedup for VAT/IVAT (from NAFIPS
2025/2026 work) and compiled C/SIMD (Cython + OpenMP) extensions that add a
further ~15–20x on top.

- **PyPI name:** `tribble-clustering` — **import name:** `tribbleclustering`
- **Package root:** `src/tribbleclustering/` (a `src/` layout)
- **Python:** requires `>=3.11`
- **License:** MIT

## Layout

```
src/tribbleclustering/
  __init__.py        # public API surface (see __all__) — keep this authoritative
  clustering_base.py # BaseClusterer — shared ABC (.fit/.predict/.fit_predict) for
                     #   KMeans, FuzzyCMeans, IVATMeans, ConiVAT
  pvat.py            # VAT/IVAT core: numba-JIT Prim MST, compute_vat/compute_ivat
  pqvat.py           # alternate fully-inlined numba Prim MST (vat_prim_mst_numba); not exported
  pcvat.pyx          # Cython/OpenMP VAT/IVAT: pairwise_distances_c, compute_vat_c,
                     #   compute_ivat_c, vat_prim_mst_c (f32 + f64 fused variants)
  fcm.py             # pure-numpy fuzzy_c_means reference implementation
  cfcm.pyx           # Cython/OpenMP fuzzy_c_means + k-means (f32 + f64 fused variants)
  fuzzycmeans.py     # FuzzyCMeans — sklearn-style class wrapper over FCM
  kmeans.py          # KMeans — sklearn-style class wrapper over the Cython kernel
  nerfcm.py          # relational_fuzzy_c_means / relational_out_of_sample_membership —
                     #   NERFCM (Hathaway & Bezdek 1994) on a dissimilarity matrix; the
                     #   geometry-consistent IVATMeans(refine="relational") back end
  lk.py              # pure-numpy Lin-Kernighan TSP solver: lin_kernighan,
                     #   tour_length (reference / fallback path)
  clk.pyx            # Cython/OpenMP Lin-Kernighan (f32 + f64 fused variants) with
                     #   multi-threaded multi-start local optimization
  linkernighan.py    # LinKernighan — sklearn-style class wrapper over LK
  ivatmeans.py       # IVATMeans — sklearn-style class wrapper over IVAT;
                     #   get_ivat_levels/get_ivat_hierarchy, IvatMeansResult, ClusterNode
  conivat.py         # ConiVAT — constraint-based iVAT (Rathore, Bezdek, Santi & Ratti,
                     #   2020): semi-supervised iVAT from must-link/cannot-link constraints
  util.py            # pairwise_distances (numba), synthetic cluster generators
tests/               # pytest suite (correctness + benchmark-marked perf tests)
benchmarks/          # dev-only scale/memory harness (NOT shipped in the wheel)
docs/                # library docs only -- no research/novelty write-ups
  design-notes.md    #   why IVATMeans has refine= and dissimilarity_power=;
                     #   cited from the ivatmeans.py / nerfcm.py docstrings
  perf-guidance.md   #   optimization guidance for developers of this package
  papers/            #   committed prior-art PDFs cited from source docstrings
  sources/           #   git-ignored scratch cache for retrieved PDFs (do not commit)
```

Top-level markdown reports (`CODE_QUALITY.md`, `PROFILING_RESULTS.md`,
`PHASE2_REVERT_SUMMARY.md`) are living design/history docs — read them for
context before touching lint config or performance code.

There is no `experiments/` tree in this repo. It used to hold research spikes
(divide-and-conquer VAT, GPU/Borůvka MST, TSP studies) but was removed in PR #53
(2026-08-01) and now lives in the separate `grad-school` repo
(fundthmcalculus/grad-school#26) alongside coursework, making this repo a pure
library. Don't recreate `experiments/` here without discussing scope first —
research spikes belong in `grad-school`, not this package.

**This package is CPU-only.** The CuPy back ends (`gpu.py`, `gpu_vat.py`) and
the `[gpu]` extra were removed in issue #106: nothing in `src/` imported them,
no CI runner has a CUDA device so the device branches were never exercised, and
on a 12 GB consumer card with FP64 at ~1/64 of FP32 the payoff did not justify
the tiling machinery — on the dtype this library deliberately defaults to.
Don't reintroduce a GPU path without discussing it first. The prior work is
preserved in PRs #24 and #76 and the `perf/gpu-vat-frontend` branch.

The **novelty and prior-art write-ups followed them** in issue #97
(`bibliography.md`, `novel-niche.md`, `novelty-review.md`,
`performance-novelty.md`, `vat-tsp-prior-art.md`,
`vat-tsp-session2-novelty.md`, `popmusic-spacefilling.md`), and now live in
`grad-school` under `ClusteringExperiments/docs/`. They are thesis material,
not package documentation. What the library actually needs from them — the
geometry argument behind `refine=` and `dissimilarity_power=` — was extracted
into `docs/design-notes.md`, which stays. Add novelty/positioning prose to
`grad-school`, not here; `docs/` holds only what a user or contributor of the
package needs.

## Compiled-vs-pure-python fallback (important)

The Cython extensions (`pcvat`, `cfcm`, `clk`) are **optional at runtime**. The
sklearn-style wrappers try to import the compiled kernel and silently fall back
to the pure-Python/numba path if it isn't built:

```python
try:
    from .cfcm import fuzzy_c_means as fcm_algorithm   # fuzzycmeans.py
    _has_compiled_fcm = True
except ImportError:
    from .fcm import fuzzy_c_means as fcm_algorithm
    _has_compiled_fcm = False
```

`ivatmeans.py` does the same for `pcvat`. **Consequences to respect:**

- The compiled and pure paths must stay **behaviorally equivalent** — a change
  to one usually needs the matching change to the other, and tests should pass
  in both configurations.
- Tests that require the extension guard with
  `@pytest.mark.skipif(not CYTHON_AVAILABLE, ...)`. Follow that pattern rather
  than assuming the extension is present.
- The `.so`/`.c` build artifacts are git-ignored; the `.pyx` sources are
  tracked and `package-data` ships the `.pyx`/`.c` so wheels can rebuild.

## Public API

Import from the top-level package (defined in `__init__.py`):

- **VAT/IVAT (functional):** `compute_vat`, `compute_ivat`, `vat_prim_mst`,
  `get_ivat_levels`, `get_ivat_hierarchy`
- **Result types:** `IvatMeansResult`, `ClusterNode`
- **FCM (functional):** `fuzzy_c_means`
- **Relational FCM (functional):** `relational_fuzzy_c_means`,
  `relational_out_of_sample_membership` (NERFCM; operates on a dissimilarity
  matrix, no coordinates — see `IVATMeans(refine="relational")`)
- **Lin-Kernighan TSP (functional):** `lin_kernighan`, `tour_length`
- **ConiVAT (constraint-based iVAT):** `compute_conivat`, `ConiVAT`,
  `expand_constraints`, `generate_constraints_from_labels`, `learn_metric`,
  `transform_with_metric`
- **Base class:** `BaseClusterer` — shared `.fit`/`.predict`/`.fit_predict`
  interface for all sklearn-style classes below
- **sklearn-style classes:** `FuzzyCMeans`, `KMeans`, `IVATMeans` (`.fit`,
  `.predict`, `.fit_predict`, `.labels_`, `.cluster_centers_`;
  `refine="medoid"` (default), `"relational"`, or `"euclidean"` — see the
  class docstring and issue #54); `LinKernighan` (`.solve`, `.fit`,
  `.fit_predict`, `.tour_`, `.tour_length_`); `ConiVAT`
- **Helpers:** `pairwise_distances`

When you add or rename anything user-facing, update `__all__` in `__init__.py` —
it is the source of truth for the API and drives the re-export lint exemption.

## Conventions

- **Distance matrices** passed to VAT/IVAT must be symmetric, PSD dissimilarity
  matrices (typically an L2 pairwise-distance matrix).
- **`inplace=` semantics:** `compute_vat`/`compute_ivat` accept `inplace=False`
  by default; the in-place path exists to avoid holding multiple `n x n` buffers
  (IVAT holds up to 3 — the documented memory wall). Preserve this when editing.
- **Fused types (f32/f64):** the `.pyx` kernels are written once per dtype
  (`_64`/`_32` variants) with a Python dispatcher (`*_c`) picking by
  `data.dtype`. `float32` roughly halves time and memory at a documented
  accuracy cost and is **opt-in, not default**. Mirror both variants on changes.
- **numba JIT:** hot kernels use `@njit(cache=True, ...)`; Prim's round loop is
  inherently serial — only the O(n²) global-max scan and permutation gather
  parallelize. Don't "parallelize" the serial dependency.
- **Typing:** the codebase uses type hints throughout; the sklearn wrappers use
  numpy-style docstrings. Match the surrounding style.

## Development workflow

This project uses [uv](https://docs.astral.sh/uv/) for dependency management and builds.

### Setup

Install the development environment with all dependencies and build the extensions:

```bash
uv sync --extra dev
```

This creates a `.venv` directory with the project and all development tools (numpy, cython,
pytest, black, flake8, mypy, etc.), and builds the compiled Cython extensions.

### Rebuilding extensions

After editing a `.pyx` file, rebuild the extensions:

```bash
uv pip install -e . --no-deps
```

Or using setuptools directly:

```bash
python setup.py build_ext --inplace
```

**Build note:** optimization flags are selected per-compiler at build time — `/O2 /arch:AVX2 /fp:fast /openmp`
on MSVC, `-O3 -march=native -ffast-math -fopenmp` on unix. Don't hardcode GCC flags; MSVC silently ignores
them (that bug is why the switch exists).

### Alternative: pip (without uv)

If you prefer pip/venv, the project still works with the traditional approach:

```bash
pip install -e ".[dev]"
```

### Quality gates (all enforced in CI — `.github/workflows/pr.yaml`)

Run all four before pushing; CI runs exactly these:

```bash
black --check .          # formatting (line length 88, target py311)
flake8 src tests         # lint (max-line-length 120, E203 ignored)
mypy src                 # types (lenient baseline: ignore_missing_imports)
pytest                   # correctness tests
```

Config for all of these lives in `pyproject.toml`. See `CODE_QUALITY.md` for the
rationale (notably the intentional black-88 / flake8-120 split, and the plan to
ratchet mypy strictness one module at a time).

### Tests

- `pytest` runs correctness tests. Benchmark tests use wall-clock timing
  assertions that are unreliable on shared CI, so they are marked `benchmark`
  and **deselected by default** (`addopts = "-m 'not benchmark'"`).
- Run perf benchmarks explicitly: `pytest -m benchmark`.
- **CI-fast mode** (`tests/conftest.py`) trims the suite on shared runners. It
  is auto-enabled on GitHub Actions (`GITHUB_ACTIONS`) / generic CI (`CI`), or
  forced anywhere with `pytest --ci-fast`. In this mode: tests marked
  `@pytest.mark.ci_slow` (scaling/plotting benchmarks with no correctness
  assertion) are skipped, and the `ci_scale(full, fast)` fixture returns the
  smaller size so heavy correctness tests run on reduced inputs. Everything
  runs full-size locally. This cuts the default CI test step from ~50s to ~5s;
  run the complete suite locally with a plain `pytest` (no flag, outside CI).
- Tests requiring the compiled extension guard with
  `@pytest.mark.skipif(not CYTHON_AVAILABLE, ...)`.
- Some tests/demos pull real datasets via `ucimlrepo` and can allocate tens of
  GB — check the size comments in `tests/demo_data.py` before running them.

### Benchmarks (`benchmarks/`)

Dev-only harness (not in the wheel) for measuring wall-clock and **peak resident
memory** vs. dataset size `n`. Each `(n, dtype, stage)` measurement runs in a
fresh subprocess for trustworthy peak-RSS numbers (see `benchmarks/README.md`).

```bash
python -m benchmarks.scale_bench --quick                       # sanity check
python -m benchmarks.scale_bench --sizes 16000 32000 --max-gb 55
```

Baselines are written to `benchmarks/baselines/<tag>_<host>.json`. The core
finding driving the roadmap: **memory (IVAT's 3-matrix footprint), not compute,
is the scaling wall.**

## Release / CI

- **`pr.yaml`** runs on PRs to `main`/`master`: black → flake8 → mypy → pytest,
  on Python 3.11 with `--ci-fast`. This is the gate; keep it fast.
- **`nightly.yaml`** runs at 04:00 UTC (and on `workflow_dispatch`) and covers
  the three things `pr.yaml` structurally cannot: the full-size correctness
  suite (no `--ci-fast` trimming), the `benchmark`-marked wall-clock
  assertions, and a **Python 3.11/3.12/3.13/3.14 matrix** for the range
  `requires-python` promises. A failure opens or comments on one accumulating
  `nightly-failure` issue; that reporting lives in a separate `report` job so
  four matrix legs can't race the dedupe. Don't move a check here that belongs
  in the gate, and don't put a slow check in the gate that belongs here.
- **`publish.yaml`** runs on `v*` tags: it rewrites the version in
  `pyproject.toml` from the tag, builds an sdist, and publishes to PyPI via
  trusted OIDC publishing. **To release: bump and push a `vX.Y.Z` tag** — do not
  hand-edit the published version in `pyproject.toml`.

## Working here

- Match the style of the code you touch (comment density, naming, docstrings).
- Keep the compiled and pure-Python paths in sync, and both f32/f64 variants.
- Keep `black`, `flake8`, `mypy`, and `pytest` green before pushing.
- Don't commit build artifacts (`.so`, generated `.c`) — they're git-ignored.
- **Never commit directly to `main`.** Always create a feature branch and open
  a pull request for review, even for small fixes/docs — this applies to AI
  agents as much as humans. `pr.yaml` only runs CI on PRs into `main`/`master`,
  so a direct commit to `main` also skips the quality gates.
