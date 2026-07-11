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
  pvat.py            # VAT/IVAT core: numba-JIT Prim MST, compute_vat/compute_ivat,
                     #   get_ivat_levels/get_ivat_hierarchy, IvatMeansResult, ClusterNode
  pqvat.py           # alternate fully-inlined numba Prim MST (vat_prim_mst_numba); not exported
  pcvat.pyx          # Cython/OpenMP VAT/IVAT: pairwise_distances_c, compute_vat_c,
                     #   compute_ivat_c, vat_prim_mst_c (f32 + f64 fused variants)
  fcm.py             # pure-numpy fuzzy_c_means reference implementation
  cfcm.pyx           # Cython/OpenMP fuzzy_c_means (f32 + f64 fused variants)
  fuzzycmeans.py     # FuzzyCMeans — sklearn-style class wrapper over FCM
  lk.py              # pure-numpy Lin-Kernighan TSP solver: lin_kernighan,
                     #   tour_length (reference / fallback path)
  clk.pyx            # Cython/OpenMP Lin-Kernighan (f32 + f64 fused variants) with
                     #   multi-threaded multi-start local optimization
  linkernighan.py    # LinKernighan — sklearn-style class wrapper over LK
  ivatmeans.py       # IVATMeans — sklearn-style class wrapper over IVAT
  util.py            # pairwise_distances (numba), synthetic cluster generators
tests/               # pytest suite (correctness + benchmark-marked perf tests)
benchmarks/          # dev-only scale/memory harness (NOT shipped in the wheel)
experiments/         # research spikes (NOT shipped): one <name>.py per experiment,
  figures/           #   generated PNG figures (run: python -m experiments.<name>)
  findings/          #   *_FINDINGS.md per experiment + the cross-cutting reports
                     #   (white-paper.md, performance-report.md, next-steps.md)
docs/                # perf guidance, bibliography, novelty write-ups
  papers/            #   committed prior-art PDFs
  sources/           #   git-ignored scratch cache for retrieved PDFs (do not commit)
```

Top-level markdown reports (`CODE_QUALITY.md`, `PROFILING_RESULTS.md`,
`PHASE2_REVERT_SUMMARY.md`) are living design/history docs — read them for
context before touching lint config or performance code.

The `experiments/` tree is a **research area, not shipped code** — spikes for
divide-and-conquer VAT, GPU/Borůvka MST, and the scaling/quality studies behind
the paper. Each `experiments/<name>.py` regenerates its own figures into
`experiments/figures/` and has a matching `experiments/findings/<NAME>_FINDINGS.md`.
Start from `experiments/findings/next-steps.md` (the roadmap/artifact index),
`white-paper.md` (claim + evidence), and `performance-report.md` (all numbers).

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
  `vat_prim_mst_seq`, `get_ivat_levels`, `get_ivat_hierarchy`
- **Result types:** `IvatMeansResult`, `ClusterNode`
- **FCM (functional):** `fuzzy_c_means`
- **Lin-Kernighan TSP (functional):** `lin_kernighan`, `tour_length`
- **sklearn-style classes:** `FuzzyCMeans`, `IVATMeans` (`.fit`, `.predict`,
  `.fit_predict`, `.labels_`, `.cluster_centers_`); `LinKernighan`
  (`.solve`, `.fit`, `.fit_predict`, `.tour_`, `.tour_length_`)
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

Install with the dev extra (also cythonizes and builds the extensions in-place):

```bash
pip install -e ".[dev]"
```

If you only need to (re)build the C extensions after editing a `.pyx`:

```bash
python setup.py build_ext --inplace     # or re-run: pip install -e .
```

Build note (`setup.py`): optimization flags are selected per-compiler at build
time — `/O2 /arch:AVX2 /fp:fast /openmp` on MSVC, `-O3 -march=native
-ffast-math -fopenmp` on unix. Don't hardcode GCC flags; MSVC silently ignores
them (that bug is why the switch exists).

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

- **`pr.yaml`** runs on PRs to `main`/`master`: black → flake8 → mypy → pytest.
- **`publish.yaml`** runs on `v*` tags: it rewrites the version in
  `pyproject.toml` from the tag, builds an sdist, and publishes to PyPI via
  trusted OIDC publishing. **To release: bump and push a `vX.Y.Z` tag** — do not
  hand-edit the published version in `pyproject.toml`.

## Working here

- Match the style of the code you touch (comment density, naming, docstrings).
- Keep the compiled and pure-Python paths in sync, and both f32/f64 variants.
- Keep `black`, `flake8`, `mypy`, and `pytest` green before pushing.
- Don't commit build artifacts (`.so`, generated `.c`) — they're git-ignored.
