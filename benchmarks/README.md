# Scale benchmarks

Development harness (not shipped in the wheel) for measuring the VAT/IVAT
pipeline's wall-clock time and **peak resident memory** as a function of
dataset size `n`, so scaling optimizations can be compared against a fixed
baseline.

## Running

```bash
# full sweep, skip any stage that would exceed 55 GB
python -m benchmarks.scale_bench --sizes 4000 16000 32000 48000 64000 --max-gb 55

# tiny sanity check
python -m benchmarks.scale_bench --quick

# one dtype, custom dimension
python -m benchmarks.scale_bench --sizes 50000 --dtypes f64 --d 20 --max-gb 55
```

Results are written to `benchmarks/baselines/<tag>_<host>.json`.

## Why subprocess isolation

Each `(n, dtype, stage)` measurement runs in a **fresh subprocess**. This is
required for trustworthy peak-memory numbers on Windows:

- `PeakWorkingSetSize` is a *process-lifetime* high-water mark with no reset
  API, and
- Windows does not promptly return freed pages to the OS,

so sequential large allocations in a single process contaminate each other's
peak (an early version reported monotonically non-decreasing memory — every
stage inherited the largest prior allocation). A clean process per measurement
gives a true, isolated peak. `memprobe.measure_peak_rss()` combines the OS
high-water mark with a 5 ms sampling thread and reports the larger.

## Baseline (reference: 32-core Intel, 64 GB, MSVC/AVX2, `d=10`)

Peak RSS is reported as an absolute value; the pipeline holds this many
simultaneous `n x n` matrices per stage:

| stage | matrices live | why |
|-------|---------------|-----|
| pairwise | 1 | the output distance matrix |
| VAT | 2 | input `D` + permuted output |
| **IVAT** | **3** | input `D` + VAT + IVAT — the memory wall |

| n | dtype | 1 matrix | pairwise | VAT | IVAT |
|---|-------|----------|----------|-----|------|
| 16000 | f32 | 1.0 GB | 293 ms / 1.0 GB | 498 ms / 2.1 GB | 1026 ms / 3.1 GB |
| 16000 | f64 | 2.0 GB | 427 ms / 2.1 GB | 749 ms / 4.1 GB | 1502 ms / 6.2 GB |
| 32000 | f32 | 4.1 GB | 1285 ms / 4.1 GB | 2049 ms / 8.2 GB | 4353 ms / 12.3 GB |
| 32000 | f64 | 8.2 GB | 1916 ms / 8.2 GB | 3288 ms / 16.4 GB | 6966 ms / 24.6 GB |
| 48000 | f32 | 9.2 GB | 3783 ms / 9.2 GB | 5314 ms / 18.4 GB | 11303 ms / 27.7 GB |
| 48000 | f64 | 18.4 GB | 6213 ms / 18.4 GB | 8935 ms / 36.9 GB | **skip (needs 55 GB)** |
| 64000 | f32 | 16.4 GB | 9666 ms / 16.4 GB | 12447 ms / 32.8 GB | 24742 ms / 49.2 GB |
| 64000 | f64 | 32.8 GB | 12032 ms / 32.8 GB | **skip (needs 66 GB)** | **skip (needs 98 GB)** |

### Takeaways that drive the optimization roadmap

1. **Memory, not compute, is the wall.** IVAT's 3-matrix footprint makes
   `n=48000` f64 impossible in 64 GB and caps f64 IVAT well below the
   pairwise-feasible size. Collapsing the IVAT buffers is the highest-leverage
   change for reaching larger `n`.
2. **IVAT construction dominates compute** and is fully serial — ~25 s at
   `n=64000` f32 while 32 cores sit idle. Parallelizing it is the top
   speed win after memory.
3. **float32 roughly halves both** time and memory, at a documented accuracy
   cost; it is opt-in, not the default.

# MST Pareto benchmark: Prim vs certificate-and-filter (`dlmst`)

`mst_pareto.py` measures runtime **and** peak RSS for every way to produce a
VAT order or an IVAT image, end to end from points. It covers numpy, numba,
Cython and the dev-only Rust port in `rust_dlmst/`, with one fresh subprocess
per point. `mst_stage.py` times the MST stage alone when given the matrix.
Algorithm and API: `docs/dlmst.md`.

```bash
python -m benchmarks.mst_pareto --sizes 2000 4000 8000 16000 24000
python -m benchmarks.plot_mst_pareto benchmarks/baselines/mst_pareto_<host>.json f64
python -m benchmarks.mst_stage --sizes 4000 8000 16000 24000
```

The reference run was on 4 CPUs, 15 GB of RAM, d=10 and 8 blobs, recorded in
`baselines/mst_pareto_vm.json`, with plots `baselines/mst_pareto_{order,ivat}_{f64,f32}.png`.
Methods that would need more than 12 GB were skipped. The skip estimate is
conservative for the in-place IVAT.

### VAT order, n = 24 000 (end to end from points)

| method | f64 time | f64 peak | f32 time | f32 peak |
|---|---|---|---|---|
| Prim, C, matrix (existing) | 3.97 s | 4.61 GB | 3.13 s | 2.30 GB |
| Prim, C, points (existing) | 5.03 s | 2 MB | 4.81 s | 2 MB |
| Prim, Rust, matrix | 4.71 s | 4.61 GB | 3.05 s | 2.30 GB |
| cert+filter, C, matrix | 3.55 s | 4.61 GB | 2.63 s | 2.31 GB |
| cert+filter, Rust, matrix | 3.44 s | 4.61 GB | 2.47 s | 2.31 GB |
| **cert+filter, C, points** | **1.50 s** | **6 MB** | **1.33 s** | **3 MB** |
| cert+filter, Rust, points | 2.12 s | 5 MB | 1.93 s | 3 MB |
| cert+filter, C, points, 1 thread | 5.07 s | 6 MB | 4.46 s | 3 MB |
| cert+filter, numpy, points (n=16 000) | 12.65 s | 0.20 GB | — | — |

### IVAT image, n = 16 000 f64 (end to end from points)

| method | time | peak |
|---|---|---|
| `compute_ivat_c` (existing, 2 buffers) | 4.60 s | 4.10 GB |
| `compute_ivat_c(inplace=True)` (existing) | 4.05 s | 2.05 GB |
| numba `compute_ivat` (existing) | 8.79 s | 6.15 GB |
| IVAT from MST, C, matrix | 2.02 s | 4.10 GB |
| IVAT from MST, Rust, matrix | 2.10 s | 4.10 GB |
| **IVAT from MST, C, points** | **1.19 s** | **2.05 GB** |
| IVAT from MST, Rust, points | 1.14 s | 2.05 GB |
| IVAT from MST, numpy, points | 18.56 s | 5.17 GB |

At n=24 000 f64 only the points paths fit under the 12 GB cap. C took
2.10 s and Rust 2.84 s, both at 4.61 GB, the size of the output matrix.

### MST stage only, given the matrix (`mst_stage.py`, median of 3)

| n | dtype | Prim C | Prim Rust | cert+filter C | cert+filter Rust | C, 1 thread | Rust, 1 thread |
|---|---|---|---|---|---|---|---|
| 8 000 | f64 | 136 ms | 184 ms | 94 ms | 96 ms | 258 ms | 312 ms |
| 16 000 | f64 | 480 ms | 671 ms | 290 ms | 370 ms | 1002 ms | 1233 ms |
| 24 000 | f64 | 1448 ms | 2111 ms | **663 ms** | 756 ms | 2152 ms | 2603 ms |
| 24 000 | f32 | 874 ms | 1298 ms | 516 ms | **492 ms** | 1918 ms | 2004 ms |

### Takeaways

1. **The points path sits alone on the Pareto front.** For a VAT order,
   cert+filter from points is 2.6× faster than Prim on the matrix and uses
   about 770× less memory. It is 3.4× faster than the existing O(n) Prim from
   points. For IVAT it is 3.4× faster than in-place IVAT at the same memory,
   and it does not destroy the input.
2. **The speedup comes from parallelism, not asymptotics.** Both are O(n²).
   With one thread, cert+filter is about 1.5× slower than Prim because it
   makes about 2 passes over the matrix where Prim makes ½. It wins once there
   are 2 or more cores, so it should scale further on the 32-core reference
   box.
3. **Rust vs Cython.** The multi-threaded matrix paths, which are bound by
   bandwidth, are equal within noise, and Rust's f32 is marginally ahead. Rust
   is 1.4–1.5× slower on the compute-bound kernels: the on-the-fly distance
   kernel (points path) and serial Prim. Cython builds with `-ffast-math`,
   which lets GCC reassociate and vectorize those reductions, and stable Rust
   has no equivalent. Rust adds no Pareto point, so the package stays on
   Cython.
