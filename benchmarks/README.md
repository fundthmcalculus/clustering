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
