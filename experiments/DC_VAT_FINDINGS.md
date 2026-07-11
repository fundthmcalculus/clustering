# Spike: 2D divide-and-conquer VAT ("2D merge-sort") — findings

**Idea (Scott):** a 2D analogue of merge-sort for VAT — bisect a 128×128
dissimilarity block into two 64×64 sub-blocks, order each in parallel, and
merge/insert. Does it produce the exact VAT/iVAT, and does it go faster?

Two concrete forms were prototyped (`experiments/dc_vat.py`) and measured
against the exact serial engine (`compute_ivat_c`).

---

## Result 1 — the idea's *exact* form is the (min, max) closure, and it works

The iVAT dissimilarity `U[i,j]` is the **minimax path distance** — the largest
edge on the MST path between `i` and `j` (Hu's theorem) — which equals the
**single-linkage cophenetic distance**. That is exactly the transitive closure
of `D` in the **(min, max) semiring**:

```
U[i,j] = min over paths of ( max edge on path )
       = closure of D under  U[i,j] <- min(U[i,j], max(U[i,k], U[k,j]))
```

Measured exactness (n = 200, 400):

```
max| (min,max)-closure  -  single-linkage cophenetic |  = 0.00e+00
max| sorted(closure)    -  sorted(iVAT values)       |  = 0.00e+00
```

**This is the real content of the "2D merge-sort" intuition.** The (min,max)
closure is computed by a Floyd–Warshall/Kleene recurrence that **tiles into
independent b×b block operations** — the 128 → two/four 64×64 sub-block
structure exactly — and each block "multiply" `C[i,j] = min_k max(A[i,k],
B[k,j])` is embarrassingly parallel. So the idea is sound *and exact* in this
form.

**Catch:** the closure is **O(n³)**, worse than the **O(n²)** serial iVAT. On a
CPU it loses. Its payoff is parallelism density: the block ops are structurally
a min-plus/GEMM-like tile, ideal for a GPU. Whether the RTX 4080's throughput
beats the O(n²) serial CPU engine for useful n is an open, promising question —
and it slots directly into the GPU work (a tiled (min,max) closure kernel).

---

## Result 2 — the *literal* recursive bisection is approximate AND slower

Approach A: diameter-split the points into two groups, VAT-order each
recursively (in parallel), merge the two ordered blocks end-to-end in the
orientation whose touching endpoints are closest (no interleaving).

**Quality** (blobs with known labels; boundary count ideal = k−1, i.e. every
cluster one contiguous run; contiguity = fraction of same-label neighbours):

| n | k | ideal | serial bnd | serial ctg | dc bnd | dc ctg |
|---|---|-------|-----------|-----------|--------|--------|
| 2000 | 10 | 9 | 9 | 0.995 | 11 | 0.994 |
| 4000 | 15 | 14 | 14 | 0.996 | 15 | 0.996 |
| 8000 | 20 | 19 | 19 | 0.998 | 29 | 0.996 |
| 16000 | 30 | 29 | 29 | 0.998 | 56 | 0.996 |

Serial VAT is essentially perfect (boundary count hits the ideal). D&C is close
in contiguity but accrues **extra seam boundaries that grow with n** — a cluster
straddling a bisection gets fragmented because the block merge does not
interleave. So it is a genuine *approximation*, in the same family as clusiVAT
sampling but via partitioning.

**Speed** (wall-clock vs serial `compute_ivat_c`):

| n | serial ms | dc parallel ms | dc serial ms | speedup |
|---|-----------|----------------|--------------|---------|
| 8000 | 404 | 485 | 613 | 0.83× |
| 16000 | 1403 | 2105 | 2600 | 0.67× |
| 32000 | 6375 | 7380 | 9995 | 0.86× |

**It is slower, not faster (0.67–0.86×).** Reasons:

1. The serial engine is already **O(n²)** and internally parallel (the gather,
   back-copy, and seed scan use OpenMP); only the Prim round-loop and the iVAT
   recurrence are serial, and those have a *small* constant.
2. The recursion re-pays **O(n²) sub-block gathers** (`D[ix(idx,idx)]` fancy
   indexing = large-constant copies) at every level → O(n² log n) overhead with
   a big constant, swamping any gain.
3. The parallelizable part (sub-block ordering) is *not* the bottleneck, so
   Amdahl caps the best case near 1.0× — and overhead pushes it below.

The ceiling for this scheme is ~1.0×; it cannot win at in-RAM scale against an
already-optimized O(n²) parallel engine.

---

## Recommendation

- **Do not pursue recursive-bisection VAT for speed at in-RAM scale.** It is
  approximate and slower. (It *might* have a niche as an out-of-core tiling
  strategy for `n` beyond RAM, where the matrix can't be held at all — a
  different problem than the one measured here.)
- **The exact, genuinely-parallel path is the (min, max) closure (Result 1).**
  It is proven exact and maps to GPU tiles. Recommended next experiment:
  a CuPy/Numba-CUDA tiled `(min,max)` closure on the RTX 4080, measuring the
  crossover against the O(n²) serial CPU engine. This unifies the "2D
  merge-sort" idea with the GPU track.
- If exact *CPU* parallelism is wanted instead, the classical route is
  **parallel Borůvka MST** (O(n²) work, O(log n) rounds) — but it only overtakes
  the serial Prim at very large n / high core counts, given Prim's tiny
  constant here.

---

## Files

- `experiments/dc_vat.py` — prototypes + quality/speed/exactness harness.
- Reference: `tribbleclustering.pcvat.compute_ivat_c` (exact serial iVAT).
