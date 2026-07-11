# Spike: Borůvka parallel MST for VAT — findings

**Question:** can a parallel MST (Borůvka) speed up VAT/iVAT, where the serial
Prim round-loop is the inherently-sequential core?

## The key observation (why the output is *exact*, not approximate)

VAT's ordering is Prim's vertex-insertion order, and by the cut property Prim
only ever traverses **MST edges**. So we can build the MST by *any* method and
then run Prim **restricted to the MST tree** from the same seed (the
global-maximum-dissimilarity vertex) to reproduce the exact VAT ordering — and
hence the exact iVAT image. Parallel VAT therefore reduces to **parallel MST +
an O(n log n) tree traversal**, with no approximation.

Confirmed: the Borůvka-derived VAT order matches serial Prim's on every tested
size (`order_match = 1.0000`), and the iVAT images are bit-identical
(`max |serial − Borůvka| = 0.0`).

![quality](figures/boruvka_vat_quality.png)

## Performance — a modest, eroding win

Borůvka does O(n² log n) work (O(log n) rounds, each an O(n²) min-outgoing-edge
scan) — a log factor *more* than serial compact-Prim's O(n²) — so it can only
win by parallelism. On 32 cores (Numba) the parallel min-edge scan does beat
the (already highly optimized) serial C/OpenMP Prim at small–mid n, but the
extra log-factor work erodes the lead as n grows:

| n | serial Prim (C) | Borůvka (Numba, 32c) | speedup | order match |
|-----|-----------------|----------------------|---------|-------------|
| 1000 | 0.8 ms | 0.2 ms | 4.0× | 1.0000 |
| 2000 | 3.7 ms | 2.0 ms | 1.9× | 1.0000 |
| 4000 | 16.0 ms | 11.8 ms | 1.4× | 1.0000 |
| 8000 | 68.9 ms | 48.9 ms | 1.4× | 1.0000 |
| 16000 | 223.8 ms | 193.8 ms | 1.2× | 1.0000 |
| 32000 | 926.9 ms | 895.5 ms | 1.0× | 1.0000 |

![scaling](figures/boruvka_vat_scaling.png)

A naive **CuPy GPU** Borůvka (per-round `n×n` mask + host union-find) is
consistently ~3–8× *slower* — it is dominated by allocating the mask each round
and by the Python-loop union-find on the host; it is included to show the naive
GPU port does **not** help without a device-side union-find and segmented
per-component reduction.

## Verdict

- **Exactness:** Borůvka-MST VAT is provably and empirically identical to serial
  Prim VAT — a real, clean result (contrast the O(n³) `(min,max)` closure, which
  was exact but hopeless for speed).
- **Speed:** a genuine but **modest** parallel win that **erodes with n** (4× →
  tied by n≈32000) because of the O(n² log n) work and the very small constant
  of the existing C Prim. And the MST is only *part* of total VAT time (the
  O(n²) gather and iVAT recurrence are unchanged), so the end-to-end speedup is
  smaller still.
- **Recommendation:** not worth replacing the serial engine as-is. It becomes
  attractive only if (a) a device-side GPU Borůvka (on-chip union-find +
  segmented reductions, no per-round host sync) is written — the min-edge scan
  is a good GPU reduction — or (b) the whole VAT pipeline moves on-device so the
  distance matrix never leaves the GPU. Both are larger efforts; this spike
  establishes the exactness result and the CPU crossover that would justify (or
  not) that investment.

## Files

- `experiments/boruvka_vat.py` — Numba + CuPy Borůvka, VAT-order-from-MST,
  quality/scaling figures.
- `experiments/figures/boruvka_vat_{quality,scaling}.png`.
