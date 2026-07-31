# GPU pairwise distances — when it helps

`tribbleclustering.gpu.pairwise_distances_gpu` computes the dense Euclidean
distance matrix on the GPU in row-tiles, streaming each tile into a host
matrix so inputs whose `n×n` output exceeds VRAM still work (only one `R×n`
tile is resident on the device). Accuracy matches the CPU C/OpenMP kernel:
float64 to rounding (`< 1e-10`), float32 to ~`1e-6` (both accumulate the
squared sum in double). A `high_precision=False` mode accumulates in the input
dtype for extra float32 throughput at ~`1e-5` accuracy cost.

## The honest result: GPU wins only for higher-dimensional float32

Measured on an RTX 4080 Laptop GPU (12 GB, consumer — **float64 throughput is
~1/64 of float32**) vs a 32-core Intel CPU. Two structural costs make the GPU
lose in the low-dimensional / float64 regime that is VAT's common case:

1. **Crippled consumer float64.** Direct distance accumulation is done in
   double for accuracy; on this card that runs at a small fraction of float32.
2. **PCIe transfer of the O(n²) result.** The whole matrix must come back to
   host RAM (32 GB for `n=64000` float64), which is bandwidth-bound.

### Speedup vs CPU (n = 16000), GPU/CPU wall-clock ratio (>1 = GPU faster)

| feature dim `d` | float64 | float32 (double acc) | float32 fast (native acc) |
|-----------------|---------|----------------------|---------------------------|
| 10  | 0.34× | 0.53× | — |
| 50  | 0.51× | 0.82× | — |
| 200 | 1.03× | 1.31× | **2.47×** (n=32000) |
| 784 | 0.50× | 1.46× | — |

Takeaways:

- **Low `d` (≤ ~32):** CPU wins comfortably — use the CPU kernel. This is the
  default for `pairwise_distances(..., backend="auto")`, which only routes to
  the GPU when `d ≥ 64`.
- **Higher `d` float32** (embeddings, images): GPU wins 1.3–2.5×, best with
  `high_precision=False`.
- **float64 on this card:** rarely worth it (compute-bound on weak FP64). A
  datacenter GPU (A100/H100, full-rate FP64) would change this.
- **Streaming works:** `n=48000` float64 (18.4 GB > 12 GB VRAM) and `n=64000`
  float32 (16.4 GB) compute correctly by tiling.

## Bigger picture

The transfer cost is the real ceiling here: bringing the matrix back to host
dominates. The larger GPU opportunity is to keep the matrix **on-device** and
run the downstream computation there too — see the divide-and-conquer VAT
findings (`experiments/DC_VAT_FINDINGS.md`): the exact iVAT dissimilarity is a
`(min, max)` transitive closure that tiles naturally onto the GPU and avoids
the round-trip. That is the recommended next GPU experiment.

## Reproduce

```bash
python -m benchmarks.scale_bench --help     # CPU baseline harness
# GPU micro-benchmarks are in the PR description / this doc; the module API is
# tribbleclustering.gpu.pairwise_distances{,_gpu} with backend/high_precision.
```
