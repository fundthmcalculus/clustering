# GPU Fuzzy C-Means — the clean GPU win

Unlike pairwise distances (one pass, huge O(n²) result that must transfer
back), FCM is **iterative**: the data (n × d) stays resident on the device for
all ~100 iterations and only the tiny (k × d) centers move. That amortizes the
transfer, so this is the regime where the GPU wins decisively.

`tribbleclustering.gpu.fuzzy_c_means_gpu` mirrors `fcm.fuzzy_c_means` and
returns `(centers, membership)`. `FuzzyCMeans(use_gpu="auto")` (the default)
routes to it when a CUDA device is present and `n_samples ≥ 5000`, and falls
back to the CPU implementation otherwise.

Each iteration is: squared distances via the gram identity
(`|x_i|² − 2 x_i·c_j + |c_j|²`, a single cuBLAS GEMM — appropriate here because
the values feed membership *ratios* in a fixed-point solve, so the
gram-trick's cancellation error is immaterial), the closed-form membership
`u_ij = D²_ij^{−1/(m−1)} / Σ_l D²_il^{−1/(m−1)}`, and the center update
`(Uᵐ)ᵀX / Σ Uᵐ` (another GEMM).

## Measured (RTX 4080 Laptop vs 32-core CPU, m=2, k=10, d=20, ≤100 iters)

| n_samples | CPU ms | GPU ms | speedup |
|-----------|--------|--------|---------|
| 50000  | 1480  | 46  | **32×** |
| 200000 | 4759  | 108 | **44×** |
| 500000 | 15933 | 286 | **56×** |

Correctness: with the same initial centers the GPU converges to the same fixed
point as the CPU — centers agree to ~1e-5, memberships to ~1e-7, hard labels
>99% identical.

## Memory

Device-resident buffers are `X (n×d)`, `U (n×k)`, `D² (n×k)` — linear in n, not
quadratic — so millions of points fit comfortably in 12 GB (e.g. n=500000,
d=20, k=10 uses well under 1 GB). No tiling needed at these sizes; add tiling
over samples only if `n×(d+2k)` approaches VRAM.
