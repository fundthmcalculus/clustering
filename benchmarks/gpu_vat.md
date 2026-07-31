# Fully on-device GPU VAT front-end

`tribbleclustering.gpu_vat.vat_gpu(X)` computes the **exact** VAT ordering with
the whole front-end on the GPU:

1. `gpu.pairwise_distances_device` builds the n×n dissimilarity matrix and keeps
   it **resident** on the device (no host copy).
2. `boruvka_mst_device` builds the exact MST on the device (CuPy RawKernels:
   coalesced min-edge scan, per-component atomicMin, on-device hooking,
   pointer-jumping union-find).
3. The VAT ordering is derived by traversing that MST from the max-dissimilarity
   seed — Prim only ever uses MST edges, so this reproduces the serial VAT order
   **bit-for-bit**. Only the length-n ordering (and parent map) return to host.

This is the design the Borůvka spike pointed to: the ~5× MST win requires the
matrix to already be on the GPU, which an on-device front-end guarantees.

## Exactness

`vat_gpu` order is identical to serial VAT (`np.array_equal`), and the iVAT
image built from it is bit-identical to `compute_ivat_c` (diff `0.0`). The
device MST weight equals scipy's `minimum_spanning_tree` total.

## Performance — end-to-end front-end (distances + MST + order)

RTX 4080 Laptop vs 32-core CPU (`pairwise_distances_c` + `compute_vat_c`),
float64, d=10:

| n | CPU ms | GPU ms | speedup |
|-----|--------|--------|---------|
| 4000 | 78.5 | 16.3 | 4.8× |
| 8000 | 290.9 | 54.6 | 5.3× |
| 16000 | 1031.5 | 169.2 | 6.1× |
| 32000 | 4947.9 | 752.4 | **6.6×** |

The speedup **grows with n** (both the O(n²·d) distances and the O(n²) MST scans
run on-device with no PCIe round-trip), the opposite of the CPU-Borůvka and
GPU-with-transfer curves, which erode. Output is exact.

## Limits

The resident matrix must fit VRAM: n×n×8 bytes for float64 caps n≈38 000 on a
12 GB card (≈54 000 for float32). Beyond that, use the tiled host path
(`gpu.pairwise_distances_gpu`) + the CPU VAT engine, or a future tiled on-device
MST. The iVAT minimax recurrence itself is still built on the host from the
returned ordering; moving it on-device is the natural follow-up.

## Full `IVATMeans` fit — exact, but not (yet) a full-fit speedup

`IVATMeans(on_device=True)` runs the whole front-end (distances + Boruvka MST +
ordering) on the GPU, then finishes the **serial iVAT minimax recurrence on the
host** (via `gpu_vat.ivat_gpu`). Labels and centers are **bit-identical** to the
CPU path. But the recurrence needs the matrix on the host, so the resident
matrix is copied back device→host once and reordered there — and that copy +
host reorder + recurrence offset the ~5–6× front-end win:

| n | CPU fit ms | on_device fit ms | ratio |
|---|-----------|------------------|-------|
| 4000 | 128 | 134 | 0.96× |
| 8000 | 376 | 546 | 0.69× |
| 16000 | 1369 | 2297 | 0.60× |
| 32000 | 6412 | 6356 | 1.01× |

So `on_device` is **opt-in and exact**, not a speedup for the *full* fit on this
hardware. The fast, recommended GPU use is `gpu_vat.vat_gpu` when you need the
**ordering / MST** (5–6.6×). A full-fit win requires moving the iVAT recurrence
on-device too (it is serial — the open follow-up), so the matrix never leaves
the GPU.
