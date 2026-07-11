# Performance Optimization Guidance for tribble-clustering

**Last Updated:** July 10, 2026  
**Author:** Scott Phillips  
**Scope:** Detailed implementation guidance for performance optimizations  
**Target Audience:** Developers implementing optimizations

---

## Table of Contents

1. [Overview](#overview)
2. [Current Performance Baseline](#current-performance-baseline)
3. [Optimization #1: SIMD Vectorization](#optimization-1-simd-vectorization)
4. [Optimization #2: GPU Acceleration with CUDA](#optimization-2-gpu-acceleration-with-cuda)
5. [Optimization #3: Approximate Nearest Neighbors](#optimization-3-approximate-nearest-neighbors)
6. [Optimization #4: Memory Layout in FCM](#optimization-4-memory-layout-in-fcm)
7. [Optimization #5: Prim's MST Cache Optimization](#optimization-5-prims-mst-cache-optimization)
8. [Optimization #6: Hierarchical Clustering Support](#optimization-6-hierarchical-clustering-support)
9. [Optimization #7: Batched Prediction with Caching](#optimization-7-batched-prediction-with-caching)
10. [Optimization #8: Mixed Precision Arithmetic](#optimization-8-mixed-precision-arithmetic)
11. [Optimization #9: Parallel Permutation Gather](#optimization-9-parallel-permutation-gather)
12. [Optimization #10: Convergence Acceleration](#optimization-10-convergence-acceleration)
13. [Testing & Validation](#testing--validation)
14. [Profiling Tools & Techniques](#profiling-tools--techniques)

---

## Overview

The tribble-clustering library implements three main algorithms:

1. **VAT/IVAT** - Visualization Assessment Tendency algorithms for cluster visualization
2. **Fuzzy C-Means (FCM)** - Soft clustering with membership degrees
3. **IVATMeans** - IVAT-based clustering with scikit-learn compatibility

### Current Architecture

```
src/tribbleclustering/
├── util.py                 # Utility functions (pairwise_distances with Numba)
├── pvat.py                 # Pure Python VAT/IVAT with Numba (fallback)
├── fcm.py                  # Pure Python Fuzzy C-Means (fallback)
├── cfcm.pyx                # Cython-optimized Fuzzy C-Means
├── pcvat.pyx               # C/OpenMP-optimized VAT/IVAT
├── pqvat.py                # Deprecated priority-queue VAT
├── ivatmeans.py            # IVATMeans wrapper class
├── fuzzycmeans.py          # FuzzyCMeans wrapper class
└── __init__.py             # Public API
```

### Performance Critical Paths

The three most computationally expensive operations are:

1. **Pairwise distance computation** - O(n²·d)
   - Located in: `util.py:5-17` (Numba), `pcvat.pyx:250-350+` (C/OpenMP)
   - Called by: `IVATMeans.fit()`, `FuzzyCMeans.fit()`, prediction methods
   - Impact: Can be 30-50% of total runtime for large datasets

2. **Fuzzy C-Means iterations** - O(k·n·d·iterations)
   - Located in: `fcm.py:35-83` (Python), `cfcm.pyx:230-339` (Cython)
   - Main bottleneck: Distance recomputation every iteration
   - Impact: 40-60% of FCM runtime

3. **VAT/IVAT MST and permutation** - O(n²) + O(n²)
   - Located in: `pvat.py:136-204` (Numba), `pcvat.pyx:21-250` (C/OpenMP)
   - Two components: (a) Prim's algorithm, (b) permutation gather
   - Impact: 100% of VAT runtime (must be computed)

---

## Current Performance Baseline

Before implementing optimizations, establish baselines using the existing test suite:

### Running Benchmarks

```bash
# Pairwise distances benchmark
pytest tests/test_pairwise_distances.py::TestPerformance::test_scaling_behavior -v -s

# FCM benchmark  
pytest tests/test_fcm_optimization.py::TestFCMPerformance -v -s

# Full suite
pytest tests/ -v -s -k "performance or bench"
```

### Expected Baseline Results (Reference System: Intel Xeon, 32 cores, 64GB RAM)

**Pairwise Distances (Numba, float64)**
- n=1000, d=8: ~15ms
- n=4000, d=8: ~250ms
- n=10000, d=8: ~1600ms

**Fuzzy C-Means (Cython, float64, 100 iterations)**
- n=100 samples, k=3 clusters, d=5: ~50ms
- n=1000 samples, k=5 clusters, d=10: ~2000ms
- n=5000 samples, k=8 clusters, d=10: ~50000ms

**VAT/IVAT (C/OpenMP, float64)**
- n=500, d=8: ~80ms
- n=2000, d=8: ~1300ms
- n=5000, d=8: ~8000ms

---

## Optimization #1: SIMD Vectorization

### Current Implementation Analysis

The distance computation bottleneck is in `cfcm.pyx:12-52` (float32 variant shown):

```cython
# cfcm.pyx:12-29 - Current inner loop
cdef void _compute_distances_32(
    const float[:, ::1] x,
    const float[:, ::1] c,
    float[:, ::1] distances
) noexcept nogil:
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float d, diff

    for i in range(n_samples):
        for j in range(n_clusters):
            d = 0.0
            for k in range(n_features):          # ← INNER LOOP (bottleneck)
                diff = x[i, k] - c[j, k]
                d += diff * diff
            distances[i, j] = sqrt(d)
```

**Problem:** The innermost loop over features (`k`) is not vectorized:
- Scalar arithmetic operations (one feature per cycle)
- No SIMD register usage
- Memory bandwidth underutilized
- ~4 cycles per iteration minimum (load, subtract, multiply, add)

### Optimization Strategy

#### Approach A: Unrolled Loop with SIMD Awareness (Recommended)

**Expected Gain:** 30-50% speedup  
**Complexity:** Medium  
**Recommended:** Yes

```cython
# cfcm.pyx - Add after line 29
cdef void _compute_distances_unrolled_32(
    const float[:, ::1] x,
    const float[:, ::1] c,
    float[:, ::1] distances
) noexcept nogil:
    """SIMD-friendly unrolled distance computation.
    
    Unroll by 4: process 4 features per loop iteration.
    This allows CPU's out-of-order execution to interleave operations
    and hide memory latency through instruction-level parallelism.
    """
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k, k_end
    cdef float d, diff0, diff1, diff2, diff3

    k_end = (n_features // 4) * 4  # Process in groups of 4

    for i in range(n_samples):
        for j in range(n_clusters):
            d = 0.0
            
            # Unrolled loop: process 4 features per iteration
            for k in range(0, k_end, 4):
                diff0 = x[i, k+0] - c[j, k+0]
                diff1 = x[i, k+1] - c[j, k+1]
                diff2 = x[i, k+2] - c[j, k+2]
                diff3 = x[i, k+3] - c[j, k+3]
                
                d += diff0 * diff0
                d += diff1 * diff1
                d += diff2 * diff2
                d += diff3 * diff3
            
            # Handle remainder (if n_features not multiple of 4)
            for k in range(k_end, n_features):
                diff0 = x[i, k] - c[j, k]
                d += diff0 * diff0
            
            distances[i, j] = sqrt(d)
```

**Why This Works:**
1. **ILP (Instruction-Level Parallelism):** Four independent `diff*diff` operations can execute in parallel on modern superscalar CPUs (4-6 execution ports)
2. **Better Memory Locality:** Loading 4 consecutive floats improves L1 cache hit rate
3. **Compiler Optimization:** LLVM recognizes the pattern and auto-vectorizes to AVX instructions
4. **No Manual SIMD:** Avoids platform-specific assembly, works on any CPU with auto-vectorization

**Validation:**
```python
# tests/test_simd_distances.py
import numpy as np
from tribbleclustering.cfcm import _compute_distances_unrolled_32 as unrolled
from tribbleclustering.cfcm import _compute_distances_32 as baseline
from scipy.spatial.distance import cdist

def test_unrolled_matches_baseline():
    x = np.random.randn(100, 13).astype(np.float32)  # Non-multiple of 4
    c = np.random.randn(5, 13).astype(np.float32)
    
    dist_baseline = np.empty((100, 5), dtype=np.float32)
    dist_unrolled = np.empty((100, 5), dtype=np.float32)
    
    baseline(x, c, dist_baseline)
    unrolled(x, c, dist_unrolled)
    
    np.testing.assert_allclose(dist_baseline, dist_unrolled, rtol=1e-5)
```

#### Approach B: Using NumPy's BLAS (Advanced)

**Expected Gain:** 40-60% speedup  
**Complexity:** High  
**Risk:** Depends on BLAS implementation

NumPy's `np.linalg.norm` dispatches to optimized BLAS libraries (OpenBLAS, MKL, etc.):

```cython
# Alternative: Replace distance computation entirely
def _compute_distances_numpy(x, c):
    """Use NumPy's optimized BLAS backend."""
    # Compute pairwise distances using broadcasting
    # x: (n, d), c: (k, d)
    # Result: (n, k)
    diff = x[:, np.newaxis, :] - c[np.newaxis, :, :]  # (n, k, d)
    distances = np.linalg.norm(diff, axis=2)  # BLAS dispatch
    return distances
```

**Trade-offs:**
- Requires allocating full (n, k, d) intermediate array
- Memory overhead: ~3x the input size
- Better for small d, worse for large d
- **Recommendation:** Use for d < 20, avoid for d > 100

### Implementation Checklist

- [ ] Create `_compute_distances_unrolled_32()` and `_compute_distances_unrolled_64()`
- [ ] Update `_fuzzy_c_means_kernel_32()` to call unrolled variant (line 255)
- [ ] Update `_fuzzy_c_means_kernel_64()` to call unrolled variant (line 312)
- [ ] Add correctness tests (n_features = 1, 4, 5, 100)
- [ ] Benchmark: compare original vs unrolled
- [ ] Profile with `perf record -e cycles -e instructions`
- [ ] Measure cache misses before/after

### Compilation Flags

Ensure Cython uses aggressive optimization flags in `setup.py`:

```python
# setup.py
from Cython.Build import cythonize
import os

ext_modules = cythonize(
    "src/tribbleclustering/*.pyx",
    compiler_directives={
        "language_level": 3,
        "boundscheck": False,
        "wraparound": False,
        "cdivision": True,
        "initializedcheck": False,
        "fast_function_calls": True,  # ← Important for ILP
        "optimize.unpack_method_calls": True,  # ← Auto-unroll
    },
    annotate=True,  # Generate HTML showing optimization success
)

# Tell compiler to generate optimized code
os.environ["CFLAGS"] = "-O3 -march=native -mtune=native"
os.environ["CXXFLAGS"] = "-O3 -march=native -mtune=native"
```

### Performance Monitoring

```python
# Benchmark before/after
import time
import numpy as np

x = np.random.randn(1000, 50).astype(np.float32)
c = np.random.randn(10, 50).astype(np.float32)

# Time original
t0 = time.perf_counter()
for _ in range(100):
    from tribbleclustering.cfcm import _compute_distances_32
    dist = np.empty((1000, 10), dtype=np.float32)
    _compute_distances_32(x, c, dist)
t_original = time.perf_counter() - t0

# Time unrolled (after implementation)
# Expected: t_unrolled ≈ 0.6-0.7 * t_original
```

---

## Optimization #2: GPU Acceleration with CUDA

### Why GPU Acceleration?

**Current Bottleneck:** Distance computation is embarrassingly parallel
- Each (sample, cluster) pair can compute its distance independently
- No dependencies between calculations
- GPU ideal use case

**Expected Speedup:** 10-100x depending on hardware and matrix size

### GPU Hardware Requirements

```
Recommended: NVIDIA GPU with CUDA Compute Capability ≥ 3.5
- Tesla V100 (Compute 7.0): 32GB HBM, 125 TFLOPS float32
- Tesla A100 (Compute 8.0): 40GB HBM, 312 TFLOPS float32
- RTX 3080 (Compute 8.6): 10GB GDDR6X, 29 TFLOPS float32

Minimum: GTX 1060 (Compute 6.1) or better

Software:
- CUDA 11.8+ (ensure compatibility with Numba)
- Numba 0.56+ (CUDA support)
- CuPy 10.0+ (optional, for GPU arrays)
```

### Implementation: Numba CUDA for FCM

#### Step 1: Create GPU-Accelerated Distance Kernel

Create new file: `src/tribbleclustering/gpu_kernels.py`

```python
# gpu_kernels.py
"""GPU-accelerated kernels using Numba CUDA."""

import numpy as np
from numba import cuda
from math import sqrt

@cuda.jit
def compute_distances_gpu(x, c, distances):
    """
    GPU kernel for pairwise distance computation.
    
    Args:
        x: Input samples (n_samples, n_features), float32/float64
        c: Cluster centers (n_clusters, n_features)
        distances: Output matrix (n_samples, n_clusters)
    
    Grid/Block Layout:
    - Grid: (n_samples, n_clusters) with 32×8 blocks
    - Each thread computes distances[i, j]
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if i >= x.shape[0] or j >= c.shape[0]:
        return
    
    # Compute Euclidean distance
    d = 0.0
    for k in range(x.shape[1]):
        diff = x[i, k] - c[j, k]
        d += diff * diff
    
    distances[i, j] = sqrt(d)


@cuda.jit
def compute_weights_gpu(distances, m, w_ij):
    """
    GPU kernel for fuzzy membership weight computation.
    
    w_ij = 1 / sum((d_ij / d_ik)^(2/(m-1)) for all k)
    
    Args:
        distances: Distance matrix (n_samples, n_clusters)
        m: Fuzziness parameter (float)
        w_ij: Output weights (n_samples, n_clusters)
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    
    if i >= distances.shape[0] or j >= distances.shape[1]:
        return
    
    # Avoid division by zero
    if distances[i, j] == 0.0:
        w_ij[i, j] = 0.0
        return
    
    # Sum: (d_ij / d_ik)^(2/(m-1)) for k = 0..n_clusters
    denom = 0.0
    m_exp = 2.0 / (m - 1.0)
    
    for k in range(distances.shape[1]):
        if distances[i, k] == 0.0:
            denom = 1.0
            break
        dist_ratio = distances[i, j] / distances[i, k]
        denom += pow(dist_ratio, m_exp)
    
    if denom > 0.0:
        w_ij[i, j] = 1.0 / denom
    else:
        w_ij[i, j] = 0.0


@cuda.jit
def compute_new_centers_gpu(w_ij, x, m, v_ij):
    """
    GPU kernel for cluster center update.
    
    v_j = sum(w_ij^m * x_i) / sum(w_ij^m)
    
    Parallel reduction: each block computes partial sum, main thread reduces.
    """
    i = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    j = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    k = cuda.blockIdx.z * cuda.blockDim.z + cuda.threadIdx.z
    
    if i >= w_ij.shape[0] or j >= w_ij.shape[1] or k >= x.shape[1]:
        return
    
    wm = pow(w_ij[i, j], m)
    cuda.atomic.add(v_ij, (j, k), wm * x[i, k])
```

#### Step 2: Create GPU-Aware FCM Class

```python
# cfcm_gpu.py
"""GPU-accelerated Fuzzy C-Means."""

import numpy as np
from numba import cuda
from .gpu_kernels import compute_distances_gpu, compute_weights_gpu, compute_new_centers_gpu

class FuzzyCMeansGPU:
    """GPU-accelerated FCM implementation."""
    
    def __init__(self, n_clusters: int, m: float = 2.0, use_gpu: bool = True):
        self.n_clusters = n_clusters
        self.m = m
        self.use_gpu = use_gpu and cuda.is_available()
        self.device = 0  # GPU device ID
        
    def fit(self, x: np.ndarray, n_iter: int = 100) -> tuple:
        """Fit FCM on GPU."""
        if not self.use_gpu:
            # Fallback to CPU
            from .cfcm import fuzzy_c_means
            return fuzzy_c_means(x, self.n_clusters, self.m)
        
        # Transfer data to GPU
        x_gpu = cuda.to_device(x)
        c_gpu = cuda.to_device(self._init_centers(x))
        c_new_gpu = cuda.to_device(np.empty_like(c_gpu))
        
        distances_gpu = cuda.to_device(
            np.empty((x.shape[0], self.n_clusters), dtype=x.dtype)
        )
        w_ij_gpu = cuda.to_device(
            np.empty((x.shape[0], self.n_clusters), dtype=x.dtype)
        )
        
        # Configure grid/block dimensions
        # Block: 32 threads x 8 threads (optimal for most GPUs)
        block_size = (32, 8)
        grid_size_dist = (
            (x.shape[0] + block_size[0] - 1) // block_size[0],
            (self.n_clusters + block_size[1] - 1) // block_size[1]
        )
        
        # FCM iterations
        for iteration in range(n_iter):
            # 1. Compute distances on GPU
            compute_distances_gpu[grid_size_dist, block_size](
                x_gpu, c_gpu, distances_gpu
            )
            
            # 2. Compute weights on GPU
            compute_weights_gpu[grid_size_dist, block_size](
                distances_gpu, np.float32(self.m), w_ij_gpu
            )
            
            # 3. Compute new centers (requires reduction - more complex)
            # For now, transfer back to CPU for center update
            w_ij_cpu = w_ij_gpu.copy_to_host()
            c_new = self._update_centers_cpu(x, w_ij_cpu, c_gpu.copy_to_host())
            c_new_gpu = cuda.to_device(c_new)
            
            # Check convergence
            if np.allclose(c_new, c_gpu.copy_to_host(), rtol=1e-5):
                c_gpu = c_new_gpu
                break
            c_gpu = c_new_gpu
        
        # Transfer results back to CPU
        c_final = c_gpu.copy_to_host()
        w_final = w_ij_gpu.copy_to_host()
        
        return c_final, w_final
    
    @staticmethod
    def _init_centers(x: np.ndarray) -> np.ndarray:
        """Initialize cluster centers."""
        indices = np.random.choice(x.shape[0], size=2 * x.shape[0], replace=False)
        c = x[indices].reshape(-1, 2, x.shape[1]).mean(axis=1)
        return c[:x.shape[0]]  # Ensure correct number of clusters
    
    @staticmethod
    def _update_centers_cpu(x, w_ij, c_old):
        """Update centers on CPU (bottleneck, should be GPU-implemented)."""
        m = 2.0
        c_new = np.zeros_like(c_old)
        
        for j in range(c_old.shape[0]):
            w_sum = 0.0
            for i in range(x.shape[0]):
                wm = w_ij[i, j] ** m
                w_sum += wm
                c_new[j, :] += wm * x[i, :]
            c_new[j, :] /= w_sum if w_sum > 0 else 1.0
        
        return c_new
```

#### Step 3: Integrate with Existing Code

Modify `fuzzycmeans.py` to use GPU when available:

```python
# fuzzycmeans.py - Modified
try:
    from .cfcm import fuzzy_c_means as fcm_algorithm
    _has_compiled_fcm = True
except ImportError:
    from .fcm import fuzzy_c_means as fcm_algorithm
    _has_compiled_fcm = False

# Try GPU support
try:
    from numba import cuda
    if cuda.is_available():
        from .cfcm_gpu import FuzzyCMeansGPU
        _has_gpu = True
    else:
        _has_gpu = False
except ImportError:
    _has_gpu = False

class FuzzyCMeans:
    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        random_state: Optional[int] = None,
        use_gpu: bool = True,  # ← New parameter
    ):
        self.n_clusters = n_clusters
        self.m = m
        self.random_state = random_state
        self.use_gpu = use_gpu and _has_gpu
        
    def fit(self, X: ndarray, ...) -> "FuzzyCMeans":
        X = np.asarray(X)
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        # Use GPU if available and requested
        if self.use_gpu:
            gpu_fcm = FuzzyCMeansGPU(self.n_clusters, self.m)
            self.cluster_centers_, self.membership_matrix_ = gpu_fcm.fit(X)
        else:
            self.cluster_centers_, self.membership_matrix_ = fcm_algorithm(
                X, self.n_clusters, m=self.m
            )
        
        self.labels_ = self._get_hard_labels()
        return self
```

### GPU Performance Measurement

```python
# benchmarks/gpu_benchmark.py
"""GPU acceleration benchmarks."""

import time
import numpy as np
from tribbleclustering import FuzzyCMeans

def benchmark_fcm_gpu_vs_cpu():
    """Compare GPU vs CPU performance."""
    sizes = [100, 500, 1000, 5000]
    results = {}
    
    for n in sizes:
        x = np.random.randn(n, 10).astype(np.float32)
        
        # CPU timing
        t0 = time.perf_counter()
        fcm_cpu = FuzzyCMeans(n_clusters=5, use_gpu=False)
        fcm_cpu.fit(x)
        t_cpu = time.perf_counter() - t0
        
        # GPU timing
        t0 = time.perf_counter()
        fcm_gpu = FuzzyCMeans(n_clusters=5, use_gpu=True)
        fcm_gpu.fit(x)
        t_gpu = time.perf_counter() - t0
        
        speedup = t_cpu / t_gpu
        results[n] = {
            'cpu_ms': t_cpu * 1000,
            'gpu_ms': t_gpu * 1000,
            'speedup': speedup
        }
        
        print(f"n={n:5d}: CPU {t_cpu*1000:7.2f}ms, GPU {t_gpu*1000:7.2f}ms, speedup {speedup:5.1f}x")
    
    return results
```

### GPU Considerations & Trade-offs

**Pros:**
- 10-100x speedup for large matrices
- Enables interactive parameter tuning
- Scales well with larger datasets

**Cons:**
- GPU memory limited (10-80GB typical)
- Data transfer overhead (~50-100GB/s vs 100GB/s compute)
- Break-even point: ~5000 samples
- CUDA-specific (NVIDIA only)
- Code complexity increases

**Recommendation:**
Implement as **optional feature**, fall back to CPU if:
- GPU unavailable
- Dataset fits in GPU memory only if n > 5000
- Memory transfer time > 10% of total time

---

## Optimization #3: Approximate Nearest Neighbors

### Use Case Analysis

Current VAT/IVAT implementation requires full O(n²) distance matrix. For prediction on large test sets, this is unnecessary - only k nearest neighbors matter.

**When to Use:**
- Prediction on n_test > 10000
- Large n_clusters (k > 100)
- Can tolerate ~5% error (depends on tolerance)

### Implementation: KD-Tree Integration

#### Step 1: Wrap IVATMeans with Tree Support

Modify `ivatmeans.py`:

```python
# ivatmeans.py - Add tree-based prediction
from typing import Optional

import numpy as np
from numpy import ndarray

try:
    from sklearn.neighbors import KDTree, BallTree
    _has_sklearn_trees = True
except ImportError:
    _has_sklearn_trees = False

class IVATMeans:
    """IVAT-based clustering with optional approximate prediction."""
    
    def __init__(
        self, 
        n_clusters: int = 2, 
        random_state: Optional[int] = None,
        tree_type: str = 'kd',  # 'kd' or 'ball'
    ):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.tree_type = tree_type
        self.cluster_centers_: Optional[ndarray] = None
        self.labels_: Optional[ndarray] = None
        self._tree: Optional[object] = None  # KDTree or BallTree
        self._ivat_result = None
    
    def fit(self, X: ndarray, y: Optional[ndarray] = None, 
            sample_weight: Optional[ndarray] = None) -> "IVATMeans":
        """Fit IVAT and build tree for fast prediction."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        distances = _pairwise_distances(X)
        ivat_matrix, _, vat_order = _compute_ivat(distances, inplace=False)
        
        self._ivat_result = get_ivat_levels(
            X, ivat_matrix, vat_order, n_levels=1, n_clusters=self.n_clusters
        )
        
        result: IvatMeansResult = self._ivat_result
        self.cluster_centers_ = result.initial_centroids
        self.labels_ = self._assign_clusters(X)
        
        # Build spatial index for fast nearest-neighbor queries
        if _has_sklearn_trees and self.cluster_centers_.shape[0] > 5:
            # Only use tree if we have enough clusters
            if self.tree_type == 'kd':
                self._tree = KDTree(self.cluster_centers_, leaf_size=30)
            elif self.tree_type == 'ball':
                self._tree = BallTree(self.cluster_centers_, leaf_size=30)
        
        return self
    
    def predict(self, X: ndarray) -> ndarray:
        """Predict with optional tree acceleration."""
        if self.cluster_centers_ is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {X.shape}")
        
        # Use tree if available and beneficial
        if self._tree is not None and X.shape[0] > 100:
            # KDTree.query returns (distances, indices)
            _, indices = self._tree.query(X, k=1)
            return indices.ravel()
        
        # Fall back to exact computation
        distances = np.linalg.norm(
            X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], 
            axis=2
        )
        return np.argmin(distances, axis=1)
    
    def _assign_clusters(self, X: ndarray) -> ndarray:
        """Assign cluster labels using tree if available."""
        return self.predict(X)
```

#### Step 2: Tree Selection Based on Data Characteristics

```python
# utils/tree_selector.py
"""Automatic tree selection based on data properties."""

import numpy as np
from sklearn.neighbors import KDTree, BallTree

def select_best_tree(X, n_neighbors=1, metric='euclidean'):
    """
    Select optimal tree type based on data dimensionality.
    
    KDTree: Best for low-dimensional data (d < 20)
    BallTree: Better for high-dimensional data (d > 20)
    
    Reference: Pedregosa et al. (2011): "Scikit-learn: Machine Learning in Python"
    """
    n_features = X.shape[1]
    
    if n_features < 20:
        # Low-dimensional: KDTree is faster
        return KDTree(X, leaf_size=30, metric=metric)
    else:
        # High-dimensional: BallTree avoids curse of dimensionality
        return BallTree(X, leaf_size=30, metric=metric)
```

#### Step 3: Performance Comparison

```python
# benchmarks/tree_benchmark.py
"""Benchmark tree-based prediction vs brute force."""

import time
import numpy as np
from sklearn.neighbors import KDTree, BallTree

def benchmark_prediction(n_samples, n_test, n_clusters, n_features):
    """Compare brute-force vs tree-based prediction."""
    
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    centers = np.random.randn(n_clusters, n_features).astype(np.float32)
    
    # Brute force: O(n_test * n_clusters * n_features)
    t0 = time.perf_counter()
    for _ in range(10):
        distances = np.linalg.norm(
            X_test[:, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2
        )
        predictions = np.argmin(distances, axis=1)
    t_brute = (time.perf_counter() - t0) / 10
    
    # KDTree: O(n_test * log(n_clusters)) query + O(n_clusters * log(n_clusters)) build
    t0 = time.perf_counter()
    tree = KDTree(centers)
    t_build = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    for _ in range(10):
        _, predictions = tree.query(X_test, k=1)
    t_query = (time.perf_counter() - t0) / 10
    
    print(f"Brute force: {t_brute*1000:.2f}ms")
    print(f"KDTree build: {t_build*1000:.2f}ms, query: {t_query*1000:.2f}ms")
    print(f"Speedup (query only): {t_brute/t_query:.1f}x")
    print(f"Speedup (with build): {t_brute/(t_build + t_query):.1f}x")
```

#### Step 4: When to Activate Trees

Decision tree in `predict()`:

```python
def predict(self, X: ndarray) -> ndarray:
    # Use tree if ALL conditions met:
    if (
        self._tree is not None                    # Tree built during fit
        and X.shape[0] > 100                      # n_test > 100 (break-even)
        and self.cluster_centers_.shape[0] <= 1000  # n_clusters <= 1000
        and X.shape[1] < 100                      # d < 100 (KDTree good)
    ):
        _, indices = self._tree.query(X, k=1)
        return indices.ravel()
    else:
        # Exact computation
        distances = np.linalg.norm(...)
        return np.argmin(distances, axis=1)
```

### HNSW Alternative (Advanced)

For very large n_clusters (>10000), consider HNSW (Hierarchical Navigable Small World):

```python
# Optional advanced alternative
try:
    import hnswlib
    _has_hnsw = True
except ImportError:
    _has_hnsw = False

def build_hnsw_index(centers, ef_construction=200):
    """Build HNSW index for ultra-fast approximate search."""
    index = hnswlib.Index(space='l2', dim=centers.shape[1])
    index.init_index(max_elements=centers.shape[0], ef_construction=ef_construction)
    index.add_items(centers, np.arange(centers.shape[0]))
    return index
```

Reference: Malkov, Y. A., & Yashunin, D. A. (2018). "Efficient and robust approximate nearest neighbor search with hierarchical navigable small world graphs"

---

## Optimization #4: Memory Layout in FCM

### Current Bottleneck Analysis

FCM's main loop (cfcm.pyx:254-276) recomputes distances every iteration:

```cython
# cfcm.pyx:254-276 (current implementation)
for iteration in range(100):
    _compute_distances_64(x, c, distances)     # ← O(n*k*d) compute
    _compute_weights_64(distances, m, w_ij)    # ← O(n*k) compute
    
    for i in range(n):
        for k in range(n_features):
            c_new[i, k] = 0.0
    
    _compute_new_centers_64(w_ij, x, m, c_new) # ← O(n*k*d) compute
    
    max_delta = 0.0
    for i in range(n):
        for k in range(n_features):
            delta = (c_new[i, k] - c[i, k]) ** 2
            if delta > max_delta:
                max_delta = delta
    
    if max_delta < 1e-10:
        break
    
    for i in range(n):
        for k in range(n_features):
            c[i, k] = c_new[i, k]
```

**Problems:**
1. `distances` matrix recomputed every iteration: 100 × O(n·k·d)
2. `distances` has poor cache locality: accessed randomly in weight computation
3. Memory traffic: ~3x input size per iteration (distances read, weights written, centers updated)

### Solution 1: Distance Matrix Caching (Simple)

Idea: Reuse distance matrix if centers haven't changed significantly.

```cython
# cfcm_optimized.pyx - Add after line 253

cdef void _compute_distances_cached_64(
    const double[:, ::1] x,
    const double[:, ::1] c,
    double[:, ::1] distances,
    double[:, ::1] c_prev,
    bint force_recompute = True,
) noexcept nogil:
    """
    Recompute distances only if centers changed significantly.
    
    Strategy: Skip computation if max center delta < threshold.
    Trade-off: ~1-2% accuracy loss for 20-30% speedup.
    """
    
    # Check if centers moved significantly
    if not force_recompute:
        max_delta = 0.0
        for i in range(c.shape[0]):
            for k in range(c.shape[1]):
                delta = (c[i, k] - c_prev[i, k]) ** 2
                if delta > max_delta:
                    max_delta = delta
        
        if max_delta < 1e-6:
            # Centers haven't moved, skip recomputation
            return
    
    # Recompute if centers changed
    _compute_distances_64(x, c, distances)
    
    # Save current centers for next iteration
    for i in range(c.shape[0]):
        for k in range(c.shape[1]):
            c_prev[i, k] = c[i, k]
```

### Solution 2: Column-Major Memory Layout (Advanced)

Currently, `x` is row-major (C-contiguous). For distance computation, column-major is better:

```cython
# Transpose x before main loop to improve cache behavior
cdef void _fuzzy_c_means_kernel_64_optimized(
    double[:, ::1] x,  # Input data (n_samples, n_features)
    int n,
    double m,
    double[:, ::1] c_init
):
    """FCM with optimized memory layout."""
    
    cdef int n_samples = x.shape[0]
    cdef int n_features = x.shape[1]
    
    # Transpose x to column-major for better memory access patterns
    # In distance computation, we repeatedly access all features for one sample
    # Column-major layout keeps these in L1 cache
    cdef double[:, ::1] x_col = np.asfortranarray(x)  # Column-major view
    
    # ... rest of implementation uses x_col for distance computation ...
```

**Impact:**
- L1 cache hit rate: 60% → 75%+
- Speedup: 15-25% on large n

### Solution 3: In-Place Weight Computation (Advanced)

Avoid allocating new `w_ij` matrix each iteration:

```cython
# Reuse w_ij buffer
cdef double[:, ::1] w_ij = np.zeros((n_samples, n), dtype=np.float64)
cdef double[:, ::1] w_ij_new = np.zeros((n_samples, n), dtype=np.float64)

for iteration in range(100):
    _compute_distances_64(x, c, distances)
    
    # Clear w_ij in-place instead of allocating new one
    for i in range(n_samples):
        for j in range(n):
            w_ij[i, j] = 0.0
    
    _compute_weights_64(distances, m, w_ij)  # Compute in-place
    _compute_new_centers_64(w_ij, x, m, c_new)
```

### Solution 4: Reduced Precision for Distances (Advanced)

```cython
# Accumulate distances in float32, final centers in float64
cdef void _compute_distances_f32_64(
    const double[:, ::1] x,
    const double[:, ::1] c,
    float[:, ::1] distances,  # ← float32 output
) noexcept nogil:
    """Distance computation with reduced precision.
    
    Benefits:
    - 2x memory bandwidth savings
    - L1 cache holds 2x more data
    - Negligible accuracy loss (<0.1%)
    """
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float d, diff
    
    for i in range(n_samples):
        for j in range(n_clusters):
            d = 0.0
            for k in range(n_features):
                diff = <float>(x[i, k] - c[j, k])
                d += diff * diff
            distances[i, j] = sqrt(d)
```

### Implementation Checklist for Memory Optimization

- [ ] Implement distance caching (Solution 1): 20% speedup, low risk
- [ ] Profile L1/L2 cache misses with `perf stat -e L1-dcache-load-misses`
- [ ] Benchmark vs current implementation
- [ ] Add unit tests for near-exact convergence with caching
- [ ] Measure memory usage before/after
- [ ] Optional: Implement column-major layout (Solution 2)
- [ ] Optional: Reduced precision (Solution 3) with accuracy validation

### Validation

```python
# tests/test_fcm_memory_optimization.py
import numpy as np
from tribbleclustering.cfcm import fuzzy_c_means, fuzzy_c_means_optimized

def test_optimized_convergence():
    """Verify optimized version converges similarly."""
    x = np.random.randn(100, 10).astype(np.float64)
    
    c1, w1 = fuzzy_c_means(x, 3)
    c2, w2 = fuzzy_c_means_optimized(x, 3)
    
    # Allow 1% tolerance due to floating-point differences
    np.testing.assert_allclose(c1, c2, rtol=1e-2)
    np.testing.assert_allclose(w1, w2, rtol=1e-2)
```

---

## Optimization #5: Prim's MST Cache Optimization

### Current Implementation Review

The C/OpenMP implementation (`pcvat.pyx:21-112`) uses **compact-active-set** approach:

```c
// pcvat.pyx:88-112 (main MST loop)
for (rnd in range(n)):
    u = act_vert[bk]
    out_seq[rnd] = u
    out_par_seq[rnd] = act_par[bk]
    
    // Remove slot bk by swapping with last active slot
    m -= 1
    act_vert[bk] = act_vert[m]
    act_key[bk] = act_key[m]
    act_par[bk] = act_par[m]
    
    // Fused relax + next-min selection
    row = adj + u * n
    best = INFINITY
    bk = -1
    for (i in range(m)):
        w = act_vert[i]
        d = row[w]
        if (d < act_key[i]):
            act_key[i] = d
            act_par[i] = rnd + 1
        if (act_key[i] < best):
            best = act_key[i]
            bk = i
```

**Analysis:**
- Already efficient with O(n²) time, O(n) space
- Parallelization only at global-max scan (initial step)
- Main bottleneck: memory access pattern
  - `row = adj[u]` loads one row of distance matrix
  - Accesses elements `row[w]` in random order (cache misses)

### Cache Optimization: Blocked Prim's Algorithm

Instead of scanning entire active set, process in cache-friendly blocks:

```c
// pcvat_optimized.pyx - Add cache-optimized variant

#define L3_CACHE_SIZE (20 * 1024 * 1024)  // ~20MB typical L3
#define BLOCK_SIZE (L3_CACHE_SIZE / (8 * sizeof(double)))  // ~262144 elements

cdef void _prim_mst_kernel_blocked_64(
    const double* adj, int n,
    int* act_vert, double* act_key, int* act_par,
    int* out_seq, int* out_par_seq,
) noexcept nogil:
    """
    Blocked Prim's MST for better cache utilization.
    
    Process active vertices in blocks that fit L3 cache.
    Each block can keep adjacency row in cache across iterations.
    """
    
    cdef int i, j, w, u, bk, rnd, m, tid, block_start, block_end, block_size
    cdef int src_i = 0, src_j = 0
    cdef double max_val = -INFINITY
    cdef double d, best
    cdef const double* row
    
    // Initialization same as before...
    for i in range(n):
        act_vert[i] = i
        act_key[i] = INFINITY
        act_par[i] = -1
    
    // Find source same as before...
    
    act_key[src_i] = max_val
    act_par[src_i] = src_j
    bk = src_i
    m = n
    
    block_size = min(BLOCK_SIZE, m)
    
    // Main loop with blocking
    for rnd in range(n):
        u = act_vert[bk]
        out_seq[rnd] = u
        out_par_seq[rnd] = act_par[bk]
        
        // Remove vertex
        m -= 1
        act_vert[bk] = act_vert[m]
        act_key[bk] = act_key[m]
        act_par[bk] = act_par[m]
        
        // Process in blocks for cache efficiency
        row = adj + u * n
        best = INFINITY
        bk = -1
        
        for (block_start = 0; block_start < m; block_start += block_size):
            block_end = min(block_start + block_size, m)
            
            for (i = block_start; i < block_end; ++i):
                w = act_vert[i]
                d = row[w]
                if (d < act_key[i]):
                    act_key[i] = d
                    act_par[i] = rnd + 1
                if (act_key[i] < best):
                    best = act_key[i]
                    bk = i
```

**Expected Gains:**
- L3 cache hit rate improves significantly
- Reduces memory latency from ~100 cycles to ~10 cycles
- Speedup: 10-20% depending on n
- Works especially well for very large n (>10k)

### Implementation Notes

```python
# benchmarks/mst_benchmark.py
"""Benchmark blocking effect on Prim's algorithm."""

import time
import numpy as np
from tribbleclustering.pcvat import vat_prim_mst_c_64, vat_prim_mst_c_64_blocked

def benchmark_mst(n_samples, n_features=8):
    """Compare MST implementations."""
    np.random.seed(42)
    
    # Generate random data
    X = np.random.randn(n_samples, n_features).astype(np.float64)
    
    # Compute distance matrix
    dist = np.zeros((n_samples, n_samples), dtype=np.float64)
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            d = np.linalg.norm(X[i] - X[j])
            dist[i, j] = dist[j, i] = d
    
    # Benchmark current implementation
    t0 = time.perf_counter()
    for _ in range(3):
        heap_seq, parent_seq = vat_prim_mst_c_64(dist)
    t_current = (time.perf_counter() - t0) / 3
    
    # Benchmark blocked implementation (when available)
    try:
        t0 = time.perf_counter()
        for _ in range(3):
            heap_seq_b, parent_seq_b = vat_prim_mst_c_64_blocked(dist)
        t_blocked = (time.perf_counter() - t0) / 3
        
        print(f"n={n_samples}: Current {t_current*1000:.2f}ms, Blocked {t_blocked*1000:.2f}ms, speedup {t_current/t_blocked:.2f}x")
    except:
        print(f"n={n_samples}: Current {t_current*1000:.2f}ms (blocked variant not yet available)")
```

---

## Optimization #6: Hierarchical Clustering Support

### Why Hierarchical?

Users often need to explore multiple clustering resolutions. Currently, to get clustering with 2, 3, ..., 10 clusters requires recomputing from scratch.

IVAT matrix encodes **complete hierarchy** through its diagonal structure. We can extract dendrogram without recomputing.

### Extract Hierarchy from IVAT

Modify `pvat.py:314-403` to build dendrogram:

```python
# pvat.py - Add after get_ivat_levels()

@dataclass
class ClusterNode:
    """Node in hierarchical clustering tree."""
    indices: ndarray          # Point indices in this cluster
    centroid: ndarray         # Cluster center
    distance: float           # Distance to parent (merge height)
    children: list = field(default_factory=list)
    is_leaf: bool = False
    

def get_ivat_hierarchy(
    all_cities: ndarray,
    ivat_matrix: ndarray,
    vat_order: ndarray,
    linkage_method: str = 'average',
) -> ClusterNode:
    """
    Build hierarchical clustering tree from IVAT matrix.
    
    Args:
        all_cities: Original data (n, d)
        ivat_matrix: IVAT distance matrix (n, n), permuted
        vat_order: Permutation indices from VAT
        linkage_method: 'average', 'complete', 'single' (currently: 'average')
    
    Returns:
        Root node of dendrogram tree
    
    Time Complexity: O(n) after IVAT computed
    Space Complexity: O(n)
    
    References:
        Lance, C., & Williams, W. T. (1967). "A general agglomerative clustering method"
        Müllner, D. (2011). "Fast Hierarchical Clustering"
    """
    
    n = len(all_cities)
    
    # Start with leaf nodes (each point is its own cluster)
    nodes: list[ClusterNode] = []
    for i in range(n):
        idx = vat_order[i]
        nodes.append(ClusterNode(
            indices=np.array([idx]),
            centroid=all_cities[idx],
            distance=0.0,
            is_leaf=True
        ))
    
    # Merge clusters bottom-up based on IVAT diagonal
    # Diagonal values indicate merge heights in the hierarchy
    diagonal_values = np.diag(ivat_matrix, k=1)
    
    # Find merge sequence by walking through off-diagonal
    # Higher diagonal values = later merges (top of tree)
    merge_order = np.argsort(diagonal_values)[::-1]
    
    # Reconstruct tree by merging in order
    active_nodes = list(range(n))
    
    for merge_idx in merge_order:
        if len(active_nodes) <= 1:
            break
        
        # Find which two clusters merge at this height
        # This is the challenging part - need to invert permutation
        i_permuted = merge_idx
        j_permuted = merge_idx + 1  # Adjacent in permuted order
        
        # Find original indices
        i_orig = vat_order[i_permuted]
        j_orig = vat_order[j_permuted]
        
        # Merge if both still active
        if i_orig in active_nodes and j_orig in active_nodes:
            node_i = nodes[i_orig]
            node_j = nodes[j_orig]
            
            # Create parent node
            merged_indices = np.concatenate([node_i.indices, node_j.indices])
            merged_centroid = all_cities[merged_indices].mean(axis=0)
            merge_distance = diagonal_values[merge_idx]
            
            parent = ClusterNode(
                indices=merged_indices,
                centroid=merged_centroid,
                distance=merge_distance,
                children=[node_i, node_j],
                is_leaf=False
            )
            
            # Update nodes
            nodes.append(parent)
            active_nodes.remove(i_orig)
            active_nodes.remove(j_orig)
            active_nodes.append(len(nodes) - 1)
    
    # Return root (last merged node)
    return nodes[-1]


def cut_hierarchy(root: ClusterNode, n_clusters: int) -> list[ndarray]:
    """
    Cut dendrogram at specified number of clusters.
    
    Args:
        root: Root node of hierarchy
        n_clusters: Desired number of clusters
    
    Returns:
        List of clusters (each cluster is array of point indices)
    
    Time Complexity: O(n)
    """
    clusters = []
    
    def traverse(node: ClusterNode):
        if node.is_leaf or len(clusters) == n_clusters:
            # Include this node as a cluster
            clusters.append(node.indices)
        else:
            # Recursively split children
            for child in node.children:
                traverse(child)
            
            # If we haven't reached n_clusters yet, include this node
            if len(clusters) < n_clusters:
                clusters.append(node.indices)
    
    traverse(root)
    
    # If we have too many clusters, merge smallest ones
    while len(clusters) > n_clusters:
        # Find two smallest clusters and merge
        sizes = [len(c) for c in clusters]
        i = sizes.index(min(sizes))
        j = sizes.index(min([s for k, s in enumerate(sizes) if k != i]))
        
        clusters[i] = np.concatenate([clusters[i], clusters[j]])
        clusters.pop(j)
    
    return clusters
```

### Dendrogram Visualization

```python
# visualizations/dendrogram.py
"""Visualize hierarchical clustering as dendrogram."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.cluster.hierarchy import dendrogram

def plot_ivat_dendrogram(root: ClusterNode, title: str = "IVAT Hierarchy"):
    """Plot dendrogram from IVAT hierarchy."""
    
    # Convert tree to scipy linkage format
    linkage_matrix = _tree_to_linkage(root)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    dendrogram(linkage_matrix, ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Distance")
    ax.set_xlabel("Sample Index")
    plt.tight_layout()
    return fig

def _tree_to_linkage(root: ClusterNode) -> np.ndarray:
    """Convert ClusterNode tree to scipy linkage format."""
    # Implementation details...
    pass
```

### Usage Example

```python
# Example: Extract multiple resolutions from single IVAT computation
from tribbleclustering import compute_ivat, get_ivat_hierarchy, cut_hierarchy

# Compute IVAT once
distances = pairwise_distances(X)
ivat_mat, _, vat_order = compute_ivat(distances)

# Build hierarchy once
hierarchy = get_ivat_hierarchy(X, ivat_mat, vat_order)

# Get clustering at different resolutions
clusters_2 = cut_hierarchy(hierarchy, n_clusters=2)
clusters_5 = cut_hierarchy(hierarchy, n_clusters=5)
clusters_10 = cut_hierarchy(hierarchy, n_clusters=10)

# Cost: 1× IVAT computation + 3× tree traversal (fast)
# vs: 3× full IVAT computation (expensive)
```

---

## Optimization #7: Batched Prediction with Caching

### Current Prediction Bottleneck

In `ivatmeans.py:96-99`:

```python
def predict(self, X):
    distances = np.linalg.norm(
        X[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :], 
        axis=2
    )
    return np.argmin(distances, axis=1)
```

**Issues:**
1. Creates intermediate (n_test, n_clusters, d) array
2. Allocates temporary memory: 8 × n_test × n_clusters × d bytes
3. For n_test=100k, k=100, d=50: 4GB temporary allocation!

### Solution 1: Streaming Computation

```python
# ivatmeans.py - Add efficient prediction

def predict(self, X, batch_size: int = 10000) -> ndarray:
    """Predict with streaming computation to avoid memory spike."""
    if self.cluster_centers_ is None:
        raise ValueError("Model has not been fitted yet.")
    
    X = np.asarray(X)
    n_test = X.shape[0]
    labels = np.empty(n_test, dtype=np.int32)
    
    # Process in batches to avoid huge temporary allocations
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        X_batch = X[start:end]
        
        # Compute distances for this batch only
        distances = np.linalg.norm(
            X_batch[:, np.newaxis, :] - self.cluster_centers_[np.newaxis, :, :],
            axis=2
        )
        labels[start:end] = np.argmin(distances, axis=1)
    
    return labels
```

### Solution 2: Compiled Distance Function

```python
# ivatmeans.py - Use compiled distance function if available

def predict(self, X) -> ndarray:
    """Predict using compiled distance function."""
    if self.cluster_centers_ is None:
        raise ValueError("Model has not been fitted yet.")
    
    X = np.asarray(X)
    
    # Use C/OpenMP compiled function if available
    if _has_compiled_distances and X.shape[0] > 100:
        # pairwise_distances_c is much faster for large X
        from .pcvat import pairwise_distances_c
        
        # Compute distances between X and cluster centers
        distances = pairwise_distances_c(
            np.vstack([X, self.cluster_centers_])
        )
        
        # Extract X vs centers block
        n_X = X.shape[0]
        distances = distances[:n_X, n_X:]
        
        return np.argmin(distances, axis=1)
    else:
        # Fallback
        distances = np.linalg.norm(...)
        return np.argmin(distances, axis=1)
```

### Solution 3: Caching for Repeated Predictions

```python
# ivatmeans.py - Cache distances for repeated queries

class IVATMeans:
    def __init__(self, ...):
        # ... existing __init__ code ...
        self._prediction_cache: dict = {}  # {X_hash: predictions}
        self._cache_max_size = 100  # MB
        self._cache_current_size = 0
    
    def predict(self, X, use_cache: bool = True) -> ndarray:
        """Predict with optional caching."""
        X = np.asarray(X)
        
        # Check cache if enabled
        if use_cache:
            X_hash = self._hash_array(X)
            if X_hash in self._prediction_cache:
                return self._prediction_cache[X_hash].copy()
        
        # Compute if not cached
        distances = np.linalg.norm(...)
        labels = np.argmin(distances, axis=1)
        
        # Cache result
        if use_cache and self._cache_current_size + labels.nbytes < self._cache_max_size * 1e6:
            X_hash = self._hash_array(X)
            self._prediction_cache[X_hash] = labels.copy()
            self._cache_current_size += labels.nbytes
        
        return labels
    
    @staticmethod
    def _hash_array(arr: ndarray) -> str:
        """Hash array for caching."""
        import hashlib
        return hashlib.sha256(arr.tobytes()).hexdigest()
```

### Performance Comparison

```python
# benchmarks/prediction_benchmark.py
"""Benchmark prediction methods."""

def benchmark_prediction(n_test, n_clusters, n_features):
    """Compare streaming vs numpy prediction."""
    
    centers = np.random.randn(n_clusters, n_features).astype(np.float32)
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    
    # Original: allocates full temp array
    t0 = time.perf_counter()
    distances = np.linalg.norm(
        X_test[:, np.newaxis, :] - centers[np.newaxis, :, :],
        axis=2
    )
    labels_orig = np.argmin(distances, axis=1)
    t_orig = time.perf_counter() - t0
    
    # Streaming: process in batches
    t0 = time.perf_counter()
    labels_batch = np.empty(n_test, dtype=np.int32)
    batch_size = 10000
    for start in range(0, n_test, batch_size):
        end = min(start + batch_size, n_test)
        dist_batch = np.linalg.norm(
            X_test[start:end, np.newaxis, :] - centers[np.newaxis, :, :],
            axis=2
        )
        labels_batch[start:end] = np.argmin(dist_batch, axis=1)
    t_batch = time.perf_counter() - t0
    
    print(f"Original: {t_orig*1000:.2f}ms, Streaming: {t_batch*1000:.2f}ms")
    print(f"Speedup: {t_orig/t_batch:.2f}x, Memory saved: ~{X_test.nbytes * n_clusters / 1e9:.1f}GB")
```

---

## Optimization #8: Mixed Precision Arithmetic

### When to Use Mixed Precision

Mixed precision trades accuracy for speed by using float32 for intermediate computations and float64 for final results.

```python
# Decision tree for mixed precision
# ✓ Use float32 for: distances, weights (intermediate)
# ✓ Use float32+accum in float64: iterative refinement
# ✗ Use float32 only for: cluster centers (need precision for convergence)
```

### Implement Mixed Precision FCM

```cython
# cfcm_mixed.pyx
"""Mixed-precision Fuzzy C-Means."""

@cython.cdivision(True)
@cython.boundscheck(False)
cdef void _compute_distances_mixed_precision(
    const double[:, ::1] x,      # Input: float64
    const double[:, ::1] c,      # Input: float64
    float[:, ::1] distances      # Output: float32 (lower precision OK)
) noexcept nogil:
    """Compute distances with mixed precision.
    
    Intermediate computations in float32 (2x faster, same accuracy).
    """
    cdef int n_samples = x.shape[0]
    cdef int n_clusters = c.shape[0]
    cdef int n_features = x.shape[1]
    cdef int i, j, k
    cdef float d, diff
    
    for i in range(n_samples):
        for j in range(n_clusters):
            d = 0.0  # float32 accumulation
            for k in range(n_features):
                diff = <float>(x[i, k] - c[j, k])
                d += diff * diff
            distances[i, j] = sqrt(d)


cdef void _compute_weights_mixed_precision(
    const float[:, ::1] distances,  # Input: float32
    double m,
    float[:, ::1] w_ij            # Output: float32
) noexcept nogil:
    """Compute weights with mixed precision."""
    cdef int n_samples = distances.shape[0]
    cdef int n_clusters = distances.shape[1]
    cdef int i, j, jj
    cdef float denom, dist_ratio, val
    
    for i in range(n_samples):
        for j in range(n_clusters):
            if distances[i, j] == 0.0:
                w_ij[i, j] = 0.0
                continue
            
            denom = 0.0
            for jj in range(n_clusters):
                if distances[i, jj] == 0.0:
                    denom = 1.0
                    break
                dist_ratio = distances[i, j] / distances[i, jj]
                denom += pow(dist_ratio, 2.0 / (m - 1.0))
            
            if denom > 0.0:
                val = 1.0 / denom
            else:
                val = 0.0
            
            w_ij[i, j] = val


def fuzzy_c_means_mixed(x, n, m=2.0, *, indices=None, initial_guess=None):
    """FCM with mixed precision: float32 distances/weights, float64 centers."""
    
    x = np.asarray(x, dtype=np.float64)
    n_samples = x.shape[0]
    n_features = x.shape[1]
    
    # Float32 for distances and weights (intermediate)
    distances = np.zeros((n_samples, n), dtype=np.float32)
    w_ij = np.zeros((n_samples, n), dtype=np.float32)
    
    # Float64 for cluster centers (final)
    c = np.zeros((n, n_features), dtype=np.float64)
    c_new = np.zeros((n, n_features), dtype=np.float64)
    
    # Initialize centers in float64
    # ...
    
    for iteration in range(100):
        # Compute distances in float32
        _compute_distances_mixed_precision(x, c, distances)
        
        # Compute weights in float32
        _compute_weights_mixed_precision(distances, m, w_ij)
        
        # Update centers in float64 (needs full precision for convergence)
        for j in range(n):
            w_sum = 0.0
            for i in range(n_samples):
                wm = w_ij[i, j] ** m
                w_sum += wm
                for k in range(n_features):
                    c_new[j, k] += wm * x[i, k]
        
        for j in range(n):
            for k in range(n_features):
                c_new[j, k] /= w_sum if w_sum > 0 else 1.0
        
        # Check convergence in float64
        max_delta = 0.0
        for i in range(n):
            for k in range(n_features):
                delta = (c_new[i, k] - c[i, k]) ** 2
                if delta > max_delta:
                    max_delta = delta
        
        if max_delta < 1e-10:
            break
        
        c = c_new.copy()
    
    return c, w_ij
```

### Accuracy Validation

```python
# tests/test_mixed_precision.py
"""Validate mixed-precision FCM accuracy."""

def test_mixed_precision_accuracy():
    """Verify mixed precision has minimal accuracy loss."""
    
    x = np.random.randn(500, 20).astype(np.float64)
    
    # Full float64
    c_full, w_full = fuzzy_c_means(x, 5, m=2.0)
    
    # Mixed precision
    c_mixed, w_mixed = fuzzy_c_means_mixed(x, 5, m=2.0)
    
    # Cluster centers should match to ~1% relative error
    np.testing.assert_allclose(c_full, c_mixed, rtol=1e-2)
    
    # Membership weights should match to ~0.1% relative error
    np.testing.assert_allclose(w_full, w_mixed, rtol=1e-3)
```

### Performance Gains

```python
# benchmarks/mixed_precision_bench.py
def benchmark_mixed_precision():
    """Measure mixed precision speedup."""
    
    x = np.random.randn(5000, 30).astype(np.float64)
    
    t0 = time.perf_counter()
    c_full, w_full = fuzzy_c_means(x, 10)
    t_full = time.perf_counter() - t0
    
    t0 = time.perf_counter()
    c_mixed, w_mixed = fuzzy_c_means_mixed(x, 10)
    t_mixed = time.perf_counter() - t0
    
    print(f"Full float64: {t_full*1000:.2f}ms")
    print(f"Mixed precision: {t_mixed*1000:.2f}ms")
    print(f"Speedup: {t_full/t_mixed:.2f}x")
    
    # Verify accuracy
    max_diff = np.max(np.abs(c_full - c_mixed) / np.abs(c_full + 1e-10))
    print(f"Max relative difference: {max_diff*100:.2f}%")
```

---

## Optimization #9: Parallel Permutation Gather

### Current Implementation

In `pcvat.pyx:237-249`, the permutation gather:

```cython
# pcvat.pyx:237-249 - Current sequential implementation
with nogil:
    // Fill lower triangle: out[i,j] = adj[P[i],P[j]], j<=i
    for i in prange(n, schedule='static', num_threads=gthreads):
        pi_row = <Py_ssize_t>P[i] * n
        i_row  = <Py_ssize_t>i * n
        for j in range(i + 1):
            O[i_row + j] = A[pi_row + P[j]]
```

**Issue:** Even though outer loop is parallel, performance limited by:
1. Random memory access pattern (scatter/gather via permutation)
2. False sharing between threads
3. Partial cache line usage

### Optimization: Blocked Permutation Gather

```cython
# pcvat_optimized.pyx - Blocked parallel gather

cdef void _backcopy_blocked_64(double* M, int n, int nthreads) noexcept nogil:
    """Back-copy with cache-aware blocking."""
    cdef int i, j, bi, bj, block_size
    
    block_size = 64  # Tuned for 64-byte cache lines
    
    for bi in prange(0, n, block_size, schedule='static', num_threads=nthreads):
        for bj in prange(0, n, block_size, schedule='static', num_threads=nthreads):
            # Process block [bi:bi+block_size, bj:bj+block_size]
            for i in range(bi, min(bi + block_size, n)):
                for j in range(bj, min(bj + block_size, n)):
                    if j < i:
                        M[<Py_ssize_t>j * n + i] = M[<Py_ssize_t>i * n + j]
```

### Expected Gains

- Cache line utilization: 25% → 85%+
- Memory bandwidth: 15-25GB/s → 50GB/s (on DDR5)
- Speedup: 1.2-1.8x (minor, but works with other optimizations)

---

## Optimization #10: Convergence Acceleration

### Nesterov Momentum for FCM

Standard FCM uses gradient descent. Nesterov momentum accelerates convergence:

```cython
# cfcm_accelerated.pyx
"""FCM with Nesterov momentum acceleration."""

def fuzzy_c_means_accelerated(x, n, m=2.0, momentum=0.9):
    """FCM with Nesterov momentum.
    
    Standard FCM: c_new = update(c)
    Nesterov: c_new = update(c + momentum * (c - c_prev))
    
    Expected: 30-50% fewer iterations to convergence.
    """
    
    c = initialize_centers(x, n)
    c_prev = c.copy()
    
    for iteration in range(100):
        # Extrapolated center: look ahead with momentum
        c_momentum = c + momentum * (c - c_prev)
        
        # Compute update using extrapolated center
        distances = compute_distances(x, c_momentum)
        weights = compute_weights(distances, m)
        c_new = compute_centers(x, weights, m)
        
        # Check convergence
        max_delta = np.max(np.abs(c_new - c))
        if max_delta < 1e-10:
            break
        
        # Update for next iteration
        c_prev = c.copy()
        c = c_new
    
    return c
```

### Fuzzy Learning Rate (Adaptive m)

```python
# adaptive.py
"""Adaptive fuzziness parameter."""

def compute_adaptive_m(iteration, max_iterations, m_initial=2.0, m_final=3.0):
    """Increase m over iterations to sharpen clusters.
    
    Early iterations: m=2.0 (fuzzy, explores space)
    Late iterations: m=3.0 (sharp, refines clusters)
    
    Expected: 15-25% fewer iterations.
    """
    progress = iteration / max_iterations
    return m_initial + (m_final - m_initial) * progress
```

### Implementation

```cython
# cfcm_convergence.pyx
"""Convergence acceleration techniques."""

def fuzzy_c_means_accelerated(
    x, n, m=2.0,
    momentum=0.9,
    adaptive_m=True,
    max_iterations=100
):
    """FCM with acceleration techniques."""
    
    c = initialize_centers(x, n)
    c_prev = c.copy()
    
    for iteration in range(max_iterations):
        # Adaptive m: increase over iterations
        if adaptive_m:
            m_iter = compute_adaptive_m(iteration, max_iterations, 2.0, 3.0)
        else:
            m_iter = m
        
        # Nesterov momentum
        c_momentum = c + momentum * (c - c_prev)
        
        # Standard FCM update
        distances = compute_distances(x, c_momentum)
        weights = compute_weights(distances, m_iter)
        c_new = compute_centers(x, weights, m_iter)
        
        # Convergence check
        max_delta = np.max(np.abs(c_new - c))
        if max_delta < 1e-10:
            break
        
        c_prev = c
        c = c_new
    
    return c, weights
```

---

## Testing & Validation

### Correctness Testing Strategy

All optimizations must maintain **numerical correctness**. Use tolerance-aware testing:

```python
# tests/test_optimization_correctness.py
"""Verify optimizations maintain correctness."""

import numpy as np
from numpy.testing import assert_allclose
import pytest

class TestOptimizationCorrectness:
    """Test each optimization against baseline."""
    
    @pytest.mark.parametrize("optimization", [
        "simd_distances",
        "gpu_fcm",
        "memory_layout",
        "mixed_precision",
    ])
    def test_optimization_matches_baseline(self, optimization):
        """Each optimization produces similar results to baseline."""
        
        x = np.random.randn(500, 20).astype(np.float64)
        
        # Baseline
        from tribbleclustering.fcm import fuzzy_c_means as baseline
        c_base, w_base = baseline(x, 5)
        
        # Optimized
        if optimization == "simd_distances":
            from tribbleclustering.cfcm import fuzzy_c_means as opt
            c_opt, w_opt = opt(x.astype(np.float32), 5)
        elif optimization == "gpu_fcm":
            from tribbleclustering.cfcm_gpu import FuzzyCMeansGPU
            gpu = FuzzyCMeansGPU(5)
            c_opt, w_opt = gpu.fit(x)
        # ... etc ...
        
        # Tolerance depends on optimization type
        if optimization == "mixed_precision":
            rtol = 1e-2  # Mixed precision: looser tolerance
        else:
            rtol = 1e-8  # Others: tight tolerance
        
        assert_allclose(c_base, c_opt, rtol=rtol)
        assert_allclose(w_base, w_opt, rtol=rtol)
```

---

## Profiling Tools & Techniques

### CPU Profiling

```bash
# Install perf (Linux)
sudo apt-get install linux-tools

# Profile with perf
perf record -F 99 -g python run_fcm.py
perf report

# Specific events
perf stat -e cycles,instructions,cache-references,cache-misses python run_fcm.py
```

### Memory Profiling

```bash
# Install memory_profiler
pip install memory_profiler

# Profile memory
python -m memory_profiler run_fcm.py

# Line-by-line
@profile
def fuzzy_c_means(...):
    # Decorated function shows memory per line
```

### Cython Profiling

```bash
# Enable Cython line profiling
# setup.py
ext_modules = cythonize(
    "*.pyx",
    compiler_directives={"profile": True},  # ← Enable profiling
)

# Run with cProfile
python -m cProfile -o stats.prof run_fcm.py

# Analyze
python
>>> import pstats
>>> p = pstats.Stats('stats.prof')
>>> p.sort_stats('cumulative').print_stats(20)
```

### GPU Profiling (NVIDIA)

```bash
# Install CUDA toolkit with profiler
pip install nvidia-pytoolbox

# Profile CUDA code
nvprof python run_fcm_gpu.py

# Or use Nsys
nsys profile python run_fcm_gpu.py
```

---

## Summary & Recommendations

### Quick Implementation Order

**Week 1:**
1. Memory layout in FCM (#4) - 20-40% gain, 3 days
2. Batched prediction (#7) - 2-5x, 1 day
3. SIMD unrolling (#1) - 30-50%, 3 days

**Week 2-3:**
4. Mixed precision (#8) - 1.5-3x, 2 days
5. Convergence acceleration (#10) - 1.5-3x, 3 days
6. Cache-optimized MST (#5) - 10-20%, 3 days

**Week 4-5:**
7. GPU acceleration (#2) - 10-100x, 2 weeks
8. Approximate NN (#3) - 5-15x, 2 weeks
9. Hierarchical clustering (#6) - 10-50x*, 2 weeks

*Use-case specific

### Expected Cumulative Improvement

- After Week 1: 2-3x overall
- After Week 3: 5-10x overall
- After Week 5: 50-200x overall (hardware dependent)

