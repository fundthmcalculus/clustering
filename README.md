# Clustering Package

An optimized implementation of VAT/IVAT, including priority-queue MST speedups as discussed at the NAFIPS 2025/2026 conferences. In addition, there are now C-based SIMD extensions which can improve the performance again by a factor of 15-20.

## Installation

```bash
pip install tribble-clustering
```

## Usage
For fuzzy-c-means:
```python
from tribbleclustering import fuzzy_c_means
import numpy as np

data = np.array([[1, 2], [2, 3], [10, 11], [11, 12]])
membership, centers = fuzzy_c_means(data, n=2, m=2.0)
print(f"Cluster centers: {centers}")
print(f"Membership matrix:\n{membership}")
```

For IVAT:

```python
from tribbleclustering import compute_ivat
from tribbleclustering.util import circle_random_clusters, pairwise_distances

cluster_cities = circle_random_clusters(10, 2, 10)
city_distances = pairwise_distances(cluster_cities)
print(compute_ivat(city_distances))
```

For K-Means (scikit-learn-compatible interface, optional GPU acceleration):

```python
from tribbleclustering import KMeans
import numpy as np

data = np.array([[1, 2], [2, 3], [10, 11], [11, 12]])
model = KMeans(n_clusters=2, random_state=0).fit(data)
print(f"Cluster centers: {model.cluster_centers_}")
print(f"Labels: {model.labels_}")
```

For ConiVAT (semi-supervised iVAT with must-link/cannot-link constraints):

```python
from tribbleclustering import ConiVAT
import numpy as np

data = np.array([[1, 2], [2, 3], [10, 11], [11, 12]])
labels = np.array([0, 0, 1, 1])  # used only to sample constraints
model = ConiVAT(n_clusters=2).fit(data, y=labels)
print(f"Cluster centers: {model.cluster_centers_}")
print(f"Labels: {model.labels_}")
```

## GPU acceleration (optional)

Installing the `gpu` extra (`pip install tribble-clustering[gpu]`) enables
CuPy-backed, device-resident kernels for pairwise distances, FCM, k-means, and
VAT/MST (`tribbleclustering.gpu`, `tribbleclustering.gpu_vat`). These are not
re-exported from the top-level package — import them directly and guard with
`gpu.is_available()`, which falls back to `False` (and callers should fall
back to the CPU path) when no CUDA device is present:

```python
from tribbleclustering import gpu

if gpu.is_available():
    ...  # use gpu.pairwise_distances_gpu / gpu.fuzzy_c_means_gpu / gpu.kmeans_gpu
```
