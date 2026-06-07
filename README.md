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
