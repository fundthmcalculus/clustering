# Clustering Package

A simple Python package for clustering tasks.

## Installation

```bash
pip install clustering-pkg
```

## Usage
For fuzzy-c-means:
```python
from clustering import fuzzy_c_means
import numpy as np

data = np.array([[1, 2], [2, 3], [10, 11], [11, 12]])
membership, centers = fuzzy_c_means(data, n=2, m=2.0)
print(f"Cluster centers: {centers}")
print(f"Membership matrix:\n{membership}")
```

For IVAT:

```python
from clustering import compute_ivat
from clustering.util import circle_random_clusters, pairwise_distances

cluster_cities = circle_random_clusters(10, 2, 10)
city_distances = pairwise_distances(cluster_cities)
print(compute_ivat(city_distances))
```