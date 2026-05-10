# Clustering Package

A simple Python package for clustering tasks.

## Installation

```bash
pip install clustering-pkg
```

## Usage

```python
from clustering import simple_cluster

data = [1, 2, 10, 11]
result = simple_cluster(data, threshold=5)
print(result) # [[1, 2], [10, 11]]
```
