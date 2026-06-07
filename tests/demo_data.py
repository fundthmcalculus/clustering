import time

import numpy as np
import pandas as pd
from PIL import Image
from ucimlrepo import fetch_ucirepo
from tribbleclustering.pcvat import compute_vat_c, pairwise_distances_c

print("\n")


def to_np_array(data: pd.DataFrame) -> np.ndarray:
    y = data.to_numpy(np.float32)
    return y


# fetch dataset
# 59 is letter recognition
# 827 is sepsis survival (allocates 80+ GB RAM)
# 148 is shuttle stat log (allocates 50 GB RAM)
dataset_id = 148
letter_recognition = fetch_ucirepo(id=dataset_id)

# data (as pandas dataframes)
X = letter_recognition.data.features
l_x = len(X)
X = to_np_array(X)

# metadata
print(f"Metadata: {letter_recognition.metadata}")

# variable information
print(f"Variable Information: {letter_recognition.variables}")

# Compute the pairwise distances - float32 for space-saving.
t0 = time.perf_counter()
matrix_of_pairwise_distance = pairwise_distances_c(X)
print(f"Time to pairwise distances: {time.perf_counter() - t0:.02f}")
del X
matrix_of_pairwise_distance = np.log(matrix_of_pairwise_distance + 1).astype(np.float32)
matrix_of_pairwise_distance = (
    matrix_of_pairwise_distance / matrix_of_pairwise_distance.max()
)
print(f"Pairwise distance matrix shape: {matrix_of_pairwise_distance.shape}")
t0 = time.perf_counter()
ordered_matrix, _, _ = compute_vat_c(matrix_of_pairwise_distance)
t1 = time.perf_counter()

print(f"Elapsed time for {l_x} data points: {t1 - t0:.02f}")

# Save the ordered matrix as an image
img_array = (ordered_matrix * 255).astype(np.uint8)
img = Image.fromarray(img_array)
img.save(f"ordered_matrix_{dataset_id}.png")
t2 = time.perf_counter()
print(f"Elapsed time for {l_x} data points image saving: {t2 - t1:.02f}")
