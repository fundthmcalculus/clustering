import itertools
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from numpy import dtype, ndarray

from tribbleclustering.util import circle_random_clusters
from tribbleclustering.pvat import compute_vat


def confine_orientation(cities: ndarray) -> ndarray:
    # Find the longest distance between two cities
    distances = np.linalg.norm(cities[:, None, :] - cities[None, :, :], axis=-1)
    i, j = np.unravel_index(np.argmax(distances), distances.shape)

    # That's the principal axis, rotate to align with x-axis
    axis_vector = cities[j] - cities[i]
    angle = np.arctan2(axis_vector[1], axis_vector[0])
    rotation_matrix = np.array([[np.cos(-angle), -np.sin(-angle)],
                                [np.sin(-angle), np.cos(-angle)]])
    cities = cities @ rotation_matrix.T

    # Then translate to put the lower-left corner of the bounding box at the origin.
    cities -= cities.min(axis=0)
    return cities


def main():
    # Get 10 random city locations.
    N = 50
    p = 7
    N = (N // p) * p
    cities = circle_random_clusters(N // p, p)
    cities = cities[np.random.permutation(len(cities))]
    cities = confine_orientation(cities)
    # cities = np.hstack((cities, np.random.rand(len(cities), 1)))

    d = np.zeros((len(cities), len(cities)))
    for ij in itertools.combinations(range(len(cities)), 2):
        i, j = ij
        d[i, j] = np.linalg.norm(cities[i] - cities[j])
        d[j, i] = d[i, j]

    # Apply VAT reordering to the distance matrix
    d_vat, order = compute_vat(d)
    new_cities = trilinear_cities(N, d_vat)
    new_cities = confine_orientation(new_cities)

    # Compute dissimilarity matrix of new cities
    d_new = np.linalg.norm(new_cities[:, None, :] - new_cities[None, :, :], axis=-1)

    # Plot the dissimilarity matrix
    fig, ax = plt.subplots(2, 2, figsize=(12, 10))
    im = ax[0, 0].imshow(d, cmap='viridis', aspect='auto')
    plt.colorbar(im, ax=ax[0, 0], label='Dissimilarity')
    ax[0, 0].set_title('Original Dissimilarity Matrix')
    ax[0, 0].set_xlabel('Index')
    ax[0, 0].set_ylabel('Index')

    im2 = ax[0, 1].imshow(d_new, cmap='viridis', aspect='auto')
    plt.colorbar(im2, ax=ax[0, 1], label='Dissimilarity')
    ax[0, 1].set_title('Reconstructed Dissimilarity Matrix')
    ax[0, 1].set_xlabel('Index')
    ax[0, 1].set_ylabel('Index')

    # Plot the cities
    ax[1, 0].scatter(cities[:, 0], cities[:, 1], marker='+', s=100, c='blue', label='Actual cities')
    ax[1, 0].scatter(new_cities[:, 0], new_cities[:, 1], marker='o', s=50, c='red', label='Reconstructed cities')
    ax[1, 0].set_title('City Locations')
    ax[1, 0].set_xlabel('X coordinate')
    ax[1, 0].set_ylabel('Y coordinate')
    ax[1, 0].legend()

    # Plot the difference between dissimilarity matrices
    diff = np.abs(d - d_new)
    im3 = ax[1, 1].imshow(diff, cmap='hot', aspect='auto')
    plt.colorbar(im3, ax=ax[1, 1], label='Absolute Difference')
    ax[1, 1].set_title('Dissimilarity Matrix Difference')
    ax[1, 1].set_xlabel('Index')
    ax[1, 1].set_ylabel('Index')
    plt.show()


def trilinear_cities(N: int, d) -> ndarray[tuple[Any, ...], dtype[Any]]:
    # ASSUME the first city is at the origin, next one at (d10, 0)
    new_cities = [np.zeros(2), (d[0, 1], 0)]

    # Handle the 3rd city (index 2) - place it in the positive half-plane
    d1 = d[0, 2]
    d2 = d[1, 2]
    d12 = d[0, 1]
    x1, y1 = new_cities[0]
    x2, y2 = new_cities[1]
    # Calculate x-coordinate using the law of cosines
    x = x1 + (d1 ** 2 - d2 ** 2 + d12 ** 2) / (2 * d12) * (x2 - x1) / d12
    # Calculate y-coordinate (positive half-plane)
    disc = d1 ** 2 - ((x - x1) ** 2 + (y1) ** 2)
    y_offset = np.sqrt(np.abs(disc))
    y = y1 + y_offset  # Always choose positive half-plane for 3rd city
    new_cities.append((x, y))

    # Now, look at the dissimilarity matrix and calculate the locations of the other cities.
    for ij in range(3, N):
        d1 = d[ij - 3, ij]
        d2 = d[ij - 2, ij]
        d3 = d[ij - 1, ij]

        x1, y1 = new_cities[ij - 3]
        x2, y2 = new_cities[ij - 2]
        x3, y3 = new_cities[ij - 1]

        # Use trilateration with the three reference cities
        # Solve: (x - x1)^2 + (y - y1)^2 = d1^2
        #        (x - x2)^2 + (y - y2)^2 = d2^2
        #        (x - x3)^2 + (y - y3)^2 = d3^2

        # Build the system matrix
        A = np.array([
            [2 * (x2 - x1), 2 * (y2 - y1)],
            [2 * (x3 - x1), 2 * (y3 - y1)]
        ])

        b = np.array([
            d1 ** 2 - d2 ** 2 + x2 ** 2 - x1 ** 2 + y2 ** 2 - y1 ** 2,
            d1 ** 2 - d3 ** 2 + x3 ** 2 - x1 ** 2 + y3 ** 2 - y1 ** 2
        ])

        try:
            coords = np.linalg.solve(A, b)
            x, y = coords[0], coords[1]
        except np.linalg.LinAlgError:
            # Degenerate case: use mean of reference cities
            x, y = (x1 + x2 + x3) / 3, (y1 + y2 + y3) / 3

        new_cities.append((x, y))

    new_cities = np.array(new_cities)
    return new_cities


if __name__ == "__main__":
    main()