import numpy as np
import typing as T


def triu(x: np.ndarray) -> np.ndarray:
    "Extracts upper triangular part of a matrix, excluding the diagonal."
    ones  = np.ones_like(x)
    return x[np.triu(ones, k=1) == 1]

