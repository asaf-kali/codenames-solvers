from typing import Sequence, Union, Tuple

import numpy as np
import pandas as pd


def cosine_similarity_with_vectors(u: np.ndarray, v: Sequence[np.ndarray]) -> np.ndarray:  # type: ignore
    u = u / np.linalg.norm(u)
    v_list = [vec / np.linalg.norm(vec) for vec in v]
    return np.array([u.T @ vec for vec in v_list])


def cosine_similarity_with_vector(u: np.ndarray, v: np.ndarray) -> float:  # type: ignore
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return u.T @ v


def cosine_similarity(
    u: np.ndarray, v: Union[np.ndarray, Sequence[np.ndarray]]  # type: ignore
) -> Union[float, np.ndarray]:  # type: ignore
    if isinstance(v, pd.Series):
        return cosine_similarity_with_vectors(u, v)
    else:
        return cosine_similarity_with_vector(u, v)  # type: ignore


def cosine_distance(u: np.ndarray, v: Union[np.ndarray, Sequence[np.ndarray]]) -> np.ndarray:  # type: ignore
    return (1 - cosine_similarity(u, v)) / 2  # type: ignore


def single_gram_schmidt(v: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
    v = v / np.linalg.norm(v)
    u = u / np.linalg.norm(u)

    projection_norm = u.T @ v

    o = u - projection_norm * v

    normed_o = o / np.linalg.norm(o)
    return v, normed_o
