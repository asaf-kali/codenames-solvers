from typing import Sequence, Union

import numpy as np
import pandas as pd


def cosine_similarity_with_vectors(u: np.ndarray, v: Sequence[np.array]) -> np.array:  # type: ignore
    u = u / np.linalg.norm(u)
    v_list = [vec / np.linalg.norm(vec) for vec in v]
    return np.array([u.T @ vec for vec in v_list])


def cosine_similarity_with_vector(u: np.ndarray, v: np.ndarray) -> float:  # type: ignore
    u = u / np.linalg.norm(u)
    v = v / np.linalg.norm(v)
    return u.T @ v


def cosine_similarity(u: np.ndarray, v: Union[np.array, Sequence[np.array]]) -> Union[float, np.array]:  # type: ignore
    if isinstance(v, pd.Series):
        return cosine_similarity_with_vectors(u, v)
    else:
        return cosine_similarity_with_vector(u, v)  # type: ignore


def cosine_distance(u: np.ndarray, v: Union[np.array, Sequence[np.array]]) -> np.array:  # type: ignore
    return (1 - cosine_similarity(u, v)) / 2
