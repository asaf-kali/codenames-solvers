from typing import Sequence, Tuple, Union

import numpy as np
import pandas as pd

Array = Union[np.ndarray, Sequence[np.ndarray]]


def cosine_similarity_with_vectors(u: np.ndarray, v: Sequence[np.ndarray]) -> np.ndarray:  # type: ignore
    u = normalize_vector(u)
    v_list = [normalize_vector(vec) for vec in v]
    return np.array([u.T @ vec for vec in v_list])


def cosine_similarity_with_vector(u: np.ndarray, v: np.ndarray) -> float:  # type: ignore
    u = normalize_vector(u)
    v = normalize_vector(v)
    return u.T @ v


def cosine_similarity(
    u: np.ndarray, v: Array  # type: ignore
) -> Union[float, np.ndarray]:  # type: ignore
    if isinstance(v, pd.Series):
        return cosine_similarity_with_vectors(u, v)
    else:
        return cosine_similarity_with_vector(u, v)  # type: ignore


def cosine_distance(u: np.ndarray, v: Array) -> Union[np.ndarray, float]:  # type: ignore  # noqa: E501
    return (1 - cosine_similarity(u, v)) / 2  # type: ignore


def normalize_vector(v):
    r = np.linalg.norm(v)
    if r == 0:
        return v
    else:
        return v / r


def single_gram_schmidt(v: np.ndarray, u: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:  # type: ignore
    v = normalize_vector(v)
    u = normalize_vector(u)

    projection_norm = u.T @ v

    o = u - projection_norm * v

    normed_o = normalize_vector(o)
    return v, normed_o


def geodesic(v, u):
    r = np.linalg.norm(v)
    v, u = normalize_vector(v), normalize_vector(u)

    v, normed_o = single_gram_schmidt(v, u)
    theta = np.arccos(np.clip(v.T @ u, -1.0, 1.0))

    def f(t):
        return (np.cos(t * theta) * v + np.sin(t * theta) * normed_o) * r

    return f


def vec_to_rotation(v, t):
    v = normalize_vector(v)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    R = np.array(
        [
            [
                cos_t + (1 - cos_t) * v[0] ** 2,
                v[0] * v[1] * (1 - cos_t) - v[2] * sin_t,
                v[0] * v[2] * (1 - cos_t) + v[1] * sin_t,
            ],
            [
                v[0] * v[1] * (1 - cos_t) + v[2] * sin_t,
                cos_t + (1 - cos_t) * v[1] ** 2,
                v[1] * v[2] * (1 - cos_t) + v[0] * sin_t,
            ],
            [
                v[0] * v[2] * (1 - cos_t) + v[1] * sin_t,
                v[1] * v[2] * (1 - cos_t) + v[0] * sin_t,
                cos_t + (1 - cos_t) * v[2] ** 2,
            ],
        ]
    )
    return R
