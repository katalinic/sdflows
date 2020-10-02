import jax.numpy as np
from jax import random


def exponential_map_sd(x, v):
    r"""Exponential map on \mathbb{S}^D.

    Args:
        x: points on \mathbb{S}^D, embedded in \mathbb{R}^{D+1}
        v: vectors in the tangent space T_x \mathbb{S}^D

    Returns:
        Image of exponential map
    """
    v_norm = np.linalg.norm(v, axis=1, keepdims=True)
    return x * np.cos(v_norm) + v / v_norm * np.sin(v_norm)


def sample_sd(rng, d=2, num_samples=1):
    r"""Uniformly sample points on \mathbb{S}^d."""
    xs = random.normal(rng, shape=(num_samples * (d + 1),)).reshape(-1, d + 1)
    xs /= np.linalg.norm(xs, axis=1, keepdims=True)
    return xs
