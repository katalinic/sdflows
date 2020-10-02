import jax.numpy as np


def spherical_to_euclidean(sph_coords):
    if sph_coords.ndim == 1:
        sph_coords = np.expand_dims(sph_coords, 0)
    theta, phi = np.split(sph_coords, 2, 1)
    return np.concatenate((
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ), 1)


def euclidean_to_spherical(euc_coords):
    if euc_coords.ndim == 1:
        euc_coords = np.expand_dims(euc_coords, 0)
    x, y, z = np.split(euc_coords, 3, 1)
    return np.concatenate((
        np.pi + np.arctan2(-y, -x),
        np.arccos(z)
    ), 1)


def softplus(x):
    return np.logaddexp(x, 0.)


def softplus_inv(x):
    return np.log(-1. + np.exp(x))


def softmax(x):
    ex = np.exp(x - np.max(x))
    return ex / np.sum(ex)
