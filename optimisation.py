import jax.numpy as np
from jax.tree_util import tree_multimap

import sd
import utils


# NB: I can only reproduce the Mollweide plots by introducing a correction
# to the third entry, i.e. instead of \mu_3 = (0.6, 0.5) as claimed in paper,
# we set it to \mu_3 = (0.6, 5.).
# Domain adjustment is simply an artefact of our choice of range of \phi.
target_mu = utils.spherical_to_euclidean(np.array([
    [1.5, 0.7 + np.pi / 2],
    [1., -1. + np.pi / 2],
    [5., 0.6 + np.pi / 2],  # 0.5 -> 5.!
    [4., -0.7 + np.pi / 2]
]))


def s2_target(x):
    xe = np.dot(x, target_mu.T)
    return np.sum(np.exp(10 * xe), axis=1)


def update_fn(param, update):
    if param.ndim <= 1:
        return param + update
    else:
        projected_update = update - param * np.sum(param * update, axis=0)
        return sd.exponential_map_sd(param, projected_update)


def apply_updates(params, updates):
    return tree_multimap(update_fn, params, updates)


def kl_ess(log_model_prob, target_prob):
    weights = target_prob / np.exp(log_model_prob)
    Z = np.mean(weights)
    KL = np.mean(log_model_prob - np.log(target_prob)) + np.log(Z)
    ESS = np.sum(weights) ** 2 / np.sum(weights ** 2)
    return Z, KL, ESS
