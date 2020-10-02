import jax.numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import utils

NUM_POINTS = 100

theta = np.linspace(0, 2 * np.pi, 2 * NUM_POINTS)
phi = np.linspace(0, np.pi, NUM_POINTS)
tp = np.array(np.meshgrid(theta, phi, indexing='ij'))
tp = tp.transpose([1, 2, 0]).reshape(-1, 2)


def plot_model_density(model_samples):
    estimated_density = gaussian_kde(
        utils.euclidean_to_spherical(model_samples).T, 0.2)
    heatmap = estimated_density(tp.T).reshape(2 * NUM_POINTS, NUM_POINTS)
    _plot_mollweide(heatmap)


def plot_target_density(target_fn):
    density = target_fn(utils.spherical_to_euclidean(tp))
    heatmap = density.reshape(2 * NUM_POINTS, NUM_POINTS)
    _plot_mollweide(heatmap)


def _plot_mollweide(heatmap):
    tt, pp = np.meshgrid(theta - np.pi, phi - np.pi / 2, indexing='ij')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='mollweide')
    ax.pcolormesh(tt, pp, heatmap, cmap=plt.cm.jet)
    ax.set_axis_off()
    plt.show()
