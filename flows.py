import jax.numpy as np
from jax import random

import sd
import utils


def ExponentialMapSumRadialFlow(K, D):
    r"""Constructor for an exponential-map-sum-of-radial-flow transformation.

    Args:
        K (int): number of radial components
        D (int): dimensionality of \mathbb{S}^D

    Returns:
        init_fun: takes an rng key and returns initial parameters
        apply_fun: takes params, inputs and returns the transformed inputs
            and log determinant Jacobian of the transformation
    """
    def init_fun(rng):
        mu_key, beta_key = random.split(rng)
        param = {
            'beta': utils.softplus_inv(
                random.uniform(beta_key, shape=(K,), minval=1.0, maxval=2.0)),
            'mu': sd.sample_sd(mu_key, D, K).T
        }
        if K > 1:
            param['alpha'] = np.ones(K) / K
        return param

    def apply_fun(param, inputs, **kwargs):
        assert inputs.ndim == 2
        assert inputs.shape[1] == D + 1
        _beta, mu = param['beta'], param['mu']

        # Enforce constraints of $\sum_k \alpha_k \leq 1; \forall \beta_k > 0$
        beta = utils.softplus(_beta)
        alpha = utils.softmax(param.get('alpha', np.ones_like(beta)))

        # Euclidean gradient and Jacobian of scalar field.
        exp_factor = np.exp(beta * (np.matmul(inputs, mu) - 1))
        dF = np.matmul(exp_factor * alpha, mu.T)
        ddF = np.einsum('ndk,pk->npd',
                        np.einsum('nk,dk->ndk', exp_factor, alpha * beta * mu),
                        mu)

        # Project into tangent space.
        Id = np.eye(D + 1)
        proj = Id - np.einsum('ni,nj->nij', inputs, inputs)
        v = np.einsum('nij,nj->ni', proj, dF)

        norm_v = np.linalg.norm(v, axis=1, keepdims=True)
        v_div_norm_v = v / norm_v
        cos_norm_v = np.cos(norm_v)
        sin_div_norm_v = np.sin(norm_v) / norm_v

        # Output of transformation (exponential map).
        z = inputs * cos_norm_v + v * sin_div_norm_v

        # Jacobian of projected gradient of scalar field.
        Dv = np.matmul(proj, ddF) - np.einsum('ni,nj->nij', inputs, dF) - \
            np.sum(inputs * dF, axis=1)[:, np.newaxis, np.newaxis] * Id

        # Jacobian of map.
        vvT = np.einsum('ni,nj->nij', v_div_norm_v, v_div_norm_v)
        xvT = np.einsum('ni,nj->nij', inputs, v)
        J = cos_norm_v[..., np.newaxis] * Id[np.newaxis, ...]
        J += sin_div_norm_v[..., np.newaxis] * Dv
        J += np.einsum('nij,njk->nik',
                       + cos_norm_v[..., np.newaxis] * vvT
                       - sin_div_norm_v[..., np.newaxis] * vvT
                       - sin_div_norm_v[..., np.newaxis] * xvT,
                       Dv)

        # ONB of tangent space.
        E = np.dstack([
            v_div_norm_v,
            np.cross(inputs, v_div_norm_v)
        ])
        JE = np.matmul(J, E)
        JETJE = np.einsum('nji,njk->nik', JE, JE)

        return z, 0.5 * np.linalg.slogdet(JETJE)[1]

    return init_fun, apply_fun


def serial(*transforms):
    """Combinator for sequentially combining transforms to create a flow.

    Direct adaptation of the same function in
    https://github.com/google/jax/blob/master/jax/experimental/stax.py
    """
    ntransforms = len(transforms)
    init_funs, apply_funs = zip(*transforms)

    def init_fun(rng):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = random.split(rng)
            param = init_fun(layer_rng)
            params.append(param)
        return params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = random.split(rng, ntransforms) if rng is not None \
            else (None,) * ntransforms
        ldjs = np.zeros(inputs.shape[0])
        for fun, param, rng in zip(apply_funs, params, rngs):
            inputs, ldj = fun(param, inputs, rng=rng, **kwargs)
            ldjs += ldj
        return inputs, ldjs

    return init_fun, apply_fun
