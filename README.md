# $\mathbb{S}^D$ flows

Attempt at reproducing "exponential-map-sum-of-radial-flow" results (EMSRE Table 1) of [Normalizing Flows on Tori and Spheres](https://arxiv.org/pdf/2002.02428.pdf) by Rezende et al., done in JAX.

The target density on $\mathbb{S}^2$ is

<img src="https://github.com/katalinic/sdflows/blob/master/s2_target_density.png" width="400">

To train the flow with hyperparameters $^\dagger$ specified in the paper, run below adjusting N and K as desired.

`$ python3 main.py --N=1 --K=12`

Comparing the authors' results (average) and ours (single run as is), with the convention of theirs / ours:

| Model        | KL             | ESS  |
| -------------   |:-------------:| -----|
| EMSRE(N=1, K=12)| 0.82 / 0.78 | 42 % / 42 % |̣
| EMSRE(N=6, K=5) | 0.19 / 0.19 | 75 % / 82 % |

Scenario N=24, K=1 was also attempted, though unsuccessfully in 20,000 iterations claimed.

$^\dagger$ OTOH, parameter initialisation and constraint enforcing are not specified so had to be assumed.
