from absl import app, flags, logging

import jax.numpy as np
from jax import grad, jit, random
from jax.experimental.optix import adam

import flows
import optimisation
import sd

flags.DEFINE_integer('N', 1, 'Number of sequential flows applied')
flags.DEFINE_integer('K', 1, 'Number of radial components per flow')
flags.DEFINE_float('lr', 2e-4, 'Learning rate')
flags.DEFINE_integer('batch', 256, 'Batch size')
flags.DEFINE_integer('iterations', 20000, 'Number of training iterations')
flags.DEFINE_integer('samples', 20000, 'Number of samples used for evaluation')
flags.DEFINE_boolean('plot', False, 'Plot resulting model density')

FLAGS = flags.FLAGS


def data_stream():
    _rng = random.PRNGKey(0)
    while True:
        _rng, rng_input = random.split(_rng)
        yield sd.sample_sd(rng_input, 2, FLAGS.batch)


def main(_):
    rng = random.PRNGKey(1)

    init_fun, apply_fun = flows.serial(
        *[flows.ExponentialMapSumRadialFlow(FLAGS.K, 2)
          for _ in range(FLAGS.N)]
    )
    params = init_fun(rng)
    opt_init, opt_update = adam(FLAGS.lr)
    opt_state = opt_init(params)

    @jit
    def loss(params, inputs):
        prior_log_prob = np.log(1 / (4 * np.pi)) * np.ones(inputs.shape[0])
        z, ldjs = apply_fun(params, inputs)
        return (prior_log_prob - ldjs -
                np.log(optimisation.s2_target(z))).mean()

    @jit
    def update(opt_state, params, batch):
        grads = grad(loss)(params, batch)
        updates, opt_state = opt_update(grads, opt_state)
        params = optimisation.apply_updates(params, updates)
        return opt_state, params

    batches = data_stream()
    uniform_s2_samples = sd.sample_sd(rng, 2, FLAGS.samples)
    for i in range(1, FLAGS.iterations + 1):
        opt_state, params = update(opt_state, params, next(batches))
        if not (i % (FLAGS.iterations // 10)):
            msg = "Iter {} | Loss {:.3f}"
            logging.info(msg.format(i, loss(params, uniform_s2_samples)))

    model_samples, ldjs = apply_fun(params, uniform_s2_samples)
    log_prob = np.log(1 / (4 * np.pi)) * np.ones(FLAGS.samples) - ldjs
    _,  kl, ess = optimisation.kl_ess(
        log_prob, optimisation.s2_target(model_samples))

    msg = "KL = {:.2f} | ESS {:.0f}%"
    logging.info(msg.format(kl, ess / FLAGS.samples * 100))
    if FLAGS.plot:
        import plotting
        plotting.plot_model_density(model_samples)


if __name__ == '__main__':
    app.run(main)
