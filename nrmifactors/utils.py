from tensorflow_probability.substrates import jax as tfp
from jax import jit, random


def _run_hmc(target_lpdf, init_state, n_leapfrog, step_size, niter, rng_key):
    """
    Runs 'niter' steps of simple HMC and returns the last value
    """
    rng_key, subkey = random.split(rng_key)
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=n_leapfrog,
        step_size=step_size)
    out = tfp.mcmc.sample_chain(
        num_results=niter, num_burnin_steps=0, current_state=init_state,
        kernel=kernel, trace_fn=None, seed=subkey)
    return out[-1, :, :], rng_key


run_hmc = jit(_run_hmc, static_argnums=(0,))