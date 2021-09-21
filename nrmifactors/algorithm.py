from nrmifactors.kernels import nnig_update, norm_lpdf
from nrmifactors.utils import run_hmc
from nrmifactors.priors import NrmiFacPrior
import jax.numpy as np
from jax import jit, random
from jax.scipy.special import logsumexp

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


def update_atoms(data, clus, nclus, kp_mu0, kp_lam, kp_a, kp_b, rng_key):
    out = []
    for c in range(nclus):
        currdata = np.ravel(data[clus == c])
        if len(currdata) == 0:
            rng_key, subk1, subk2 = random.split(rng_key, 3)
            var = tfd.Gamma(kp_a, kp_b).sample(seed=subk1)
            mu = tfd.Normal(kp_mu0, np.sqrt(var / kp_lam)).sample(seed=subk2)
            val = np.array([mu, var])
        else:
            val, rng_key = nnig_update(
                np.ravel(data[clus == c]), kp_mu0, kp_lam, kp_a, kp_b, rng_key)
        out.append(val)

    return np.vstack(out), rng_key

@jit
def update_Js(clus, lam, m, u, j_prior_a, j_prior_b, rng_key):
    nclus = m.shape[1]
    out = []
    lam_m = np.matmul(lam, m)
    cards = np.array([np.sum(clus == c)
                     for c in np.arange(nclus)]).astype(float)
    a_post = cards + j_prior_a
    b_post = np.matmul(lam_m.T, u) + j_prior_b
    rng_key, subkey = random.split(rng_key)
    out = tfd.Gamma(a_post, b_post).sample(seed=subkey)
    return out, rng_key

@jit
def update_clus(data, lam, m, j, atoms, rng_key):
    prior_probas = np.log(np.matmul(lam, m) * j)
    likes = norm_lpdf(data, atoms)
    probas = np.exp(prior_probas + likes)
    probas /= np.sum(probas, axis=1)[:, np.newaxis, :]
    rng_key, subkey = random.split(rng_key)
    clus = tfd.Categorical(probs=probas).sample(seed=subkey).T
    return clus, rng_key

@jit
def lambda_full_cond_lpdf(lam, clus, m, j, u, lam_prior_a, lam_prior_b):
    lm = np.matmul(lam, m)
    out = np.sum(tfd.Gamma(
        lam_prior_a, lam_prior_b,
        force_probs_to_zero_outside_support=True).log_prob(lam))
    out -= np.sum(np.sum(lm * j, axis=1) * u)
    cluscount = np.sum(clus[:, :, np.newaxis] ==
                        np.arange(j.shape[0]), axis=1)
    out += np.sum(np.log(lm) * cluscount)
    return out


@jit
def update_lambda(
        clus, lam, m, j, u, lam_prior_a, lam_prior_b, rng_key, step_size,
        nsteps=50, adapt_rate=0.01):

    def target_lpdf(x): return lambda_full_cond_lpdf(
        x, clus, m, j, u, lam_prior_a, lam_prior_b)

    rng_key, subkey = random.split(rng_key)
    nburn = int(niter * 0.8)
    kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=niter, num_burnin_steps=nburn, current_state=lam,
        kernel=kernel, trace_fn=None, seed=subkey)
    return out[-1, :, :], rng_key


@jit
def update_m(
        clus, lam, m, j, u, m_prior_a, m_prior_b, rng_key, step_size=0.0001,
        nsteps=50, niter=100):
    @jit
    def full_cond_lpdf(m, prior_shape, prior_rate):
        lm = np.matmul(lam, m)
        out = np.sum(tfd.Gamma(
            prior_shape, prior_rate,
            force_probs_to_zero_outside_support=True).log_prob(m))
        out -= np.sum(np.sum(lm * j, axis=1) * u)
        cluscount = np.sum(clus[:, :, np.newaxis] ==
                           np.arange(j.shape[0]), axis=1)
        out += np.sum(np.log(lm) * cluscount)
        return out

    def target_lpdf(x): return full_cond_lpdf(x, m_prior_a, m_prior_b)

    rng_key, subkey = random.split(rng_key)
    nburn = int(niter * 0.8)
    inner_kernel = tfp.mcmc.HamiltonianMonteCarlo(
        target_log_prob_fn=target_lpdf, num_leapfrog_steps=nsteps,
        step_size=step_size)
    kernel = tfp.mcmc.SimpleStepSizeAdaptation(
        inner_kernel=inner_kernel, num_adaptation_steps=int(nburn * 0.8))
    out = tfp.mcmc.sample_chain(
        num_results=niter, num_burnin_steps=nburn, current_state=m,
        kernel=kernel, trace_fn=None, seed=subkey)
    return out[-1, :, :], rng_key

@jit
def update_u(data, lam, m, j, rng_key):
    n = np.array([len(x) for x in data]).astype(float)
    t = np.sum(np.matmul(lam, m) * j, axis=1)
    rng_key, subkey = random.split(rng_key)
    out = tfd.Gamma(n, t).sample(seed=subkey)
    return out, rng_key


def run_one_step(state, data, prior, hmc_lambda, hmc_m, rng_key):
    nclus = state.atoms.shape[0]

    state.clus, rng_key = update_clus(
        data, state.lam, state.m, state.j, state.atoms, rng_key)

    state.atoms, rng_key = update_atoms(
        data, state.clus, nclus, prior.kern_prior.mu0, prior.kern_prior.lam, prior.
        kern_prior.a, prior.kern_prior.b, rng_key)

    state.j, rng_key = update_Js(
        state.clus, state.lam, state.m, state.u, prior.j_prior.a, 
        prior.j_prior.b, rng_key)

    state.lam, rng_key = update_lambda(
        state.clus, state.lam, state.m, state.j, state.u, 
        prior.lam_prior.a, prior.lam_prior.b, rng_key)

    # state.lam, rng_key = update_lambda_marg(
    #     data, state.lam, state.m, state.j, state.u, state.atoms,
    #     prior.lam_prior.a, prior.lam_prior.b, rng_key)

    # state.clus, rng_key = update_clus(
    #     data, state.lam, state.m, state.j, state.atoms, rng_key)
    
    state.m, rng_key = update_m(
        state.clus, state.lam, state.m, state.j, state.u, 
        prior.m_prior.a, prior.m_prior.b, rng_key) 

    state.u, rng_key = update_u(data, state.lam, state.m, state.j, rng_key)

    return state, rng_key
