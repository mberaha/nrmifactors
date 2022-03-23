import os
os.environ["JAX_ENABLE_X64"] = "true"
os.environ["xla_cpu_multi_thread_eigen"] = "False"
os.environ["intra_op_parallelism_threads"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


import jax.numpy as np
from jax import random
from jax.experimental.sparse import COO
from jax.ops import index
from joblib import Parallel, delayed

from nrmifactors import algorithm as algo
from nrmifactors.state import State
import nrmifactors.priors as priors
import pickle
from sklearn.cluster import KMeans
from copy import deepcopy

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

key = random.PRNGKey(0)

import numpy as onp
import pandas as pd

def get_weights(Nx, Ny):

    N = Nx*Ny
    centers = onp.zeros((N, 2))
    for i in range(Nx):
        for j in range(Ny):
            centers[i + j*Nx, :] = np.array([i + 0.5, j + 0.5])
    c = 0.3
    alpha1 = c
    alpha2 = -c
    beta1 = c
    beta2 = -c

    weights = []
    mean_centers = np.mean(centers, axis=0)
    for center in centers:
        w1 = alpha1 * (center[0] - mean_centers[0]) \
             + beta1 * (center[1] - mean_centers[1])
        w2 = alpha2 * (center[0] - mean_centers[0]) \
            + beta2 * (center[1] - mean_centers[1])
        weights.append(inv_alr([w1, w2]))

    return np.array(weights)

def inv_alr(x):
    out = onp.exp(np.hstack((x, 0)))
    return np.array(out / np.sum(out))


def simulate_from_mixture(weights):
    means = [-5, 0, 5]
    comp = onp.random.choice(3, p=weights)
    return onp.random.normal(loc=means[comp], scale=1)


def simulate_data(weights, numSamples):
    data = []
    for i in range(len(weights)):
        for j in range(numSamples):
            data.append([i, simulate_from_mixture(weights[i])])
    return pd.DataFrame(data, columns=["group", "datum"])


def compute_G(Nx, Ny):
    N = Nx*Ny
    G = onp.diag(np.ones(N-1), 1) + onp.diag(np.ones(N-1), -1) +\
        onp.diag(np.ones(N-Nx), Nx) + onp.diag(np.ones(N-Nx), -Nx)
    # tolgo i bordi
    border_indices = Nx*np.arange(1, Ny)
    G[border_indices, border_indices - 1] = 0
    G[border_indices - 1, border_indices] = 0

    return np.array(G)


def eval_densities(xgrid, lam, m, j, atoms):
    weights = np.matmul(lam, m) * j
    weights /= weights.sum(axis=1)[:, np.newaxis]
    eval_comps = tfd.Normal(loc=atoms[:, 0], scale=np.sqrt(atoms[:, 1])).prob(xgrid[:, np.newaxis])
    dens = eval_comps[:, np.newaxis, :] * weights[np.newaxis, :, :]
    dens = np.sum(dens, axis=-1).T
    return dens


def get_true_dens(xgrid, weights, atoms):
    eval_comps = tfd.Normal(loc=atoms[:, 0], scale=np.sqrt(atoms[:, 1])).prob(xgrid[:, np.newaxis])
    dens = eval_comps[:, np.newaxis, :] * weights[np.newaxis, :, :]
    dens = np.sum(dens, axis=-1).T
    return dens


def fit(prior, ngroups, nlat, natoms, init_atoms, clus, data, outdir):
    key = random.PRNGKey(0)

    lam = np.ones((data.shape[0], nlat))
    m = tfd.Gamma(prior.m_prior.a, prior.m_prior.b).sample(
        (nlat, natoms), seed=key).astype(float)

    j = np.ones(natoms).astype(float) * 0.5
    u = np.ones(ngroups).astype(float)

    state = State(iter=0, atoms=init_atoms, j=j, lam=lam, m=m, 
                  clus=clus, u=u, tau=2.5)

    nan_idx = index[np.isnan(data)]
    nobs_by_group = np.array(
            [np.count_nonzero(~np.isnan(x)) for x in data]).astype(float)


    niter = 10000
    nburn = 5000
    thin = 1

    states = []
    for i in range(niter):
        print("\r{0}/{1}".format(i+1, niter), flush=True, end=" ")
        state, key = algo.run_one_step(state, data, nan_idx, nobs_by_group, prior, key)
        if (i > nburn) and (i % thin == 0):
            states.append(deepcopy(state))

    filename = os.path.join(outdir, "chains_groups_{0}_lat_{1}.pickle".format(ngroups, nlat))
    with open(filename, "wb") as fp:
        pickle.dump(states, fp)
    
    return np.array([1])



if __name__ == "__main__":

    nxx = [4, 8, 16]

    for nx in nxx:
        ngroups = nx**2
        W = compute_G(nx, nx)

        weights = get_weights(nx, nx)
        datas = simulate_data(weights, 100)

        # first our model, in parallel
        groupedData = []
        for g in range(ngroups):
            groupedData.append(datas[datas['group'] == g]['datum'].values)

        data = np.stack(groupedData)

        prec = np.diag(W.sum(axis=1)) - 0.95 * W
        eigvals, eigvecs = np.linalg.eigh(prec)
        prec_logdet = np.sum(np.log(eigvals[eigvals > 1e-6]))
        prec = COO.fromdense(prec)

        natoms = 20
        km = KMeans(natoms)
        km.fit(data.reshape(-1, 1))
        clus = km.predict(data.reshape(-1,1)).reshape(data.shape)
        means = km.cluster_centers_
        init_atoms = np.hstack([means, np.ones_like(means) * 0.3])
        prior = priors.NrmiFacPrior(
            kern_prior=priors.NNIGPrior(0.0, 0.01, 5.0, 5.0),
            lam_prior_gmrf=priors.GMRFPrior(sigma=prec, sigma_logdet=prec_logdet,
                                            tau_a=2, tau_b=2),
            lam_prior="gmrf",
            m_prior=priors.GammaPrior(2.0, 2.0),
            j_prior=priors.GammaPrior(2.0, 2.0))

        dir_path = os.path.dirname(os.path.realpath(__file__))
        nlat = np.array([1, 3, 5, 10])

        delayed_fn = delayed(
            lambda x: fit(deepcopy(prior), ngroups, x, natoms, init_atoms, clus, data, dir_path))

        out = Parallel(n_jobs=5, prefer="threads")(delayed_fn(x) for x in nlat)

        with open(os.path.join(dir_path, "all_out_{0}.pickle".format(nx)), "wb") as fp:
            pickle.dump(out, fp)



            
            