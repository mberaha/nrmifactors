import argparse
import os
os.environ["JAX_ENABLE_X64"] = "true"

import pickle
import jax.numpy as np

from copy import deepcopy
from jax import random
from jax.ops import index_update, index

from nrmifactors import algorithm as algo
from nrmifactors.state import State
import nrmifactors.priors as priors

from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=str)
    parser.add_argument("--chains_file", type=str)
    args = parser.parse_args()

    seed = 0
    key = random.PRNGKey(seed)

    with open(args.data_file, "rb") as fp:
        data = pickle.load(fp)
    
    data = np.array(data)
    nobs_by_group = np.array(
        [np.count_nonzero(~np.isnan(x)) for x in data]).astype(float)
    nan_idx = index[np.isnan(data)]
    ngroups = data.shape[0]

    natoms = 5
    nlat = 5

    # initialize stuff
    init_atoms = np.hstack([
        tfd.Normal(loc=0, scale=10).sample(natoms, seed=key).reshape(-1, 1),
        np.ones((natoms, 1)) * 3
    ])

    prior = priors.NrmiFacPrior(
        kern_prior=priors.NNIGPrior(0.0, 0.001, 3.0, 3.0),
        lam_prior=priors.GammaPrior(2.0, 1.0),
        m_prior=priors.GammaPrior(2.0, 1.0),
        j_prior=priors.GammaPrior(1.0, 1.0))

    lam = tfd.Gamma(prior.lam_prior.a, prior.lam_prior.b).sample(
        (ngroups, nlat), seed=key).astype(float)
    m = tfd.Gamma(prior.m_prior.a, prior.m_prior.b).sample(
        (nlat, natoms), seed=key).astype(float)

    j = np.ones(natoms).astype(float) * 0.5
    u = np.ones(ngroups).astype(float)

    clus = tfd.Categorical(probs=np.ones(natoms)/natoms).sample(data.shape, seed=key)
    clus = index_update(clus, np.isnan(data), -10)
    state = State(init_atoms, j, lam, m, clus, u)

    niter = 10000
    nburn = 5000
    thin = 1

    states = [deepcopy(state)]
    for i in range(niter):
        if (i%100) == 0:
            print("\r{0}/{1}".format(i+1, niter), flush=True, end=" ")

        state, key = algo.run_one_step(
            state, data, nan_idx, nobs_by_group, prior, key)
        if (i > nburn) and (i % thin == 0):
            states.append(deepcopy(state))

    with open(args.chains_file, "wb") as fp:
        pickle.dump(states, fp)
