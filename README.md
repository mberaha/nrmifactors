# NRMI Factor Models

Code implementing the MCMC algorithm proposed in ``Normalized Latent Measure Factor Models'' by Mario Beraha and Jim E. Griffin

### Code Structure

The source code is inside the `nrmifactors` folder. Speficially

- priors.py collects Python dataclasses specifying the different priors for all the parameters involved in the model

- state.py defines the Python dataclass representing the MCMC state

- algorithm.py collects the functions needed to sample from the full conditionals.
These are implemented using `JAX` and `tensorflow-probability` and expect as input either native Python tipes or `JAX` objects.

In particular
1) `update_gaussmix_atoms` updates the component-specific parameters under a Gaussian mixture model
2) `update_Js_gamma` updates the J's parameter when they are i.i.d. Gamma distributed
3) `update_Js_beta` updates the J's parameter when they are i.i.d. Beta distributed
4) `update_clus` updates the cluster allocations under a Gaussian mixture model
5) `update_lambda_unconstrained` and `update_lambda` update the matrix Lambda when its entries are i.i.d. Gamma distributed with and without transformation in the log-domain respectively
6) `update_lambda_mgp` updates the matrix Lambda under the multiplicative Gamma process prior
7) `adapt_lambda_mgp` performs the adaptation step (either adding or removing columns) on Lambda under the multiplicative Gamma process prior
8) `update_lambda_gmrf` updates the matrix Lambda under the log-Gaussian Markov random field prior
9) `update_m` updates the matrix M when its entries are i.i.d. Gamma distributed
10) `update_u` updates the auxiliary variables u

Each of these functions takes as impupt the part of the state of the MCMC needed to compute the full-conditional distribution, the prior hyperparameters and the rng state (see JAX's documentation) and returns the updated parameters, the new
rng state and possibly some tuning parameters such as the HMC step size.

Then, one iteration of the MCMC sampling consists in updating each of the parameters calling the appropriate functions in sequence. See for instance the `run_one_step` function that is used in all of our examples.
To perform the adaptation of Lambda, use the `adapt_mgp` function. 

- postprocess.py contains the generic implementation of the dissipative RATTLE algorithm on the special linear group (`dissipative_lie_rattle_fast`), as well as a speficic version of for penalized loss functions (`dissipative_lie_rattle_penalized`), the implementation of the Riemannian augmented lagrangian optimization method (`ralm`), and the implementation of two alignment functions that align the latent densities to a template (`greedy_align` and `optimal_align`)


Finally, the folderds `invalsi`, `income_data` and `spatial_simu` contain files (python scripts and jupyter notebooks) to run the examples in the paper.

### Example

See the notebook `Simulation_1.ipynb` for a full working example
