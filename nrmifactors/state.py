import jax.numpy as np
from dataclasses import dataclass


@dataclass
class State:
    atoms: np.array
    j: np.array
    lam: np.array
    m: np.array
    clus: np.array
    u: np.array
    m_step_size: float = 1e-4
    lam_step_size: float = 1e-4

