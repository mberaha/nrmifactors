import jax.numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class State:
    iter: int
    atoms: np.array
    j: np.array
    m: np.array
    clus: np.array
    u: np.array
    m_step_size: float = 1e-4
    lam_step_size: float = 1e-4
    lam: Optional[np.array] = None
    phis: Optional[np.array] = None
    deltas: Optional[np.array] = None
    tau: Optional[float] = None
    tau_step_size: float = 1e-2
    accept_lam_scaling: Optional[int] = 0

