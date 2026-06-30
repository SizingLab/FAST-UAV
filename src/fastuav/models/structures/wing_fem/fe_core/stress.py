"""
Stress-recovery helpers shared by the FEM wing models (pure NumPy).

* :func:`ks_aggregate`     -- smooth Kreisselmeier-Steinhauser max.
* :func:`tube_fibre_stress`-- outer-fibre bending stress of a circular tube.
"""

from __future__ import annotations

import numpy as np


def ks_aggregate(values: np.ndarray, rho: float = 100.0) -> float:
    """
    Kreisselmeier-Steinhauser smooth upper bound on ``max(values)``.

        KS = v_max + (1/rho) * ln( sum_i exp( rho (v_i - v_max) ) )

    ``v_max`` is subtracted inside the exp for numerical stability. As
    ``rho -> inf`` the result converges to ``max(values)`` from above.
    """
    values = np.asarray(values, dtype=float)
    v_max = values.max()
    return float(v_max + (1.0 / rho) * np.log(np.sum(np.exp(rho * (values - v_max)))))


def tube_fibre_stress(M_bending: np.ndarray,
                      R_nodes: np.ndarray,
                      t_nodes: np.ndarray) -> np.ndarray:
    """
    Outer-fibre bending stress sigma = M * R / I at element midpoints, for a
    tapered circular tube whose radius/thickness are given at the FE nodes.
    """
    R_mid = 0.5 * (R_nodes[:-1] + R_nodes[1:])
    t_mid = 0.5 * (t_nodes[:-1] + t_nodes[1:])
    R_in = R_mid - t_mid
    I_mid = (np.pi / 4.0) * (R_mid**4 - R_in**4)
    return M_bending * R_mid / I_mid
