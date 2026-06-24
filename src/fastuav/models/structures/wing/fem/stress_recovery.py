"""
Bending stress recovery and KS-aggregated failure constraint.

Takes the FEM bending moments at element midpoints and the spar geometry
at FE nodes, and produces:

  - sigma(y) at element midpoints      [Pa]
  - sigma_max_ks  (smooth max via KS)  [Pa]
  - failure_margin = sigma_max_ks - sigma_allow / SF  [Pa]

The optimizer drives failure_margin <= 0 as the structural constraint.

KS aggregation
--------------
The non-smooth max(sigma_i) is replaced by a smooth, differentiable upper
bound (Kreisselmeier-Steinhauser):

    KS(sigma_i) = sigma_max + (1/rho) * ln( sum_i exp( rho (sigma_i - sigma_max) ) )

where sigma_max = max(sigma_i) is subtracted inside the exp for numerical
stability. As rho -> infinity the KS function converges to max(sigma_i)
from above; rho ~ 50-200 is the usual range. Larger rho is tighter but
sharpens gradients.
"""

from __future__ import annotations

import numpy as np

try:
    import openmdao.api as om
    _HAS_OPENMDAO = True
except ImportError:
    _HAS_OPENMDAO = False

from .section import section_properties


# ---------------------------------------------------------------------------
# Pure-NumPy core
# ---------------------------------------------------------------------------

def recover_stress(M_bending: np.ndarray,
                   R_nodes: np.ndarray,
                   t_nodes: np.ndarray,
                   ks_rho: float = 100.0,
                   sigma_allow: float = 600.0e6,
                   safety_factor: float = 1.5,
                   spar_model: str = "pipe"):
    """
    Compute element-midpoint bending stress and KS-aggregated failure margin.

    Parameters
    ----------
    M_bending : bending moment at each element midpoint [N*m], shape (n_elem,).
    R_nodes, t_nodes : the two spar taper dimensions at FE nodes,
        shape (n_elem + 1,). Their meaning depends on ``spar_model``
        (pipe: outer radius / wall thickness; I_beam: depth / thickness).
    ks_rho : KS smoothing parameter [-].
    sigma_allow : ultimate allowable stress [Pa].
    safety_factor : safety factor applied on top of the ultimate load factor.
    spar_model : "pipe" or "I_beam" cross-section configuration.

    Returns
    -------
    sigma : signed stress at each element midpoint [Pa]
    sigma_abs_max_ks : KS-aggregated max of |sigma| [Pa]
    failure_margin : sigma_abs_max_ks - sigma_allow / safety_factor [Pa]
        Constraint: failure_margin <= 0.
    """
    d1_mid = 0.5 * (R_nodes[:-1] + R_nodes[1:])
    d2_mid = 0.5 * (t_nodes[:-1] + t_nodes[1:])
    props = [section_properties(spar_model, d1, d2) for d1, d2 in zip(d1_mid, d2_mid)]
    I_mid = np.array([p[1] for p in props])   # Iy: vertical-bending second moment
    c_mid = np.array([p[4] for p in props])   # extreme-fibre distance

    # Bending stress at the outer fibre: sigma = M * c / I.
    sigma = M_bending * c_mid / I_mid

    # KS aggregation of |sigma|.
    s = np.abs(sigma)
    s_hat = s.max()
    sigma_abs_max_ks = s_hat + (1.0 / ks_rho) * np.log(
        np.sum(np.exp(ks_rho * (s - s_hat)))
    )

    sigma_allow_w = sigma_allow / safety_factor
    failure_margin = sigma_abs_max_ks - sigma_allow_w

    return sigma, sigma_abs_max_ks, failure_margin


# ---------------------------------------------------------------------------
# OpenMDAO wrapper
# ---------------------------------------------------------------------------

if _HAS_OPENMDAO:

    class StressRecovery(om.ExplicitComponent):
        """
        OpenMDAO wrapper around recover_stress.
        """

        def initialize(self):
            self.options.declare("n_elements", types=int, default=20)
            self.options.declare("ks_rho", types=float, default=100.0)
            self.options.declare("spar_model", default="pipe",
                                 values=["pipe", "I_beam"],
                                 desc="Spar cross-section configuration.")

        def setup(self):
            n_elem = self.options["n_elements"]
            n_nodes = n_elem + 1

            self.add_input("data:loads:wing:M_bending",
                           val=np.zeros(n_elem), shape=n_elem, units="N*m")
            self.add_input("data:geometry:wing:spar:R_nodes",
                           val=0.02 * np.ones(n_nodes), shape=n_nodes, units="m")
            self.add_input("data:geometry:wing:spar:t_nodes",
                           val=0.002 * np.ones(n_nodes), shape=n_nodes, units="m")
            self.add_input("data:material:spar:sigma_allow", val=600e6, units="Pa")
            self.add_input("data:material:spar:safety_factor", val=1.5)

            self.add_output("data:loads:wing:sigma_bending",
                            val=np.zeros(n_elem), shape=n_elem, units="Pa")
            self.add_output("data:loads:wing:sigma_max_ks", val=0.0, units="Pa")
            self.add_output("data:constraints:wing:failure_margin",
                            val=0.0, units="Pa")

            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            sigma, sigma_max_ks, margin = recover_stress(
                M_bending     = inputs["data:loads:wing:M_bending"],
                R_nodes       = inputs["data:geometry:wing:spar:R_nodes"],
                t_nodes       = inputs["data:geometry:wing:spar:t_nodes"],
                ks_rho        = self.options["ks_rho"],
                sigma_allow   = float(inputs["data:material:spar:sigma_allow"]),
                safety_factor = float(inputs["data:material:spar:safety_factor"]),
                spar_model    = self.options["spar_model"],
            )
            outputs["data:loads:wing:sigma_bending"]      = sigma
            outputs["data:loads:wing:sigma_max_ks"]       = sigma_max_ks
            outputs["data:constraints:wing:failure_margin"] = margin
