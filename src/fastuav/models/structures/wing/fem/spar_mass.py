"""
Spar mass: integrates rho * A(y) along the span using trapezoidal quadrature
on the FE node mesh. Returns the half-spar mass; the full spar mass is 2x
this value (FAST-UAV's wing mass convention is for the whole wing).
"""

from __future__ import annotations

import numpy as np

# NumPy >= 2.0 renamed np.trapz to np.trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

try:
    import openmdao.api as om
    _HAS_OPENMDAO = True
except ImportError:
    _HAS_OPENMDAO = False

from .section import section_properties


def compute_spar_mass(R_nodes: np.ndarray,
                      t_nodes: np.ndarray,
                      y_nodes: np.ndarray,
                      density: float,
                      spar_model: str = "pipe"):
    """
    Returns the full-span spar mass (both half-wings) [kg].
    """
    A_nodes = np.array([section_properties(spar_model, d1, d2)[0]
                        for d1, d2 in zip(R_nodes, t_nodes)])
    half_mass = density * _trapz(A_nodes, y_nodes)
    return 2.0 * half_mass


if _HAS_OPENMDAO:

    class SparMass(om.ExplicitComponent):

        def initialize(self):
            self.options.declare("n_elements", types=int, default=20)
            self.options.declare("spar_model", default="pipe",
                                 values=["pipe", "I_beam"],
                                 desc="Spar cross-section configuration.")

        def setup(self):
            n_nodes = self.options["n_elements"] + 1
            self.add_input("data:geometry:wing:spar:R_nodes",
                           val=0.02 * np.ones(n_nodes), shape=n_nodes, units="m")
            self.add_input("data:geometry:wing:spar:t_nodes",
                           val=0.002 * np.ones(n_nodes), shape=n_nodes, units="m")
            self.add_input("data:geometry:wing:spar:y_nodes",
                           val=np.zeros(n_nodes), shape=n_nodes, units="m")
            self.add_input("data:material:spar:density", val=1600.0, units="kg/m**3")

            self.add_output("data:weight:airframe:wing:spar:mass",
                            val=0.1, units="kg")

            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            outputs["data:weight:airframe:wing:spar:mass"] = compute_spar_mass(
                R_nodes = inputs["data:geometry:wing:spar:R_nodes"],
                t_nodes = inputs["data:geometry:wing:spar:t_nodes"],
                y_nodes = inputs["data:geometry:wing:spar:y_nodes"],
                density = float(inputs["data:material:spar:density"]),
                spar_model = self.options["spar_model"],
            )
