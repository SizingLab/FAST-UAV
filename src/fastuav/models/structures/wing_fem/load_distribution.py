"""
Spanwise aerodynamic load distribution for the FEM wing models.

Builds the net distributed transverse load q(y) [N/m] along the half-span and
resamples it onto the FE node mesh. The limit lift is fixed by the ultimate
load factor and the (guessed) MTOW, consistent with FAST-UAV's analytic spar
sizing (``F_max = n_ult * MTOW * g / 2``):

    integral over the half-span of q(y) dy = n_ult * MTOW * g / 2

so the dynamic pressure / air density / true airspeed all cancel out and are
not needed. Only the *shape* of the lift distribution matters; it is taken
from the VLM spanwise ``CL_vector * chord_vector`` when available, otherwise an
elliptical stand-in ``sqrt(1 - (y/(b/2))^2)`` derived from geometry alone.

Wing self-weight relief is intentionally omitted (as in the analytic model):
it is a secondary effect and would create a feedback loop with the wing mass
that this group also computes.
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om
from scipy.constants import g

from fastuav.models.aerodynamics.constants import SPAN_MESH_POINT


def build_q_distribution(semi_span, y_src, lift_shape_src, n_ult, mass, n_elements):
    """
    Resample a (relative) lift-distribution shape onto the FE node mesh and
    scale it so the half-span integral carries ``n_ult * mass * g / 2``.

    Parameters
    ----------
    semi_span : half-span b/2 [m].
    y_src : source spanwise stations [m], ascending from 0 to ~b/2.
    lift_shape_src : relative lift per unit span at ``y_src`` (any positive scale).
    n_ult : ultimate load factor [-].
    mass : sizing mass (MTOW guess) [kg].
    n_elements : number of beam elements (FE mesh has n_elements + 1 nodes).

    Returns
    -------
    q_nodes : distributed transverse load at FE nodes [N/m], length n_elements + 1.
    """
    n_nodes = n_elements + 1
    y_fe = np.linspace(0.0, semi_span, n_nodes)

    shape_fe = np.interp(y_fe, y_src, lift_shape_src)
    shape_fe = np.clip(shape_fe, 0.0, None)        # only upward lift sizes the spar

    integral = np.trapezoid(shape_fe, y_fe)
    target = n_ult * mass * g / 2.0                # half-wing limit lift [N]
    if integral > 0.0:
        q_nodes = target * shape_fe / integral
    else:
        q_nodes = np.full(n_nodes, target / semi_span)
    return q_nodes


class WingLoadDistribution(om.ExplicitComponent):
    """
    OpenMDAO wrapper around :func:`build_q_distribution`.

    With ``use_aero_vectors=False`` (default) the lift shape is an elliptical
    stand-in built from the span only, so the component is self-sufficient and
    does not depend on a VLM aerodynamic solve. Set it ``True`` to drive the
    shape from the VLM spanwise ``CL_vector * chord_vector``.
    """

    def initialize(self):
        self.options.declare("n_elements", types=int, default=20)
        self.options.declare("use_aero_vectors", types=bool, default=False,
                             desc="Use VLM spanwise CL/chord vectors for the lift shape.")

    def setup(self):
        n_nodes = self.options["n_elements"] + 1

        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("mission:sizing:load_factor:ultimate", val=3.0, units=None)
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")

        if self.options["use_aero_vectors"]:
            self.add_input("data:aerodynamics:wing:low_speed:Y_vector",
                           shape=SPAN_MESH_POINT, val=0.0, units="m")
            self.add_input("data:aerodynamics:wing:low_speed:CL_vector",
                           shape=SPAN_MESH_POINT, val=0.0)
            self.add_input("data:aerodynamics:wing:low_speed:chord_vector",
                           shape=SPAN_MESH_POINT, val=0.0, units="m")

        self.add_output("data:loads:wing:q_nodes",
                        val=np.zeros(n_nodes), shape=n_nodes, units="N/m")

    def setup_partials(self):
        # Central + relative-step FD to keep the load-distribution derivatives clean
        # for the gradient-based driver (see TubeSparFoamModel for rationale).
        self.declare_partials("*", "*", method="fd", form="central", step_calc="rel_avg")

    def compute(self, inputs, outputs):
        semi_span = 0.5 * float(inputs["data:geometry:wing:span"])
        n_ult = float(inputs["mission:sizing:load_factor:ultimate"])
        mass = float(inputs["optimization:variables:weight:mtow:guess"])

        use_aero = self.options["use_aero_vectors"]
        y_vlm = cl_vlm = chord_vlm = None
        if use_aero:
            y_vlm = np.asarray(inputs["data:aerodynamics:wing:low_speed:Y_vector"])
            cl_vlm = np.asarray(inputs["data:aerodynamics:wing:low_speed:CL_vector"])
            chord_vlm = np.asarray(inputs["data:aerodynamics:wing:low_speed:chord_vector"])

        if use_aero and np.max(np.abs(cl_vlm)) > 0.0:
            # Sort by y in case the VLM mesh is not ascending; drop zero padding.
            order = np.argsort(y_vlm)
            y_src = y_vlm[order]
            shape_src = (cl_vlm * chord_vlm)[order]
        else:
            # Elliptical stand-in from geometry only.
            y_src = np.linspace(0.0, semi_span, SPAN_MESH_POINT)
            eta = y_src / semi_span
            shape_src = np.sqrt(np.clip(1.0 - eta**2, 0.0, 1.0))

        outputs["data:loads:wing:q_nodes"] = build_q_distribution(
            semi_span=semi_span,
            y_src=y_src,
            lift_shape_src=shape_src,
            n_ult=n_ult,
            mass=mass,
            n_elements=self.options["n_elements"],
        )
