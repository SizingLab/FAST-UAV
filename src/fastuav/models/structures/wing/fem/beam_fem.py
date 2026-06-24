"""
Wing-spar beam FEM, OpenMDAO-wrapped.

Solves K u = F for a clamped-root cantilever spar of linearly-tapered
radius and wall thickness, under a distributed transverse load q(y).

Two layers:

1. solve_beam_fem(...) -- pure-NumPy core. Takes geometry, material, mesh
   refinement, distributed-load samples, and an element class. Returns
   nodal DOFs, element bending moments, and tip displacement. Used both
   by the OpenMDAO component below and by the verification script.

2. BeamFEM -- OpenMDAO ExplicitComponent that wraps solve_beam_fem(...).
   Inputs/outputs use FAST-UAV-style naming. Partials are by complex step
   for now; we can add analytic Jacobians once the model is settled.

Element type is selectable via the `element_class` option. Default is
BeamElement2D; the 3D spatial element will plug in here once implemented
(only the q-input shape changes).
"""

from __future__ import annotations

import numpy as np

try:
    import openmdao.api as om
    _HAS_OPENMDAO = True
except ImportError:
    _HAS_OPENMDAO = False

from .beam_element import BeamElement2D, BeamElementBase
from .section import section_properties


# ---------------------------------------------------------------------------
# Pure-NumPy FEM core
# ---------------------------------------------------------------------------

def solve_beam_fem(
    semi_span: float,
    R_root: float, R_tip: float,
    t_root: float, t_tip: float,
    E: float, G: float,
    q_nodes: np.ndarray,
    n_elements: int = 20,
    element_class: type[BeamElementBase] = BeamElement2D,
    spar_model: str = "pipe",
):
    """
    Solve the clamped-root spar beam under a distributed transverse load.

    Parameters
    ----------
    semi_span : half-span b/2 [m]
    R_root, R_tip : root/tip value of the first spar taper dimension [m]
        (pipe: outer radius R; I_beam: total section depth H -- see section.py)
    t_root, t_tip : root/tip value of the second spar taper dimension [m]
        (pipe: wall thickness t; I_beam: web/flange thickness t)
    E : Young's modulus [Pa]
    G : shear modulus [Pa] (torsion, used by the 3D/BEAM3 element)
    spar_model : "pipe" or "I_beam" cross-section configuration
    q_nodes : distributed transverse load sampled at the (n_elements + 1) FE
              nodes [N/m]. For 2D elements this is a 1-D array of length
              n_nodes; for 3D elements it would be a 2-D array of shape
              (n_nodes, n_load_components).
    n_elements : number of beam elements along the half-span
    element_class : BeamElement2D (default) or BeamElement3D

    Returns
    -------
    result : dict with keys
        'y_nodes'   : node y-coordinates, shape (n_nodes,) [m]
        'R_nodes'   : tube radius at each node [m]
        't_nodes'   : tube wall thickness at each node [m]
        'u'         : nodal DOF vector, shape (n_nodes * dof_per_node,)
        'M_bending' : bending moment at each element midpoint [N*m]
        'w_tip'     : transverse tip displacement [m]
    """
    n_nodes = n_elements + 1
    dpn = element_class.dof_per_node
    n_dof = n_nodes * dpn

    if q_nodes.shape[0] != n_nodes:
        raise ValueError(
            f"q_nodes has length {q_nodes.shape[0]}, expected {n_nodes} "
            f"(= n_elements + 1)."
        )

    # Mesh: uniformly spaced nodes from root to tip.
    y_nodes = np.linspace(0.0, semi_span, n_nodes)

    # Linear taper of R(y) and t(y).
    eta = y_nodes / semi_span
    R_nodes = R_root + (R_tip - R_root) * eta
    t_nodes = t_root + (t_tip - t_root) * eta

    # Assemble global stiffness K and load vector F.
    K = np.zeros((n_dof, n_dof))
    F = np.zeros(n_dof)

    for e in range(n_elements):
        # Section properties at element midpoint.
        R_e = 0.5 * (R_nodes[e] + R_nodes[e + 1])
        t_e = 0.5 * (t_nodes[e] + t_nodes[e + 1])
        A_e, Iy_e, Iz_e, J_e, _ = section_properties(spar_model, R_e, t_e)
        L_e = y_nodes[e + 1] - y_nodes[e]

        K_e = element_class.stiffness_matrix(L_e, E, G, A_e, Iy_e, Iz_e, J_e)
        F_e = element_class.consistent_load_vector(L_e, q_nodes[e], q_nodes[e + 1])

        # Scatter into global system.
        i0 = e * dpn
        i1 = (e + 2) * dpn
        K[i0:i1, i0:i1] += K_e
        F[i0:i1] += F_e

    # Boundary conditions: clamp at root (node 0).
    fixed = np.arange(0, dpn)
    free = np.arange(dpn, n_dof)
    K_ff = K[np.ix_(free, free)]
    F_f = F[free]

    u = np.zeros(n_dof)
    u[free] = np.linalg.solve(K_ff, F_f)

    # Recover bending moment at each element midpoint.
    M_bending = np.zeros(n_elements)
    for e in range(n_elements):
        R_e = 0.5 * (R_nodes[e] + R_nodes[e + 1])
        t_e = 0.5 * (t_nodes[e] + t_nodes[e + 1])
        _, Iy_e, _, _, _ = section_properties(spar_model, R_e, t_e)
        L_e = y_nodes[e + 1] - y_nodes[e]
        i0 = e * dpn
        i1 = (e + 2) * dpn
        M_bending[e] = element_class.bending_moment_at_midpoint(L_e, E, Iy_e, u[i0:i1])

    # Tip vertical displacement -- DOF index depends on element layout.
    if dpn == 2:        # [w, theta] per node
        w_tip = u[(n_nodes - 1) * dpn + 0]
    elif dpn == 6:      # [u, v, w, theta_x, theta_y, theta_z] per node
        w_tip = u[(n_nodes - 1) * dpn + 2]
    else:
        w_tip = float("nan")

    return {
        "y_nodes": y_nodes,
        "R_nodes": R_nodes,
        "t_nodes": t_nodes,
        "u": u,
        "M_bending": M_bending,
        "w_tip": w_tip,
    }


# ---------------------------------------------------------------------------
# OpenMDAO wrapper
# ---------------------------------------------------------------------------

if _HAS_OPENMDAO:

    class BeamFEM(om.ExplicitComponent):
        """
        OpenMDAO ExplicitComponent wrapping solve_beam_fem.

        Naming follows the FAST-UAV convention. Spanwise mesh size is fixed at
        instantiation (n_elements is a component option, not an input) so the
        I/O shapes don't change at runtime.
        """

        def initialize(self):
            self.options.declare("n_elements", types=int, default=20,
                                 desc="Number of beam elements along the half-span.")
            self.options.declare("element_class", default=BeamElement2D,
                                 desc="BeamElement2D (default) or BeamElement3D.")
            self.options.declare("spar_model", default="pipe",
                                 values=["pipe", "I_beam"],
                                 desc="Spar cross-section configuration.")

        def setup(self):
            n_elem = self.options["n_elements"]
            elem_cls = self.options["element_class"]
            n_nodes = n_elem + 1
            n_dof = n_nodes * elem_cls.dof_per_node

            # --- inputs -----------------------------------------------------
            self.add_input("data:geometry:wing:semi_span", val=1.0, units="m")
            self.add_input("data:geometry:wing:spar:R_root", val=0.020, units="m")
            self.add_input("data:geometry:wing:spar:R_tip",  val=0.010, units="m")
            self.add_input("data:geometry:wing:spar:t_root", val=0.002, units="m")
            self.add_input("data:geometry:wing:spar:t_tip",  val=0.001, units="m")
            self.add_input("data:material:spar:E", val=70.0e9, units="Pa")
            self.add_input("data:material:spar:G", val=27.0e9, units="Pa")

            # Distributed transverse load at FE nodes.
            self.add_input(
                "data:loads:wing:q_nodes",
                val=np.zeros(n_nodes), shape=n_nodes, units="N/m",
                desc="Distributed transverse force per unit span at FE nodes (positive up).",
            )

            # --- outputs ----------------------------------------------------
            self.add_output("u", val=np.zeros(n_dof), shape=n_dof,
                            desc="Nodal DOF vector (root clamped).")
            self.add_output("data:loads:wing:M_bending",
                            val=np.zeros(n_elem), shape=n_elem, units="N*m",
                            desc="Bending moment at each element midpoint.")
            self.add_output("data:geometry:wing:spar:y_nodes",
                            val=np.zeros(n_nodes), shape=n_nodes, units="m")
            self.add_output("data:geometry:wing:spar:R_nodes",
                            val=np.zeros(n_nodes), shape=n_nodes, units="m")
            self.add_output("data:geometry:wing:spar:t_nodes",
                            val=np.zeros(n_nodes), shape=n_nodes, units="m")
            self.add_output("data:loads:wing:w_tip", val=0.0, units="m",
                            desc="Tip transverse displacement.")

            # Complex-step partials are honest and cheap for a small FEM.
            # Replace with analytic Jacobians once the model is stable.
            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            res = solve_beam_fem(
                semi_span = float(inputs["data:geometry:wing:semi_span"]),
                R_root    = float(inputs["data:geometry:wing:spar:R_root"]),
                R_tip     = float(inputs["data:geometry:wing:spar:R_tip"]),
                t_root    = float(inputs["data:geometry:wing:spar:t_root"]),
                t_tip     = float(inputs["data:geometry:wing:spar:t_tip"]),
                E         = float(inputs["data:material:spar:E"]),
                G         = float(inputs["data:material:spar:G"]),
                q_nodes   = inputs["data:loads:wing:q_nodes"],
                n_elements    = self.options["n_elements"],
                element_class = self.options["element_class"],
                spar_model    = self.options["spar_model"],
            )
            outputs["u"]                                  = res["u"]
            outputs["data:loads:wing:M_bending"]          = res["M_bending"]
            outputs["data:geometry:wing:spar:y_nodes"]    = res["y_nodes"]
            outputs["data:geometry:wing:spar:R_nodes"]    = res["R_nodes"]
            outputs["data:geometry:wing:spar:t_nodes"]    = res["t_nodes"]
            outputs["data:loads:wing:w_tip"]              = res["w_tip"]
