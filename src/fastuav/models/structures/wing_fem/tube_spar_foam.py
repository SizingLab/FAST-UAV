"""
``tube_spar_foam`` wing model: a tapered tubular spar (Euler-Bernoulli beam FEM)
with a foam-cored sandwich skin that *shares* the spanwise bending.

Section idealisation at each spanwise station (neutral axis at mid-thickness,
upper/lower symmetric):

* Tube spar              -> E_spar * I_tube
* Two sandwich faces     -> E_skin * I_faces,  I_faces ~= 2 * A_face * (h/2)^2
  separated by the airfoil thickness h, each of area A_face = w_box * t_face
* Foam core (depth h)    -> E_foam * I_foam,   I_foam = w_box * h^3 / 12
  the foam stabilises the faces and provides the shear connection that lets the
  skin carry bending alongside the spar.

Effective bending stiffness  EI_eff(y) = E_spar*I_tube + E_skin*I_faces +
E_foam*I_foam  feeds the shared :func:`solve_bending_beam`. Outer-fibre stresses
are recovered for both the spar and the faces, normalised by their respective
allowables, and KS-aggregated into a single dimensionless failure margin
(``<= 0`` is feasible). The objective for the top-level MDO is the total wing
structural mass (spar + faces + foam, both half-wings).
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from .buckling import sandwich_wrinkling_stress, tube_local_buckling_stress
from .fe_core.beam_element import tube_section_properties
from .fe_core.beam_solver import solve_bending_beam
from .fe_core.stress import ks_aggregate


class TubeSparFoamModel(om.ExplicitComponent):
    """FEM sizing of the tube-spar + foam-sandwich-skin wing."""

    def initialize(self):
        self.options.declare("n_elements", types=int, default=20)
        self.options.declare("ks_rho", types=float, default=100.0)
        self.options.declare("box_chord_ratio", types=float, default=0.5,
                             desc="Chord fraction over which the sandwich skin "
                                  "carries spanwise bending.")
        self.options.declare("tube_buckling_coeff", types=float, default=0.3,
                             desc="Knockdown x classical coefficient for thin-tube "
                                  "local (shell) buckling, sigma_cr = coeff*E*t/R.")
        self.options.declare("wrinkling_coeff", types=float, default=0.5,
                             desc="Coefficient for sandwich-face wrinkling, "
                                  "sigma_wr = coeff*(E_face*E_core*G_core)^(1/3).")

    def setup(self):
        n_elem = self.options["n_elements"]
        n_nodes = n_elem + 1

        # Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")

        # Materials
        self.add_input("data:structures:wing:spar:material:E", val=70.0e9, units="Pa")
        self.add_input("data:structures:wing:spar:material:stress:max", val=600.0e6, units="Pa")
        self.add_input("data:structures:wing:skin:material:E", val=70.0e9, units="Pa")
        self.add_input("data:structures:wing:skin:material:stress:max", val=300.0e6, units="Pa")
        self.add_input("data:structures:wing:foam:E", val=30.0e6, units="Pa")
        self.add_input("data:weight:airframe:wing:spar:density", val=1600.0, units="kg/m**3")
        self.add_input("data:structures:wing:skin:material:density", val=1600.0, units="kg/m**3")
        self.add_input("data:structures:wing:foam:density", val=60.0, units="kg/m**3")

        # Design variables (tapered tube + face thickness)
        self.add_input("data:structures:wing:spar:diameter:outer:root", val=0.040, units="m")
        self.add_input("data:structures:wing:spar:diameter:outer:tip", val=0.024, units="m")
        self.add_input("data:structures:wing:spar:wall:thickness:root", val=0.002, units="m")
        self.add_input("data:structures:wing:spar:wall:thickness:tip", val=0.001, units="m")
        self.add_input("data:structures:wing:skin:face:thickness", val=0.0003, units="m")

        # Load (from WingLoadDistribution)
        self.add_input("data:loads:wing:q_nodes",
                       val=np.zeros(n_nodes), shape=n_nodes, units="N/m")

        # Outputs: masses
        self.add_output("data:weight:airframe:wing:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:spar:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:skin:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:foam:mass", units="kg", lower=0.0)
        # Outputs: sizing constraints + diagnostics
        self.add_output("data:constraints:structures:wing:failure_margin", units=None)
        self.add_output("data:constraints:structures:wing:buckling_margin", units=None)
        self.add_output("data:loads:wing:sigma_max", units="Pa")
        self.add_output("data:loads:wing:w_tip", units="m")

    def setup_partials(self):
        # Central differencing with a relative step: the KS-aggregated margins are
        # smooth but sensitive, and an absolute FD step is far too coarse for the
        # thin-gauge design variables (e.g. face thickness ~3e-4 m). This keeps the
        # total derivatives clean enough for SLSQP to converge.
        self.declare_partials("*", "*", method="fd", form="central", step_calc="rel_avg")

    def compute(self, inputs, outputs):
        n_elem = self.options["n_elements"]
        n_nodes = n_elem + 1
        kbox = self.options["box_chord_ratio"]

        semi_span = 0.5 * float(inputs["data:geometry:wing:span"][0])
        c_root = float(inputs["data:geometry:wing:root:chord"][0])
        c_tip = float(inputs["data:geometry:wing:tip:chord"][0])
        h_root = float(inputs["data:geometry:wing:root:thickness"][0])
        h_tip = float(inputs["data:geometry:wing:tip:thickness"][0])

        E_spar = float(inputs["data:structures:wing:spar:material:E"][0])
        sig_spar = float(inputs["data:structures:wing:spar:material:stress:max"][0])
        E_skin = float(inputs["data:structures:wing:skin:material:E"][0])
        sig_skin = float(inputs["data:structures:wing:skin:material:stress:max"][0])
        E_foam = float(inputs["data:structures:wing:foam:E"][0])
        rho_spar = float(inputs["data:weight:airframe:wing:spar:density"][0])
        rho_skin = float(inputs["data:structures:wing:skin:material:density"][0])
        rho_foam = float(inputs["data:structures:wing:foam:density"][0])

        d_out_root = float(inputs["data:structures:wing:spar:diameter:outer:root"][0])
        d_out_tip = float(inputs["data:structures:wing:spar:diameter:outer:tip"][0])
        t_wall_root = float(inputs["data:structures:wing:spar:wall:thickness:root"][0])
        t_wall_tip = float(inputs["data:structures:wing:spar:wall:thickness:tip"][0])
        t_face = float(inputs["data:structures:wing:skin:face:thickness"][0])

        q_nodes = np.asarray(inputs["data:loads:wing:q_nodes"], dtype=float)

        # --- Spanwise station geometry (linear taper, root->tip) ---
        y_nodes = np.linspace(0.0, semi_span, n_nodes)
        eta = y_nodes / semi_span if semi_span > 0 else np.zeros(n_nodes)
        R_out = 0.5 * (d_out_root + (d_out_tip - d_out_root) * eta)
        t_wall = t_wall_root + (t_wall_tip - t_wall_root) * eta
        chord = c_root + (c_tip - c_root) * eta
        h = h_root + (h_tip - h_root) * eta          # airfoil thickness -> box height
        w_box = kbox * chord

        # --- Section properties at nodes ---
        A_tube = np.array([tube_section_properties(R, t)[0] for R, t in zip(R_out, t_wall)])
        I_tube = np.array([tube_section_properties(R, t)[1] for R, t in zip(R_out, t_wall)])
        A_face = w_box * t_face
        I_faces = 2.0 * (A_face * (0.5 * h) ** 2 + w_box * t_face**3 / 12.0)
        I_foam = w_box * h**3 / 12.0
        EI_eff = E_spar * I_tube + E_skin * I_faces + E_foam * I_foam

        # --- Bending solution on the shared 1D backbone ---
        EI_mid = 0.5 * (EI_eff[:-1] + EI_eff[1:])
        sol = solve_bending_beam(y_nodes, EI_mid, q_nodes)
        M = sol["M_bending"]
        w_tip = sol["w_tip"]

        # --- Stress recovery at element midpoints ---
        R_mid = 0.5 * (R_out[:-1] + R_out[1:])
        h_mid = 0.5 * (h[:-1] + h[1:])
        EImid = 0.5 * (EI_eff[:-1] + EI_eff[1:])
        sigma_spar = np.abs(M) * E_spar * R_mid / EImid
        sigma_face = np.abs(M) * E_skin * (0.5 * h_mid) / EImid

        u_spar = sigma_spar / sig_spar
        u_face = sigma_face / sig_skin
        u_all = np.concatenate([u_spar, u_face])
        failure_margin = ks_aggregate(u_all, self.options["ks_rho"]) - 1.0

        # --- Buckling utilisation (compression side; |bending stress| is the
        #     compressed-fibre magnitude for the symmetric section) ---
        t_wall_mid = 0.5 * (t_wall[:-1] + t_wall[1:])
        sig_cr_tube = tube_local_buckling_stress(
            E_spar, t_wall_mid, R_mid, coeff=self.options["tube_buckling_coeff"])
        sig_wr = sandwich_wrinkling_stress(
            E_skin, E_foam, coeff=self.options["wrinkling_coeff"])
        u_tube_buck = sigma_spar / sig_cr_tube
        u_face_buck = sigma_face / sig_wr
        u_buck = np.concatenate([u_tube_buck, u_face_buck])
        buckling_margin = ks_aggregate(u_buck, self.options["ks_rho"]) - 1.0

        # --- Mass (both half-wings) ---
        m_spar = 2.0 * rho_spar * np.trapezoid(A_tube, y_nodes)
        m_faces = 2.0 * rho_skin * np.trapezoid(2.0 * A_face, y_nodes)
        m_foam = 2.0 * rho_foam * np.trapezoid(w_box * h, y_nodes)

        outputs["data:weight:airframe:wing:spar:mass"] = m_spar
        outputs["data:weight:airframe:wing:skin:mass"] = m_faces
        outputs["data:weight:airframe:wing:foam:mass"] = m_foam
        outputs["data:weight:airframe:wing:mass"] = m_spar + m_faces + m_foam
        outputs["data:constraints:structures:wing:failure_margin"] = failure_margin
        outputs["data:constraints:structures:wing:buckling_margin"] = buckling_margin
        outputs["data:loads:wing:sigma_max"] = max(sigma_spar.max(), sigma_face.max())
        outputs["data:loads:wing:w_tip"] = w_tip
