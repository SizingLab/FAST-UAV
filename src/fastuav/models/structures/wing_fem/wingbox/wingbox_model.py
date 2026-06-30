"""
``wingbox_shell`` wing model: a true shell + beam finite-element wingbox.

Upper/lower skins and chordwise ribs are SRI-Q4 shell elements; the four box
corners are spanwise beam "booms" (the spars-as-beams idealisation). The box is
clamped at the root and loaded by the spanwise lift (applied vertically at the
section nodes). Skin von-Mises and cap axial stresses are normalised by their
allowables and KS-aggregated into a single failure margin (``<= 0`` feasible).
The objective is the total wing structural mass (skins + ribs + caps, both
half-wings).
"""

from __future__ import annotations

import numpy as np
import openmdao.api as om

from ..buckling import column_buckling_stress, plate_buckling_stress
from ..fe_core.assembly import assemble, node_dofs, solve_clamped
from ..fe_core.beam_element import BeamElement3D
from ..fe_core.shell_element import membrane_stress, shell_stiffness
from ..fe_core.stress import ks_aggregate
from .mesh import build_wingbox_mesh


def _quad_area(p):
    """Planar quad area from its diagonals: 0.5 |d1 x d2|."""
    d1 = p[2] - p[0]
    d2 = p[3] - p[1]
    return 0.5 * np.linalg.norm(np.cross(d1, d2))


def _taper(root, tip, eta):
    return root + (tip - root) * eta


def solve_wingbox(mesh, *, E_skin, nu, rho_skin, t_skin_root, t_skin_tip,
                  t_rib, rho_rib, E_cap, nu_cap, rho_cap,
                  A_cap_root, A_cap_tip, sig_skin, sig_cap, q_stations, ks_rho,
                  t_web_root, t_web_tip, E_web, rho_web, sig_web,
                  kc_skin=4.0, cap_fixity=1.0, ks_web=5.35):
    """Assemble, solve, and recover stresses/mass for the wingbox. Pure NumPy."""
    nodes = mesh["nodes"]
    n_nodes = len(nodes)
    G_cap = E_cap / (2.0 * (1.0 + nu_cap))

    elements = []          # (node_ids, K_global)
    skin_records = []      # (node_ids, t, eta)
    # --- skin shells ---
    for q, eta in mesh["skin_quads"]:
        t = _taper(t_skin_root, t_skin_tip, eta)
        K, _ = shell_stiffness(nodes[list(q)], E_skin, nu, t)
        elements.append((q, K))
        skin_records.append((q, t, eta))
    # --- rib shells ---
    for q, eta in mesh["rib_quads"]:
        K, _ = shell_stiffness(nodes[list(q)], E_skin, nu, t_rib)
        elements.append((q, K))
    # --- spar-web shells ---
    web_records = []       # (node_ids, t, eta)
    for q, eta in mesh["web_quads"]:
        t = _taper(t_web_root, t_web_tip, eta)
        K, _ = shell_stiffness(nodes[list(q)], E_web, nu, t)
        elements.append((q, K))
        web_records.append((q, t, eta))
    # --- cap beams ---
    cap_records = []
    for e, eta in mesh["cap_beams"]:
        A = _taper(A_cap_root, A_cap_tip, eta)
        R = np.sqrt(A / np.pi)
        I = 0.25 * np.pi * R**4
        J = 2.0 * I
        L = np.linalg.norm(nodes[e[1]] - nodes[e[0]])
        Kl = BeamElement3D.stiffness_matrix(L, E_cap, G_cap, A, I, I, J)
        T = BeamElement3D.transformation_matrix(nodes[e[0]], nodes[e[1]])
        elements.append((e, T.T @ Kl @ T))
        cap_records.append((e, A, L, T))

    K = assemble(n_nodes, elements)

    # --- vertical lift applied at each station's nodes ---
    y = mesh["y_stations"]
    F = np.zeros(n_nodes * 6)
    for j, snodes in enumerate(mesh["station_nodes"]):
        lo = y[j] - y[j - 1] if j > 0 else 0.0
        hi = y[j + 1] - y[j] if j < len(y) - 1 else 0.0
        trib = 0.5 * (lo + hi)
        Fz = q_stations[j] * trib
        for n in snodes:
            F[n * 6 + 2] += Fz / len(snodes)

    fixed = [rn * 6 + d for rn in mesh["root_nodes"] for d in range(6)]
    u = solve_clamped(K, F, fixed)

    # --- skin stress (max von Mises over the two surfaces) + panel buckling ---
    u_skin = []
    u_skin_buck = []
    skin_mass = 0.0
    for q, t, eta in skin_records:
        coords = nodes[list(q)]
        sigma_m, vm_t, vm_b = membrane_stress(coords, E_skin, nu, t, u[node_dofs(q)])
        u_skin.append(max(vm_t, vm_b) / sig_skin)
        skin_mass += _quad_area(coords) * t * rho_skin
        # Local x runs root->tip (edge 0->1), so sigma_m[0] is the spanwise
        # membrane stress; only compressed panels (sx < 0) can buckle. The
        # buckling width is the chordwise panel dimension (transverse to load).
        sx = sigma_m[0]
        if sx < 0.0:
            b = np.linalg.norm(coords[3] - coords[0])
            sig_cr = plate_buckling_stress(E_skin, t, b, nu=nu, kc=kc_skin)
            u_skin_buck.append(-sx / sig_cr)

    # --- cap axial stress + column buckling ---
    u_cap = []
    u_cap_buck = []
    cap_mass = 0.0
    for (e, A, L, T), (_, eta) in zip(cap_records, mesh["cap_beams"]):
        u_local = T @ u[node_dofs(e)]
        N = BeamElement3D.axial_force(L, E_cap, A, u_local)
        u_cap.append(abs(N / A) / sig_cap)
        cap_mass += A * L * rho_cap
        if N < 0.0:  # only compressed caps buckle (Euler, free length = rib pitch)
            R = np.sqrt(A / np.pi)
            I = 0.25 * np.pi * R**4
            sig_cr = column_buckling_stress(E_cap, I, A, L, coeff=cap_fixity)
            u_cap_buck.append(abs(N / A) / sig_cr)

    # --- spar-web shear stress + shear buckling ---
    u_web = []
    u_web_buck = []
    web_mass = 0.0
    for q, t, eta in web_records:
        coords = nodes[list(q)]
        sigma_m, vm_t, vm_b = membrane_stress(coords, E_web, nu, t, u[node_dofs(q)])
        u_web.append(max(vm_t, vm_b) / sig_web)
        web_mass += _quad_area(coords) * t * rho_web
        # The web carries the transverse shear: sigma_m[2] is the in-plane shear
        # (local x runs spanwise, local y vertical). Check shear buckling on the
        # panel, whose governing width is the web height (vertical edge 0->3).
        tau = abs(sigma_m[2])
        b = np.linalg.norm(coords[3] - coords[0])
        tau_cr = plate_buckling_stress(E_web, t, b, nu=nu, kc=ks_web)
        u_web_buck.append(tau / tau_cr)

    # --- rib mass ---
    rib_mass = sum(_quad_area(nodes[list(q)]) * t_rib * rho_rib
                   for q, _ in mesh["rib_quads"])

    utilization = np.array(u_skin + u_cap + u_web)
    failure_margin = ks_aggregate(utilization, ks_rho) - 1.0
    sigma_max = utilization.max() * min(sig_skin, sig_cap, sig_web)  # representative scale

    buck_list = u_skin_buck + u_cap_buck + u_web_buck
    buck_util = np.array(buck_list) if buck_list else np.array([0.0])
    buckling_margin = ks_aggregate(buck_util, ks_rho) - 1.0

    # tip transverse displacement
    tip_nodes = mesh["station_nodes"][-1]
    w_tip = float(np.max(np.abs([u[n * 6 + 2] for n in tip_nodes])))

    return {
        "skin_mass": skin_mass, "rib_mass": rib_mass, "cap_mass": cap_mass,
        "web_mass": web_mass,
        "failure_margin": failure_margin, "buckling_margin": buckling_margin,
        "sigma_max": sigma_max, "w_tip": w_tip,
        "max_skin_util": max(u_skin), "max_cap_util": max(u_cap),
        "max_web_util": max(u_web) if u_web else 0.0,
    }


class WingboxModel(om.ExplicitComponent):
    """Shell + beam wingbox FEM sizing component."""

    # Finite penalties substituted when the FE solve returns non-finite results
    # (degenerate mesh): a clearly-too-heavy wing and a strongly violated
    # feasibility margin, so the optimiser is steered away without NaN crashing
    # the nonlinear solver.
    _PENALTY_MASS = 1.0e3      # kg (both half-wings); ~3 orders above feasible
    _PENALTY_MARGIN = 1.0e2    # dimensionless KS utilisation - 1 (>> 0)

    def initialize(self):
        self.options.declare("n_span", types=int, default=10)
        self.options.declare("n_chord", types=int, default=6)
        self.options.declare("ks_rho", types=float, default=100.0)
        self.options.declare("fs_ratio", types=float, default=0.15,
                             desc="Front-spar chordwise position (fraction of chord).")
        self.options.declare("rs_ratio", types=float, default=0.65,
                             desc="Rear-spar chordwise position (fraction of chord).")
        self.options.declare("poisson", types=float, default=0.3)
        self.options.declare("kc_skin", types=float, default=4.0,
                             desc="Plate buckling coefficient for the skin panels "
                                  "(~4 for a long simply-supported plate).")
        self.options.declare("cap_fixity", types=float, default=1.0,
                             desc="Euler end-fixity coefficient for cap column "
                                  "buckling (1.0 = pinned-pinned).")
        self.options.declare("ks_web", types=float, default=5.35,
                             desc="Shear buckling coefficient for the spar-web "
                                  "panels (~5.35 for a long simply-supported "
                                  "plate in shear).")

    def setup(self):
        n_stations = self.options["n_span"] + 1

        # Geometry
        self.add_input("data:geometry:wing:span", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:chord", val=np.nan, units="m")
        self.add_input("data:geometry:wing:root:thickness", val=np.nan, units="m")
        self.add_input("data:geometry:wing:tip:thickness", val=np.nan, units="m")

        # Materials
        self.add_input("data:structures:wing:skin:material:E", val=70.0e9, units="Pa")
        self.add_input("data:structures:wing:skin:material:stress:max", val=300.0e6, units="Pa")
        self.add_input("data:structures:wing:spar:material:E", val=70.0e9, units="Pa")
        self.add_input("data:structures:wing:spar:material:stress:max", val=600.0e6, units="Pa")
        self.add_input("data:structures:wing:skin:material:density", val=1600.0, units="kg/m**3")
        self.add_input("data:weight:airframe:wing:ribs:density", val=1600.0, units="kg/m**3")
        self.add_input("data:weight:airframe:wing:spar:density", val=1600.0, units="kg/m**3")

        # Design variables
        self.add_input("data:structures:wing:skin:thickness:root", val=0.0008, units="m")
        self.add_input("data:structures:wing:skin:thickness:tip", val=0.0004, units="m")
        self.add_input("data:structures:wing:spar:cap_area:root", val=4.0e-5, units="m**2")
        self.add_input("data:structures:wing:spar:cap_area:tip", val=1.0e-5, units="m**2")
        self.add_input("data:structures:wing:spar:web_thickness:root", val=0.0006, units="m")
        self.add_input("data:structures:wing:spar:web_thickness:tip", val=0.0004, units="m")
        self.add_input("data:structures:wing:ribs:thickness", val=0.0006, units="m")

        # Load (from WingLoadDistribution with n_elements = n_span)
        self.add_input("data:loads:wing:q_nodes",
                       val=np.zeros(n_stations), shape=n_stations, units="N/m")

        # Outputs
        self.add_output("data:weight:airframe:wing:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:spar:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:skin:mass", units="kg", lower=0.0)
        self.add_output("data:weight:airframe:wing:ribs:mass", units="kg", lower=0.0)
        self.add_output("data:constraints:structures:wing:failure_margin", units=None)
        self.add_output("data:constraints:structures:wing:buckling_margin", units=None)
        self.add_output("data:loads:wing:sigma_max", units="Pa")
        self.add_output("data:loads:wing:w_tip", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        opt = self.options
        nu = opt["poisson"]
        semi_span = 0.5 * float(inputs["data:geometry:wing:span"][0])

        mesh = build_wingbox_mesh(
            semi_span=semi_span,
            c_root=float(inputs["data:geometry:wing:root:chord"][0]),
            c_tip=float(inputs["data:geometry:wing:tip:chord"][0]),
            h_root=float(inputs["data:geometry:wing:root:thickness"][0]),
            h_tip=float(inputs["data:geometry:wing:tip:thickness"][0]),
            fs_ratio=opt["fs_ratio"], rs_ratio=opt["rs_ratio"],
            n_span=opt["n_span"], n_chord=opt["n_chord"],
        )

        res = solve_wingbox(
            mesh,
            E_skin=float(inputs["data:structures:wing:skin:material:E"][0]),
            nu=nu,
            rho_skin=float(inputs["data:structures:wing:skin:material:density"][0]),
            t_skin_root=float(inputs["data:structures:wing:skin:thickness:root"][0]),
            t_skin_tip=float(inputs["data:structures:wing:skin:thickness:tip"][0]),
            t_rib=float(inputs["data:structures:wing:ribs:thickness"][0]),
            rho_rib=float(inputs["data:weight:airframe:wing:ribs:density"][0]),
            E_cap=float(inputs["data:structures:wing:spar:material:E"][0]),
            nu_cap=nu,
            rho_cap=float(inputs["data:weight:airframe:wing:spar:density"][0]),
            A_cap_root=float(inputs["data:structures:wing:spar:cap_area:root"][0]),
            A_cap_tip=float(inputs["data:structures:wing:spar:cap_area:tip"][0]),
            sig_skin=float(inputs["data:structures:wing:skin:material:stress:max"][0]),
            sig_cap=float(inputs["data:structures:wing:spar:material:stress:max"][0]),
            q_stations=np.asarray(inputs["data:loads:wing:q_nodes"], dtype=float),
            ks_rho=opt["ks_rho"],
            # Spar web: same material as the caps, with its own taperable gauge.
            t_web_root=float(inputs["data:structures:wing:spar:web_thickness:root"][0]),
            t_web_tip=float(inputs["data:structures:wing:spar:web_thickness:tip"][0]),
            E_web=float(inputs["data:structures:wing:spar:material:E"][0]),
            rho_web=float(inputs["data:weight:airframe:wing:spar:density"][0]),
            sig_web=float(inputs["data:structures:wing:spar:material:stress:max"][0]),
            kc_skin=opt["kc_skin"],
            cap_fixity=opt["cap_fixity"],
            ks_web=opt["ks_web"],
        )

        skin_m = 2.0 * res["skin_mass"]
        rib_m = 2.0 * res["rib_mass"]
        # Spar mass = caps (flanges) + webs (shear panels), both half-wings.
        spar_m = 2.0 * (res["cap_mass"] + res["web_mass"])
        wing_m = skin_m + rib_m + spar_m
        fail_margin = res["failure_margin"]
        buck_margin = res["buckling_margin"]
        sigma_max = res["sigma_max"]
        w_tip = res["w_tip"]

        # Robustness: a near-singular shell FE solve (degenerate mesh probed by
        # the optimiser) can yield non-finite displacements -> non-finite mass /
        # stresses, which would poison the whole MDA with NaN. Emit a strong but
        # *finite* infeasibility penalty instead so the optimiser is pushed away
        # from the bad region rather than crashing the nonlinear solver.
        if not np.all(np.isfinite([skin_m, rib_m, spar_m, wing_m,
                                   fail_margin, buck_margin, sigma_max, w_tip])):
            skin_m = rib_m = spar_m = self._PENALTY_MASS / 3.0
            wing_m = self._PENALTY_MASS
            fail_margin = buck_margin = self._PENALTY_MARGIN
            sigma_max = 0.0
            w_tip = 0.0

        outputs["data:weight:airframe:wing:skin:mass"] = skin_m
        outputs["data:weight:airframe:wing:ribs:mass"] = rib_m
        outputs["data:weight:airframe:wing:spar:mass"] = spar_m
        outputs["data:weight:airframe:wing:mass"] = wing_m
        outputs["data:constraints:structures:wing:failure_margin"] = fail_margin
        outputs["data:constraints:structures:wing:buckling_margin"] = buck_margin
        outputs["data:loads:wing:sigma_max"] = sigma_max
        outputs["data:loads:wing:w_tip"] = w_tip
