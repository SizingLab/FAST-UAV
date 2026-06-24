"""
Spanwise load distribution for the wing-spar FEM.

This component fuses the FAST-GA aerodynamic_loads + structural_loads logic
into a single UAV-tailored component. Given:

  - VLM-derived spanwise unit CL distribution at a reference flight state,
  - the critical load factor and aircraft mass for the sizing case,
  - the wing self-weight and any wing-mounted point masses,

it builds the net distributed transverse load q(y) [N/m] along the half-span
and resamples it onto the FE node mesh expected by BeamFEM.

Sign convention: q > 0 is up. Lift is positive, weight is negative.

Pipeline (matches FAST-GA logic, stripped of fuel-tank / landing-gear /
multi-engine bookkeeping that doesn't apply to a tractor monoprop UAV):

    1. Scale the unit CL_vector to match the target lift coefficient
       cl_target = n_ult * m * g / (q_dyn * S_ref)
    2. Multiply by chord(y) and dynamic pressure to get q_aero(y) in N/m.
    3. Subtract chord-weighted wing self-weight relief, factored by n_ult.
    4. Subtract any point-mass relief (battery, payload, servos, ...),
       smeared over a small spanwise width to avoid Dirac singularities.
    5. Interpolate onto the n_elements + 1 FE nodes of the BeamFEM.
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


# ---------------------------------------------------------------------------
# Pure-NumPy core
# ---------------------------------------------------------------------------

def build_q_distribution(
    semi_span: float,
    y_vlm: np.ndarray,
    cl_vlm: np.ndarray,
    chord_vlm: np.ndarray,
    cl_ref: float,
    n_ult: float,
    mass: float,
    wing_area: float,
    air_density: float,
    v_tas: float,
    wing_mass: float,
    point_mass_y: np.ndarray,
    point_mass_m: np.ndarray,
    point_mass_smear_width: float,
    n_elements: int,
    g: float = 9.81,
):
    """
    Build q(y) [N/m] on the FE node mesh.

    Parameters
    ----------
    semi_span : half-span b/2 [m]
    y_vlm, cl_vlm, chord_vlm : VLM spanwise stations [m, -, m].
        y_vlm runs from 0 (root) to b/2 (tip). Must be sorted ascending.
        cl_vlm is the local lift coefficient at the reference state.
        chord_vlm is local chord at each station [m].
    cl_ref : reference wing CL at which the VLM was evaluated [-]
    n_ult : ultimate load factor [-]
    mass : aircraft mass at the sizing case [kg]
    wing_area : wing reference area S_ref [m^2]
    air_density : rho at the sizing flight condition [kg/m^3]
    v_tas : true airspeed at the sizing case [m/s]
    wing_mass : total wing structural mass (both halves) [kg]
    point_mass_y : y-coordinates of wing-mounted point masses [m]
    point_mass_m : masses [kg], same shape as point_mass_y
    point_mass_smear_width : width over which to spread each point mass [m]
        Avoids Dirac-like spikes that the FE mesh cannot resolve. A few
        percent of the semi-span is a sensible default.
    n_elements : number of beam elements (must match the BeamFEM mesh)

    Returns
    -------
    q_nodes : net distributed transverse load at FE nodes [N/m], length n_elements + 1.
    """
    n_nodes = n_elements + 1
    y_fe = np.linspace(0.0, semi_span, n_nodes)

    # --- 1. Aerodynamic part -----------------------------------------------
    q_dyn = 0.5 * air_density * v_tas**2
    cl_target = n_ult * mass * g / (q_dyn * wing_area)

    # Scale the unit-state CL distribution to the target CL.
    # FAST-GA caps the scaling factor implicitly via cl_ref; for robustness
    # we just guard against an accidental zero reference.
    scale = cl_target / max(cl_ref, 1e-9)
    cl_local = scale * cl_vlm

    # q_aero(y) = q_dyn * cl_local(y) * chord(y)   [N/m]
    q_aero_vlm = q_dyn * cl_local * chord_vlm

    # Resample onto the FE mesh.
    q_aero_fe = np.interp(y_fe, y_vlm, q_aero_vlm)

    # --- 2. Wing self-weight relief, distributed proportional to chord -----
    # Half-wing weight (in newtons), factored by n_ult.
    half_wing_weight = 0.5 * wing_mass * g * n_ult

    # Chord at FE nodes (interpolated from VLM mesh).
    chord_fe = np.interp(y_fe, y_vlm, chord_vlm)
    chord_integral = _trapz(chord_fe, y_fe)
    if chord_integral > 0.0:
        q_wing_relief = half_wing_weight * chord_fe / chord_integral
    else:
        q_wing_relief = np.zeros_like(y_fe)

    # --- 3. Point-mass relief, smeared as raised-cosine bumps --------------
    q_point_relief = np.zeros_like(y_fe)
    half_w = 0.5 * point_mass_smear_width
    for y_p, m_p in zip(np.atleast_1d(point_mass_y), np.atleast_1d(point_mass_m)):
        if m_p <= 0.0:
            continue
        # Raised cosine with unit integral, centered at y_p, width smear_width.
        # phi(y) = (1/W)(1 + cos(pi*(y - y_p)/half_w)) for |y - y_p| < half_w
        # Integral from y_p - half_w to y_p + half_w  =  1.
        d = y_fe - y_p
        mask = np.abs(d) < half_w
        phi = np.zeros_like(y_fe)
        phi[mask] = (1.0 / point_mass_smear_width) * (
            1.0 + np.cos(np.pi * d[mask] / half_w)
        )
        q_point_relief += m_p * g * n_ult * phi

    # --- 4. Net distributed load -------------------------------------------
    q_nodes = q_aero_fe - q_wing_relief - q_point_relief
    return q_nodes


# ---------------------------------------------------------------------------
# OpenMDAO wrapper
# ---------------------------------------------------------------------------

if _HAS_OPENMDAO:

    class WingLoadDistribution(om.ExplicitComponent):
        """
        OpenMDAO wrapper around build_q_distribution.

        VLM mesh size (n_vlm) and FE mesh size (n_elements) are fixed at
        instantiation. point_masses_n is also fixed; pass 0 if there are
        no wing-mounted point masses.
        """

        def initialize(self):
            self.options.declare("n_elements",   types=int, default=20)
            self.options.declare("n_vlm",        types=int, default=20,
                                 desc="Number of VLM spanwise stations.")
            self.options.declare("n_point_mass", types=int, default=0,
                                 desc="Number of wing-mounted point masses.")
            self.options.declare("smear_width_ratio", types=float, default=0.05,
                                 desc="Point-mass smear width as a fraction of semi-span.")

        def setup(self):
            n_fe   = self.options["n_elements"] + 1
            n_vlm  = self.options["n_vlm"]
            n_pm   = self.options["n_point_mass"]

            # Geometry
            self.add_input("data:geometry:wing:semi_span", val=1.0, units="m")
            self.add_input("data:geometry:wing:area",      val=0.5, units="m**2")

            # VLM outputs
            self.add_input("data:aerodynamics:wing:Y_vector",
                           val=np.linspace(0.0, 1.0, n_vlm), shape=n_vlm, units="m")
            self.add_input("data:aerodynamics:wing:CL_vector",
                           val=np.zeros(n_vlm), shape=n_vlm)
            self.add_input("data:aerodynamics:wing:chord_vector",
                           val=np.zeros(n_vlm), shape=n_vlm, units="m")
            self.add_input("data:aerodynamics:wing:CL_ref", val=1.0)

            # Sizing case
            self.add_input("data:loads:n_ult",       val=4.5)
            self.add_input("data:loads:sizing_mass", val=10.0, units="kg")
            self.add_input("data:loads:air_density", val=1.225, units="kg/m**3")
            self.add_input("data:TLAR:v_tas_sizing", val=30.0, units="m/s")

            # Mass relief
            self.add_input("data:weight:airframe:wing:mass", val=0.5, units="kg")

            if n_pm > 0:
                self.add_input("data:weight:airframe:wing:point_mass:y",
                               val=np.zeros(n_pm), shape=n_pm, units="m")
                self.add_input("data:weight:airframe:wing:point_mass:mass",
                               val=np.zeros(n_pm), shape=n_pm, units="kg")

            # Output: distributed load at FE nodes
            self.add_output("data:loads:wing:q_nodes",
                            val=np.zeros(n_fe), shape=n_fe, units="N/m")

            self.declare_partials("*", "*", method="fd")

        def compute(self, inputs, outputs):
            n_pm = self.options["n_point_mass"]
            if n_pm > 0:
                pm_y = inputs["data:weight:airframe:wing:point_mass:y"]
                pm_m = inputs["data:weight:airframe:wing:point_mass:mass"]
            else:
                pm_y = np.array([0.0])
                pm_m = np.array([0.0])

            semi_span = float(inputs["data:geometry:wing:semi_span"])

            outputs["data:loads:wing:q_nodes"] = build_q_distribution(
                semi_span      = semi_span,
                y_vlm          = inputs["data:aerodynamics:wing:Y_vector"],
                cl_vlm         = inputs["data:aerodynamics:wing:CL_vector"],
                chord_vlm      = inputs["data:aerodynamics:wing:chord_vector"],
                cl_ref         = float(inputs["data:aerodynamics:wing:CL_ref"]),
                n_ult          = float(inputs["data:loads:n_ult"]),
                mass           = float(inputs["data:loads:sizing_mass"]),
                wing_area      = float(inputs["data:geometry:wing:area"]),
                air_density    = float(inputs["data:loads:air_density"]),
                v_tas          = float(inputs["data:TLAR:v_tas_sizing"]),
                wing_mass      = float(inputs["data:weight:airframe:wing:mass"]),
                point_mass_y   = pm_y,
                point_mass_m   = pm_m,
                point_mass_smear_width = self.options["smear_width_ratio"] * semi_span,
                n_elements     = self.options["n_elements"],
            )
