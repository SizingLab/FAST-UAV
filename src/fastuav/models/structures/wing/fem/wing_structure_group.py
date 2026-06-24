"""
Top-level wing-structure group: load distribution -> FEM -> stress + mass.

Wiring (everything promoted to the FAST-UAV-style top-level namespace):

    WingLoadDistribution  --q_nodes-->  BeamFEM  --M_bending,
                                                  R_nodes, t_nodes-->
                                                                StressRecovery
                                                                SparMass
"""

from __future__ import annotations

try:
    import openmdao.api as om
    _HAS_OPENMDAO = True
except ImportError:
    _HAS_OPENMDAO = False

if _HAS_OPENMDAO:

    from .beam_element import BeamElement3D
    from .beam_fem import BeamFEM
    from .load_distribution import WingLoadDistribution
    from .spar_mass import SparMass
    from .stress_recovery import StressRecovery


    class WingStructure(om.Group):
        """
        Inputs (top-level promotions):
          data:geometry:wing:semi_span                [m]
          data:geometry:wing:area                     [m^2]
          data:geometry:wing:spar:R_root, R_tip       [m]
          data:geometry:wing:spar:t_root, t_tip       [m]
          data:aerodynamics:wing:Y_vector             [m]
          data:aerodynamics:wing:CL_vector            [-]
          data:aerodynamics:wing:chord_vector         [m]
          data:aerodynamics:wing:CL_ref               [-]
          data:loads:n_ult, sizing_mass, air_density  [-, kg, kg/m^3]
          data:TLAR:v_tas_sizing                      [m/s]
          data:weight:airframe:wing:mass              [kg]
          data:material:spar:E, G, sigma_allow,
                              safety_factor, density  [Pa, Pa, Pa, -, kg/m^3]
          (optional) point_mass:y, point_mass:mass    [m, kg]

        Outputs:
          data:loads:wing:q_nodes, M_bending,
                          sigma_bending, sigma_max_ks [N/m, N*m, Pa, Pa]
          data:loads:wing:w_tip                       [m]
          data:constraints:wing:failure_margin        [Pa]
          data:weight:airframe:wing:spar:mass         [kg]
        """

        def initialize(self):
            self.options.declare("n_elements",   types=int, default=20)
            self.options.declare("n_vlm",        types=int, default=20)
            self.options.declare("n_point_mass", types=int, default=0)
            self.options.declare("element_class", default=BeamElement3D,
                                 desc="Beam element formulation (default: BEAM3 / "
                                      "BeamElement3D, the 6-DOF spatial frame).")
            self.options.declare("spar_model",   default="pipe",
                                 values=["pipe", "I_beam"],
                                 desc="Spar cross-section configuration.")
            self.options.declare("ks_rho",       types=float, default=100.0)

        def setup(self):
            n_elem = self.options["n_elements"]
            n_vlm  = self.options["n_vlm"]
            n_pm   = self.options["n_point_mass"]
            elem_cls = self.options["element_class"]
            spar_model = self.options["spar_model"]
            ks_rho = self.options["ks_rho"]

            self.add_subsystem(
                "load_distribution",
                WingLoadDistribution(n_elements=n_elem, n_vlm=n_vlm,
                                     n_point_mass=n_pm),
                promotes=["*"],
            )
            self.add_subsystem(
                "beam_fem",
                BeamFEM(n_elements=n_elem, element_class=elem_cls,
                        spar_model=spar_model),
                promotes=["*"],
            )
            self.add_subsystem(
                "stress_recovery",
                StressRecovery(n_elements=n_elem, ks_rho=ks_rho,
                               spar_model=spar_model),
                promotes=["*"],
            )
            self.add_subsystem(
                "spar_mass",
                SparMass(n_elements=n_elem, spar_model=spar_model),
                promotes=["*"],
            )
