"""
Fixed Wing Airframe Aerodynamics
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import (
    add_subsystem_with_deviation,
    add_model_deviation,
)


class AerodynamicsFW(om.Group):
    """
    Group containing the airframe aerodynamics calculation
    """

    def setup(self):
        # self.add_subsystem("parasitic_drag", ParasiticDrag(), promotes=["*"])
        add_subsystem_with_deviation(
            self,
            "parasitic_drag",
            ParasiticDrag(),
            uncertain_outputs={"data:airframe:aerodynamics:CD0": None},
        )
        self.add_subsystem("lift_to_drag", MaxLiftToDrag(), promotes=["*"])
        self.add_subsystem("constraints", ParasiticDragConstraint(), promotes=["*"])


class AirframeAerodynamicsModel:
    """
    Aerodynamics model for the airframe
    """
    @staticmethod
    def friction_flatplate(V_air, L, nu_air, a_air):
        re = V_air * L / nu_air  # Reynolds number [-]
        mach = V_air / a_air  # mach number [-]
        cf_turb = 0.455 / (np.log10(re) ** 2.58 * (1 + 0.144 * mach ** 2) ** 0.65)  # flat-plate skin friction [-]
        return cf_turb


class ParasiticDrag(om.ExplicitComponent):
    """
    Computes Airframe Aerodynamics
    """

    def setup(self):
        self.add_input("mission:design_mission:cruise:atmosphere:viscosity", val=np.nan, units="m**2/s")
        self.add_input("mission:design_mission:cruise:atmosphere:speedofsound", val=np.nan, units="m/s")
        self.add_input("data:airframe:wing:tc", val=0.15, units=None)
        self.add_input("data:airframe:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:airframe:tail:horizontal:MAC:length", val=np.nan, units="m")
        self.add_input("data:airframe:tail:vertical:MAC:length", val=np.nan, units="m")
        self.add_input("data:airframe:fuselage:length", val=np.nan, units="m")
        self.add_input("data:airframe:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:airframe:fuselage:fineness", val=5.0, units=None)
        self.add_input("data:airframe:wing:surface", val=np.nan, units="m**2")
        self.add_input("data:airframe:tail:horizontal:surface", val=np.nan, units="m**2")
        self.add_input("data:airframe:tail:vertical:surface", val=np.nan, units="m**2")
        self.add_input("mission:design_mission:cruise:speed", val=np.nan, units="m/s")
        self.add_output("data:airframe:aerodynamics:CD0", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        nu_air = inputs["mission:design_mission:cruise:atmosphere:viscosity"]
        a_air = inputs["mission:design_mission:cruise:atmosphere:speedofsound"]
        tc_ratio = inputs["data:airframe:wing:tc"]
        c_MAC_w = inputs["data:airframe:wing:MAC:length"]
        c_MAC_ht = inputs["data:airframe:tail:horizontal:MAC:length"]
        c_MAC_vt = inputs["data:airframe:tail:vertical:MAC:length"]
        l_fus = inputs["data:airframe:fuselage:length"]
        d_fus_mid = inputs["data:airframe:fuselage:diameter:mid"]
        lmbda_f = inputs["data:airframe:fuselage:fineness"]
        S_w = inputs["data:airframe:wing:surface"]
        S_ht = inputs["data:airframe:tail:horizontal:surface"]
        S_vt = inputs["data:airframe:tail:vertical:surface"]
        V_cruise = inputs["mission:design_mission:cruise:speed"]

        # Friction coefficients assuming cruise conditions
        cf_wing = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_w, nu_air, a_air)
        cf_ht = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_ht, nu_air, a_air)
        cf_vt = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_vt, nu_air, a_air)
        cf_fus = AirframeAerodynamicsModel.friction_flatplate(V_cruise, l_fus, nu_air, a_air)

        # Form drag factors
        FF_w = (1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio ** 4) * (1.34 * (V_cruise / a_air) ** 0.18)
        FF_ht = (1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio ** 4) * (1.34 * (V_cruise / a_air) ** 0.18)
        FF_vt = (1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio ** 4) * (1.34 * (V_cruise / a_air) ** 0.18)
        FF_fus = 1 + 60 / lmbda_f ** 3 + lmbda_f / 400

        # Wetted areas
        S_wet_w = 2 * S_w
        S_wet_ht = 2 * S_ht
        S_wet_vt = 2 * S_vt
        S_wet_fus = np.pi * d_fus_mid * l_fus * (1 - 2 / lmbda_f) ** (2 / 3) * (1 + 1 / (lmbda_f ** 2))

        # Parasitic drag coefficient
        CD_0 = (1 / S_w) * (cf_wing * S_wet_w * FF_w
                            + cf_ht * S_wet_ht * FF_ht
                            + cf_vt * S_wet_vt * FF_vt
                            + cf_fus * S_wet_fus * FF_fus)

        outputs["data:airframe:aerodynamics:CD0"] = CD_0


class MaxLiftToDrag(om.ExplicitComponent):
    """
    Computes maximum lift to drag ratio and speed for best L/D.
    """

    def setup(self):
        self.add_input("data:airframe:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:airframe:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:airframe:aerodynamics:LD:max", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        CD_0 = inputs["data:airframe:aerodynamics:CD0"]
        K = inputs["data:airframe:aerodynamics:CDi:K"]

        LDmax = 0.5 / np.sqrt(CD_0 * K)  # [-] max lift-to-drag ratio
        # V_opt = np.sqrt(2 * m_tot * g / rho_air / S_w * np.sqrt(K / CD_0))

        outputs["data:airframe:aerodynamics:LD:max"] = LDmax


class ParasiticDragConstraint(om.ExplicitComponent):
    """
    Computes Aerodynamics Constraints
    """

    def setup(self):
        self.add_input("data:airframe:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:airframe:aerodynamics:CD0", val=np.nan, units=None)
        self.add_output("data:airframe:aerodynamics:constraints:CD0:consistency", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        CD_0_guess = inputs["data:airframe:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:airframe:aerodynamics:CD0"]

        CD0_cnstr = (CD_0_guess - CD_0) / CD_0

        outputs["data:airframe:aerodynamics:constraints:CD0:consistency"] = CD0_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        CD_0_guess = inputs["data:airframe:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:airframe:aerodynamics:CD0"]

        partials["data:airframe:aerodynamics:constraints:CD0:consistency",
                 "data:airframe:aerodynamics:CD0:guess"] = 1 / CD_0
        partials["data:airframe:aerodynamics:constraints:CD0:consistency",
                 "data:airframe:aerodynamics:CD0"] = - CD_0_guess / CD_0 ** 2


