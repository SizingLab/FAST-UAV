"""
Fixed Wing Aerodynamics (external)
"""

import fastoad.api as oad
import numpy as np
import openmdao.api as om
from stdatm import AtmosphereWithPartials

from fastuav.constants import FW_PROPULSION
from fastuav.models.aerodynamics.external.vlm import ComputeAeroVLM
from fastuav.utils.uncertainty import (
    add_subsystem_with_deviation,
)

from .constants import (
    DEFAULT_INPUT_AOA,
)


class AirframeAerodynamicsModel:
    """
    Aerodynamics model for the airframe
    """

    @staticmethod
    def friction_flatplate(V_air, L, nu_air, a_air):
        re = V_air * L / nu_air  # Reynolds number [-]
        # mach = V_air / a_air  # mach number [-]
        cf_turb = 0.455 / (
            np.log10(re) ** 2.58  # * (1 + 0.144 * mach**2) ** (-0.65)
        )  # flat-plate skin friction [-]
        return cf_turb

    @staticmethod
    def friction_flatplate_derivatives(V_air, L, nu_air, a_air):
        """
        Partials of friction coefficient wrt (V_air, L, nu_air, a_air).
        Returns a dict with keys: 'dV', 'dL', 'dnu'
        (a_air not used in current formula, but kept for API consistency)
        """
        re = V_air * L / nu_air
        log10_re = np.log10(re)

        # dcf/d(log10(Re))
        dcf_dlog10re = -0.455 * 2.58 / (log10_re**3.58)

        # d(log10(Re))/dRe
        dlog10re_dre = 1.0 / (re * np.log(10))

        # dRe/d(...)
        dre_dV = L / nu_air
        dre_dL = V_air / nu_air
        dre_dnu = -V_air * L / (nu_air**2)

        # Chain rule
        dcf_dV = dcf_dlog10re * dlog10re_dre * dre_dV
        dcf_dL = dcf_dlog10re * dlog10re_dre * dre_dL
        dcf_dnu = dcf_dlog10re * dlog10re_dre * dre_dnu

        return {"dV": dcf_dV, "dL": dcf_dL, "dnu": dcf_dnu}


@oad.RegisterOpenMDAOSystem("fastuav.aerodynamics.fixedwing")
class Aerodynamics(om.Group):
    """
    Group containing the external aerodynamics calculations
    """

    def setup(self):

        # Parasitic drag calculations
        parasitic_drag = self.add_subsystem("parasitic_drag", om.Group(), promotes=["*"])
        parasitic_drag.add_subsystem("wing", WingParasiticDrag(), promotes=["*"])
        parasitic_drag.add_subsystem(
            "horizontal_tail", TailParasiticDrag(tail="horizontal"), promotes=["*"]
        )
        parasitic_drag.add_subsystem(
            "vertical_tail", TailParasiticDrag(tail="vertical"), promotes=["*"]
        )
        parasitic_drag.add_subsystem("fuselage", FuselageParasiticDrag(), promotes=["*"])
        add_subsystem_with_deviation(
            parasitic_drag,
            "parasitic_drag",
            ParasiticDrag(),
            uncertain_outputs={"data:aerodynamics:CD0": None},
        )
        parasitic_drag.add_subsystem("constraint", ParasiticDragConstraint(), promotes=["*"])

        # Lift to drag
        self.add_subsystem("lift_to_drag", MaxLiftToDrag(), promotes=["*"])


@oad.RegisterOpenMDAOSystem(
    "fastuav.aerodynamics.highspeed.legacy.fixedwing"
)  # , domain=ModelDomain.AERODYNAMICS
class AerodynamicsHighSpeed(om.Group):
    """Models for high speed aerodynamics."""

    def initialize(self):
        self.options.declare("result_folder_path", default="", types=str)
        self.options.declare("openvsp_exe_path", default="", types=str, allow_none=True)
        self.options.declare("airfoil_folder_path", default=None, types=str, allow_none=True)
        self.options.declare("wing_airfoil", default="naca23012.af", types=str, allow_none=True)
        self.options.declare("htp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("vtp_airfoil", default="naca0012.af", types=str, allow_none=True)
        self.options.declare("input_angle_of_attack", default=DEFAULT_INPUT_AOA, types=float)
        self.options.declare("use_neuralfoil", default=False, types=bool)

    def setup(self):
        self.add_subsystem(
            "aero_vlm",
            ComputeAeroVLM(
                low_speed_aero=False,
                result_folder_path=self.options["result_folder_path"],
                # compute_mach_interpolation=True,
                airfoil_folder_path=self.options["airfoil_folder_path"],
                wing_airfoil_file=self.options["wing_airfoil"],
                htp_airfoil_file=self.options["htp_airfoil"],
                input_angle_of_attack=self.options["input_angle_of_attack"],
                use_neuralfoil=self.options["use_neuralfoil"],
            ),
            promotes=["*"],
        )

        # Parasitic drag calculations
        parasitic_drag = self.add_subsystem("parasitic_drag", om.Group(), promotes=["*"])
        parasitic_drag.add_subsystem("wing", WingParasiticDrag(), promotes=["*"])
        parasitic_drag.add_subsystem(
            "horizontal_tail", TailParasiticDrag(tail="horizontal"), promotes=["*"]
        )
        parasitic_drag.add_subsystem(
            "vertical_tail", TailParasiticDrag(tail="vertical"), promotes=["*"]
        )
        parasitic_drag.add_subsystem("fuselage", FuselageParasiticDrag(), promotes=["*"])
        add_subsystem_with_deviation(
            parasitic_drag,
            "parasitic_drag",
            ParasiticDrag(),
            uncertain_outputs={"data:aerodynamics:CD0": None},
        )
        parasitic_drag.add_subsystem("constraint", ParasiticDragConstraintVLM(), promotes=["*"])


class ParasiticDrag(om.ExplicitComponent):
    """
    Sums up the individual parasitic drags at cruise conditions
    """

    def setup(self):
        self.add_input("data:aerodynamics:CD0:wing", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:tail:horizontal", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:tail:vertical", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0:fuselage", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CD0", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        outputs["data:aerodynamics:CD0"] = (
            inputs["data:aerodynamics:CD0:wing"]
            + inputs["data:aerodynamics:CD0:tail:horizontal"]
            + inputs["data:aerodynamics:CD0:tail:vertical"]
            + inputs["data:aerodynamics:CD0:fuselage"]
        )

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials["data:aerodynamics:CD0", "data:aerodynamics:CD0:wing"] = 1.0
        partials["data:aerodynamics:CD0", "data:aerodynamics:CD0:tail:horizontal"] = 1.0
        partials["data:aerodynamics:CD0", "data:aerodynamics:CD0:tail:vertical"] = 1.0
        partials["data:aerodynamics:CD0", "data:aerodynamics:CD0:fuselage"] = 1.0


class ParasiticDragConstraintVLM(om.ExplicitComponent):
    """
    Parasitic drag consistency constraint, to ensure that the initial guess equals the estimated value.
    """

    def setup(self):
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("optimization:variables:aerodynamics:CDi:K:guess", val=np.nan, units=None)
        self.add_input("data:aerodynamics:wing:cruise:CDi:k", val=np.nan, units=None)
        self.add_output("optimization:constraints:aerodynamics:CD0:consistency", units=None)
        self.add_output("optimization:constraints:aerodynamics:CDi:K:consistency", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]
        k_CD_i_guess = inputs["optimization:variables:aerodynamics:CDi:K:guess"]
        CD_0_com = inputs["data:aerodynamics:CD0"]
        k_CD_i = inputs["data:aerodynamics:wing:cruise:CDi:k"]

        CD0_cnstr = (CD_0_guess - CD_0_com) / CD_0_com
        k_cnstr = (k_CD_i_guess - k_CD_i) / k_CD_i

        outputs["optimization:constraints:aerodynamics:CD0:consistency"] = CD0_cnstr
        outputs["optimization:constraints:aerodynamics:CDi:K:consistency"] = k_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]
        CD_0_com = inputs["data:aerodynamics:CD0"]
        k_CD_i_guess = inputs["optimization:variables:aerodynamics:CDi:K:guess"]
        k_CD_i = inputs["data:aerodynamics:wing:cruise:CDi:k"]

        partials[
            "optimization:constraints:aerodynamics:CD0:consistency",
            "optimization:variables:aerodynamics:CD0:guess",
        ] = 1 / CD_0_com
        partials[
            "optimization:constraints:aerodynamics:CD0:consistency", "data:aerodynamics:CD0"
        ] = (  # , "data:aerodynamics:CD0"#data:aerodynamics:aircraft:cruise:CD0
            -CD_0_guess / CD_0_com**2
        )
        partials[
            "optimization:constraints:aerodynamics:CDi:K:consistency",
            "optimization:variables:aerodynamics:CDi:K:guess",
        ] = 1 / k_CD_i
        partials[
            "optimization:constraints:aerodynamics:CDi:K:consistency",
            "data:aerodynamics:wing:cruise:CDi:k",
        ] = -k_CD_i_guess / k_CD_i**2


class WingParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of the wing at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s"
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:wing", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # Geometry
        propulsion_id = self.options["propulsion_id"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        c_MAC_w = inputs["data:geometry:wing:MAC:length"]
        S_w = S_ref = inputs["data:geometry:wing:surface"]

        # Flight parameters
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        atm.true_airspeed = V_cruise
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Friction coefficients assuming cruise conditions
        cf_wing = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_w, nu_air, a_air)

        # Form drag factor
        FF_w = 1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio**4
        # * (1.34 * (V_cruise / a_air) ** 0.18)  # compressibility effects

        # Wetted area
        S_wet_w = 2 * S_w

        # Parasitic drag coefficient
        CD_0_wing = (cf_wing * FF_w * S_wet_w) / S_ref

        outputs["data:aerodynamics:CD0:wing"] = CD_0_wing

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        c_MAC_w = inputs["data:geometry:wing:MAC:length"]
        # S_w = inputs["data:geometry:wing:surface"]
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]

        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Compute cf and its derivatives
        cf_wing = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_w, nu_air, a_air)
        dcf = AirframeAerodynamicsModel.friction_flatplate_derivatives(
            V_cruise, c_MAC_w, nu_air, a_air
        )

        # Form factor: FF_w = 1 + 0.6/0.3 * tc + 100*tc^4 = 1 + 2*tc + 100*tc^4
        FF_w = 1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio**4
        dFF_dtc = 2.0 + 400 * tc_ratio**3

        # Wetted area
        # S_wet_w = 2 * S_w
        # S_ref = S_w

        # CD_0_wing = (cf_wing * FF_w * S_wet_w) / S_ref = cf_wing * FF_w * 2
        out = "data:aerodynamics:CD0:wing"

        # dCD0/dtc_ratio = cf_wing * dFF/dtc
        partials[out, "data:geometry:wing:tc"] = cf_wing * dFF_dtc * 2

        # dCD0/dS_w: CD0 = cf*FF*2 (independent of S_w), so = 0
        partials[out, "data:geometry:wing:surface"] = 0.0

        # dCD0/dc_MAC = dcf/dc_MAC * FF_w * 2
        partials[out, "data:geometry:wing:MAC:length"] = dcf["dL"] * FF_w * 2

        # dCD0/dV_cruise = dcf/dV * FF_w * 2
        partials[out, "mission:sizing:main_route:cruise:speed:%s" % propulsion_id] = (
            dcf["dV"] * FF_w * 2
        )

        # Atmospheric partials (stdatm derivatives)
        # atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        dnu_dh = atm.partial_kinematic_viscosity_altitude

        # dISA uses fixed-pressure convention in stdatm: dnu/d(dISA) ≈ -nu/T
        S_sutherland = 110.4  # Sutherland's constant for air [K]
        T = atm.temperature
        dnu_ddISA = (nu_air / T) * (2.5 - T / (T + S_sutherland))

        partials[out, "mission:sizing:main_route:cruise:altitude"] = (
            (dcf["dnu"] * dnu_dh) * FF_w * 2
        )

        partials[out, "mission:sizing:dISA"] = (dcf["dnu"] * dnu_ddISA) * FF_w * 2


class TailParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of the tails at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])
        self.options.declare("tail", default=None, values=["horizontal", "vertical"])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        tail = self.options["tail"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s"
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:tail:%s:MAC:length" % tail, val=np.nan, units="m")
        self.add_input("data:geometry:tail:%s:surface" % tail, val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:tail:%s" % tail, units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # Geometry
        propulsion_id = self.options["propulsion_id"]
        tail = self.options["tail"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        c_MAc_t = inputs["data:geometry:tail:%s:MAC:length" % tail]
        S_t = inputs["data:geometry:tail:%s:surface" % tail]
        S_ref = inputs["data:geometry:wing:surface"]

        # Flight parameters
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        atm.true_airspeed = V_cruise
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Friction coefficients assuming cruise conditions
        cf_tail = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAc_t, nu_air, a_air)

        # Form drag factors TODO: add sweep @ 0.3
        FF_tail = (1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio**4) * (
            1.34 * (V_cruise / a_air) ** 0.18
        )

        # Wetted area
        S_wet_t = 2 * S_t

        # Parasitic drag coefficient
        CD_0_tail = (cf_tail * FF_tail * S_wet_t) / S_ref

        outputs["data:aerodynamics:CD0:tail:%s" % tail] = CD_0_tail

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        tail = self.options["tail"]
        tc_ratio = inputs["data:geometry:wing:tc"]
        c_MAC_t = inputs["data:geometry:tail:%s:MAC:length" % tail]
        S_t = inputs["data:geometry:tail:%s:surface" % tail]
        S_ref = inputs["data:geometry:wing:surface"]
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]

        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Compute cf and its derivatives
        cf_tail = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_t, nu_air, a_air)
        dcf = AirframeAerodynamicsModel.friction_flatplate_derivatives(
            V_cruise, c_MAC_t, nu_air, a_air
        )

        # Form factor components
        F_form = 1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio**4
        dF_form_dtc = 2.0 + 400 * tc_ratio**3

        Mach = V_cruise / a_air
        F_comp = 1.34 * (Mach**0.18)
        FF_tail = F_form * F_comp

        # Derivatives of form factor wrt tc
        dFF_dtc = dF_form_dtc * F_comp

        # Derivatives of compressibility factor wrt V and a
        # F_comp = 1.34 * (V/a)^0.18, so dF_comp/dV = 0.18 * F_comp / V
        dF_comp_dV = 0.18 * F_comp / V_cruise if V_cruise > 0 else 0.0
        # dF_comp/da = -0.18 * F_comp / a
        dF_comp_da = -0.18 * F_comp / a_air

        out = "data:aerodynamics:CD0:tail:%s" % tail

        # dCD0/dtc_ratio = cf * dFF/dtc * 2*S_t/S_ref
        partials[out, "data:geometry:wing:tc"] = cf_tail * dFF_dtc * 2 * S_t / S_ref

        # dCD0/dS_t = cf * FF * 2 / S_ref
        partials[out, "data:geometry:tail:%s:surface" % tail] = cf_tail * FF_tail * 2 / S_ref

        # dCD0/dS_ref = -cf * FF * 2*S_t / S_ref^2
        partials[out, "data:geometry:wing:surface"] = -cf_tail * FF_tail * 2 * S_t / (S_ref**2)

        # dCD0/dc_MAC = dcf/dc_MAC * FF * 2*S_t/S_ref
        partials[out, "data:geometry:tail:%s:MAC:length" % tail] = (
            dcf["dL"] * FF_tail * 2 * S_t / S_ref
        )

        # dCD0/dV = (dcf/dV * FF + cf * dFF/dV) * 2*S_t/S_ref
        # where dFF/dV = F_form * dF_comp/dV
        dFF_dV = F_form * dF_comp_dV
        partials[out, "mission:sizing:main_route:cruise:speed:%s" % propulsion_id] = (
            (dcf["dV"] * FF_tail + cf_tail * dFF_dV) * 2 * S_t / S_ref
        )

        # Atmospheric partials
        # datm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        dnu_dh = atm.partial_kinematic_viscosity_altitude
        da_dh = atm.partial_speed_of_sound_altitude

        # dFF/dh = F_form * dF_comp/da * da/dh
        dFF_dh = F_form * dF_comp_da * da_dh

        # dCD0/dh = (dcf/dnu*dnu/dh + cf*dFF/dh) * FF * 2*S_t/S_ref
        partials[out, "mission:sizing:main_route:cruise:altitude"] = (
            (dcf["dnu"] * dnu_dh * FF_tail + cf_tail * dFF_dh) * 2 * S_t / S_ref
        )

        # dISA partials (Sutherland correction for kinematic viscosity)
        S_sutherland = 110.4  # Sutherland's constant for air [K]
        T = atm.temperature
        dnu_ddISA = (nu_air / T) * (2.5 - T / (T + S_sutherland))
        da_ddISA = 0.5 * a_air / T

        # dFF/d(dISA) = F_form * dF_comp/da * da/d(dISA)
        dFF_ddISA = F_form * dF_comp_da * da_ddISA

        # dCD0/d(dISA) = (dcf/dnu*dnu/d(dISA) + cf*dFF/d(dISA)) * FF * 2*S_t/S_ref
        partials[out, "mission:sizing:dISA"] = (
            (dcf["dnu"] * dnu_ddISA * FF_tail + cf_tail * dFF_ddISA) * 2 * S_t / S_ref
        )


class FuselageParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of the fuselage at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s"
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:fineness", val=5.0, units=None)
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:fuselage", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # Geometry
        propulsion_id = self.options["propulsion_id"]
        l_fus = inputs["data:geometry:fuselage:length"]
        d_fus_mid = inputs["data:geometry:fuselage:diameter:mid"]
        lmbda_f = inputs["data:geometry:fuselage:fineness"]
        S_ref = inputs["data:geometry:wing:surface"]

        # Flight parameters
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        atm.true_airspeed = V_cruise
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Friction coefficients assuming cruise conditions
        cf_fus = AirframeAerodynamicsModel.friction_flatplate(V_cruise, l_fus, nu_air, a_air)

        # Form drag factors
        FF_fus = 1 + 60 / lmbda_f**3 + lmbda_f / 400

        # Wetted areas
        S_wet_fus = (
            np.pi * d_fus_mid * l_fus * (1 - 2 / lmbda_f) ** (2 / 3) * (1 + 1 / (lmbda_f**2))
        )

        # Parasitic drag coefficient
        CD_0_fus = (cf_fus * S_wet_fus * FF_fus) / S_ref

        outputs["data:aerodynamics:CD0:fuselage"] = CD_0_fus

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        l_fus = inputs["data:geometry:fuselage:length"]
        d_fus_mid = inputs["data:geometry:fuselage:diameter:mid"]
        lmbda_f = inputs["data:geometry:fuselage:fineness"]
        S_ref = inputs["data:geometry:wing:surface"]
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]

        atm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Compute cf and its derivatives
        cf_fus = AirframeAerodynamicsModel.friction_flatplate(V_cruise, l_fus, nu_air, a_air)
        dcf = AirframeAerodynamicsModel.friction_flatplate_derivatives(
            V_cruise, l_fus, nu_air, a_air
        )

        # Form factor
        FF_fus = 1 + 60 / (lmbda_f**3) + lmbda_f / 400
        dFF_dlambda = -180 / (lmbda_f**4) + 1 / 400

        # Wetted area components
        f1 = (1 - 2 / lmbda_f) ** (2 / 3)
        f2 = 1 + 1 / (lmbda_f**2)
        S_wet_fus = np.pi * d_fus_mid * l_fus * f1 * f2

        # Derivatives of wetted area wrt fineness
        df1_dlambda = (2 / 3) * (1 - 2 / lmbda_f) ** (-1 / 3) * (2 / (lmbda_f**2))
        df2_dlambda = -2 / (lmbda_f**3)
        dS_wet_dlambda = np.pi * d_fus_mid * l_fus * (df1_dlambda * f2 + f1 * df2_dlambda)

        # Derivatives of wetted area wrt geometry
        dS_wet_dl_fus = np.pi * d_fus_mid * f1 * f2
        dS_wet_dd_mid = np.pi * l_fus * f1 * f2

        out = "data:aerodynamics:CD0:fuselage"

        # dCD0/dl_fus = (dcf/dl_fus * S_wet * FF + cf * dS_wet/dl_fus * FF) / S_ref
        partials[out, "data:geometry:fuselage:length"] = (
            dcf["dL"] * S_wet_fus * FF_fus + cf_fus * dS_wet_dl_fus * FF_fus
        ) / S_ref

        # dCD0/dd_fus_mid = cf * dS_wet/dd_mid * FF / S_ref
        partials[out, "data:geometry:fuselage:diameter:mid"] = (
            cf_fus * dS_wet_dd_mid * FF_fus
        ) / S_ref

        # dCD0/dlambda_f = (cf * dS_wet/dlambda * FF + cf * S_wet * dFF/dlambda) / S_ref
        partials[out, "data:geometry:fuselage:fineness"] = (
            cf_fus * dS_wet_dlambda * FF_fus + cf_fus * S_wet_fus * dFF_dlambda
        ) / S_ref

        # dCD0/dS_ref = -(cf * S_wet * FF) / S_ref^2
        partials[out, "data:geometry:wing:surface"] = -(cf_fus * S_wet_fus * FF_fus) / (S_ref**2)

        # dCD0/dV_cruise = dcf/dV * S_wet * FF / S_ref
        partials[out, "mission:sizing:main_route:cruise:speed:%s" % propulsion_id] = (
            dcf["dV"] * S_wet_fus * FF_fus
        ) / S_ref

        # Atmospheric partials
        # datm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        dnu_dh = atm.partial_kinematic_viscosity_altitude

        # dCD0/daltitude = dcf/dnu * dnu/dh * S_wet * FF / S_ref
        partials[out, "mission:sizing:main_route:cruise:altitude"] = (
            dcf["dnu"] * dnu_dh * S_wet_fus * FF_fus
        ) / S_ref

        # dISA partials (Sutherland correction)
        S_sutherland = 110.4
        T = atm.temperature
        dnu_ddISA = (nu_air / T) * (2.5 - T / (T + S_sutherland))

        # dCD0/d(dISA) = dcf/dnu * dnu/d(dISA) * S_wet * FF / S_ref
        partials[out, "mission:sizing:dISA"] = (dcf["dnu"] * dnu_ddISA * S_wet_fus * FF_fus) / S_ref


class ParasiticDragConstraint(om.ExplicitComponent):
    """
    Parasitic drag consistency constraint, to ensure that the initial guess equals the estimated value.
    """

    def setup(self):
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_output("optimization:constraints:aerodynamics:CD0:consistency", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:aerodynamics:CD0"]

        CD0_cnstr = (CD_0_guess - CD_0) / CD_0

        outputs["optimization:constraints:aerodynamics:CD0:consistency"] = CD0_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:aerodynamics:CD0"]

        partials[
            "optimization:constraints:aerodynamics:CD0:consistency",
            "optimization:variables:aerodynamics:CD0:guess",
        ] = 1 / CD_0
        partials[
            "optimization:constraints:aerodynamics:CD0:consistency", "data:aerodynamics:CD0"
        ] = -CD_0_guess / CD_0**2


class MaxLiftToDrag(om.ExplicitComponent):
    """
    Computes maximum lift to drag ratio and speed for best L/D.
    """

    def setup(self):
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=0.035, units=None)
        self.add_output("data:aerodynamics:LD:max", units=None, lower=0.0)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        CD_0 = inputs["data:aerodynamics:CD0"]
        K = inputs["data:aerodynamics:CDi:K"]

        LDmax = 0.5 / np.sqrt(CD_0 * K)  # [-] max lift-to-drag ratio
        # V_opt = np.sqrt(2 * m_uav * g / rho_air / S_w * np.sqrt(K / CD_0))

        outputs["data:aerodynamics:LD:max"] = LDmax

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        CD_0 = inputs["data:aerodynamics:CD0"]
        K = inputs["data:aerodynamics:CDi:K"]

        partials["data:aerodynamics:LD:max", "data:aerodynamics:CD0"] = (
            -0.25 * K ** (-0.5) * CD_0 ** (-1.5)
        )
        partials["data:aerodynamics:LD:max", "data:aerodynamics:CDi:K"] = (
            -0.25 * CD_0 ** (-0.5) * K ** (-1.5)
        )


class SpanEfficiency(om.ExplicitComponent):
    """
    Computes the span efficiency (used as a preliminary calculation for sizing scenarios)
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:e", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]

        e = (
            1.78 * (1 - 0.045 * AR_w**0.68) - 0.64
        )  # span efficiency factor (empirical estimation for straight wings, Raymer)

        outputs["data:aerodynamics:CDi:e"] = e

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]

        partials["data:aerodynamics:CDi:e", "optimization:variables:geometry:wing:AR"] = (
            -1.78 * 0.045 * 0.68 * AR_w ** (0.68 - 1)
        )


class InducedDragConstant(om.ExplicitComponent):
    """
    Computes the induced drag constant (used as a preliminary calculation for sizing scenarios)
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:K", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        K = 1 / (
            np.pi * e * AR_w
        )  # induced drag constant (correction term for non-elliptical lift distribution)

        outputs["data:aerodynamics:CDi:K"] = K

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        partials["data:aerodynamics:CDi:K", "optimization:variables:geometry:wing:AR"] = -1 / (
            np.pi * e * AR_w**2
        )
        partials["data:aerodynamics:CDi:K", "data:aerodynamics:CDi:e"] = -1 / (np.pi * e**2 * AR_w)
