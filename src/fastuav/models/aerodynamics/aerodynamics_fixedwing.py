"""
Fixed Wing Aerodynamics (external)
"""
import fastoad.api as oad
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import (
    add_subsystem_with_deviation,
)
from fastuav.constants import FW_PROPULSION
from stdatm import AtmosphereSI


class AirframeAerodynamicsModel:
    """
    Aerodynamics model for the airframe
    """

    @staticmethod
    def friction_flatplate(V_air, L, nu_air, a_air):
        re = V_air * L / nu_air  # Reynolds number [-]
        # mach = V_air / a_air  # mach number [-]
        cf_turb = 0.455 / (
            np.log10(re) ** 2.58 #* (1 + 0.144 * mach**2) ** (-0.65)
        )  # flat-plate skin friction [-]
        return cf_turb


@oad.RegisterOpenMDAOSystem("fastuav.aerodynamics.fixedwing")
class Aerodynamics(om.Group):
    """
    Group containing the external aerodynamics calculations
    """

    def setup(self):

        # Parasitic drag calculations
        parasitic_drag = self.add_subsystem("parasitic_drag", om.Group(), promotes=["*"])
        parasitic_drag.add_subsystem("wing", WingParasiticDrag(), promotes=["*"])
        parasitic_drag.add_subsystem("horizontal_tail", TailParasiticDrag(tail="horizontal"), promotes=["*"])
        parasitic_drag.add_subsystem("vertical_tail", TailParasiticDrag(tail="vertical"), promotes=["*"])
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
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        outputs["data:aerodynamics:CD0"] = inputs["data:aerodynamics:CD0:wing"] \
                                           + inputs["data:aerodynamics:CD0:tail:horizontal"] \
                                           + inputs["data:aerodynamics:CD0:tail:vertical"] \
                                           + inputs["data:aerodynamics:CD0:fuselage"]


class WingParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of the wing at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:wing", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

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
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_cruise
        a_air = atm.speed_of_sound
        nu_air = atm.kinematic_viscosity

        # Friction coefficients assuming cruise conditions
        cf_wing = AirframeAerodynamicsModel.friction_flatplate(V_cruise, c_MAC_w, nu_air, a_air)

        # Form drag factor
        FF_w = (1 + 0.6 / 0.3 * tc_ratio + 100 * tc_ratio**4)
        # * (1.34 * (V_cruise / a_air) ** 0.18)  # compressibility effects

        # Wetted area
        S_wet_w = 2 * S_w

        # Parasitic drag coefficient
        CD_0_wing = (cf_wing * FF_w * S_wet_w) / S_ref

        outputs["data:aerodynamics:CD0:wing"] = CD_0_wing


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
        self.add_input("mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:wing:tc", val=0.15, units=None)
        self.add_input("data:geometry:tail:%s:MAC:length" % tail, val=np.nan, units="m")
        self.add_input("data:geometry:tail:%s:surface" % tail, val=np.nan, units="m**2")
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:tail:%s" % tail, units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

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
        atm = AtmosphereSI(altitude_cruise, dISA)
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


class FuselageParasiticDrag(om.ExplicitComponent):
    """
    Computes parasitic drag of the fuselage at cruise conditions
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:geometry:fuselage:length", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:diameter:mid", val=np.nan, units="m")
        self.add_input("data:geometry:fuselage:fineness", val=5.0, units=None)
        self.add_input("data:geometry:wing:surface", val=np.nan, units="m**2")
        self.add_output("data:aerodynamics:CD0:fuselage", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

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
        atm = AtmosphereSI(altitude_cruise, dISA)
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


class ParasiticDragConstraint(om.ExplicitComponent):
    """
    Parasitic drag consistency constraint, to ensure that the initial guess equals the estimated value.
    """

    def setup(self):
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CD0:guess:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:aerodynamics:CD0"]

        CD0_cnstr = (CD_0_guess - CD_0) / CD_0

        outputs["data:aerodynamics:CD0:guess:constraint"] = CD0_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        CD_0 = inputs["data:aerodynamics:CD0"]

        partials["data:aerodynamics:CD0:guess:constraint", "data:aerodynamics:CD0:guess"] = (
            1 / CD_0
        )
        partials["data:aerodynamics:CD0:guess:constraint", "data:aerodynamics:CD0"] = (
            - CD_0_guess / CD_0 ** 2
        )


class MaxLiftToDrag(om.ExplicitComponent):
    """
    Computes maximum lift to drag ratio and speed for best L/D.
    """

    def setup(self):
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:aerodynamics:LD:max", units=None, lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        CD_0 = inputs["data:aerodynamics:CD0"]
        K = inputs["data:aerodynamics:CDi:K"]

        LDmax = 0.5 / np.sqrt(CD_0 * K)  # [-] max lift-to-drag ratio
        # V_opt = np.sqrt(2 * m_uav * g / rho_air / S_w * np.sqrt(K / CD_0))

        outputs["data:aerodynamics:LD:max"] = LDmax


class SpanEfficiency(om.ExplicitComponent):
    """
    Computes the span efficiency (used as a preliminary calculation for sizing scenarios)
    """

    def setup(self):
        self.add_input("data:geometry:wing:AR", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:e", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:geometry:wing:AR"]

        e = (
            1.78 * (1 - 0.045 * AR_w**0.68) - 0.64
        )  # span efficiency factor (empirical estimation for straight wings, Raymer)

        outputs["data:aerodynamics:CDi:e"] = e

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:geometry:wing:AR"]

        partials["data:aerodynamics:CDi:e", "data:geometry:wing:AR"] = (
            -1.78 * 0.045 * 0.68 * AR_w ** (0.68 - 1)
        )


class InducedDragConstant(om.ExplicitComponent):
    """
    Computes the induced drag constant (used as a preliminary calculation for sizing scenarios)
    """

    def setup(self):
        self.add_input("data:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_output("data:aerodynamics:CDi:K", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        K = 1 / (
            np.pi * e * AR_w
        )  # induced drag constant (correction term for non-elliptical lift distribution)

        outputs["data:aerodynamics:CDi:K"] = K

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        AR_w = inputs["data:geometry:wing:AR"]
        e = inputs["data:aerodynamics:CDi:e"]

        partials["data:aerodynamics:CDi:K", "data:geometry:wing:AR"] = -1 / (np.pi * e * AR_w**2)
        partials["data:aerodynamics:CDi:K", "data:aerodynamics:CDi:e"] = -1 / (
            np.pi * e**2 * AR_w
        )