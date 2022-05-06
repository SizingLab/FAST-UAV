"""
Wing loading requirements.
"""

import numpy as np
import openmdao.api as om
import logging
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from fastuav.utils.constants import FW_PROPULSION
from stdatm import AtmosphereSI

_LOGGER = logging.getLogger(__name__)  # Logger for this module
WS_MIN = 100  # [N/m2] lower limit for the wing loading. Under this value, increasing CLmax is recommended.


class WingLoadingStall(om.ExplicitComponent):
    """
    Computes wing loading for stall speed (at cruise altitude) requirement.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=0.0, units="m")
        self.add_input("data:scenarios:%s:stall:speed" % propulsion_id, val=0.0, units="m/s")
        self.add_input("data:scenarios:dISA", val=0.0, units="K")
        self.add_input("data:aerodynamics:CLmax", val=1.3, units=None)
        self.add_output("data:scenarios:wing_loading:stall", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]

        # Flight parameters
        V_stall = inputs["data:scenarios:%s:stall:speed" % propulsion_id]
        altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
        dISA = inputs["data:scenarios:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_stall
        q_stall = atm.dynamic_pressure

        # Wing lift parameters
        CL_max = inputs["data:aerodynamics:CLmax"]

        # Wing loading calculation
        WS_stall = q_stall * CL_max  # wing loading required to meet stall speed requirement [N/m2]

        outputs["data:scenarios:wing_loading:stall"] = WS_stall

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     propulsion_id = self.options["propulsion_id"]
    #     V_stall = inputs["data:scenarios:%s:stall:speed" % propulsion_id]
    #     altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
    #     dISA = inputs["data:scenarios:dISA"]
    #     atm = AtmosphereSI(altitude_cruise, dISA)
    #     atm.true_airspeed = V_stall
    #     q_stall = atm.dynamic_pressure
    #
    #     partials["data:scenarios:wing_loading:stall", "data:aerodynamics:CLmax"] = q_stall


class WingLoadingCruise(om.ExplicitComponent):
    """
    Computes wing loading to maximize the achievable range (at cruise altitude and speed)
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=0.0, units="m")
        self.add_input("data:scenarios:%s:cruise:speed" % propulsion_id, val=0.0, units="m/s")
        self.add_input("data:scenarios:dISA", val=0.0, units="K")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:scenarios:wing_loading:cruise", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]

        # Flight parameters
        V_cruise = inputs["data:scenarios:%s:cruise:speed" % propulsion_id]
        altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
        dISA = inputs["data:scenarios:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_cruise
        q_cruise = atm.dynamic_pressure

        # Induced drag parameter
        K = inputs["data:aerodynamics:CDi:K"]

        # Parasitic drag parameter
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]

        # Wing loading calculation
        WS_cruise = q_cruise * np.sqrt(
            CD_0_guess / K
        )  # wing loading that maximizes range during cruise [N/m2] (simplified drag model CD = CD_0 + K * CL^2)

        outputs["data:scenarios:wing_loading:cruise"] = WS_cruise

    # def compute_partials(self, inputs, partials, discrete_inputs=None):
    #     propulsion_id = self.options["propulsion_id"]
    #     V_cruise = inputs["data:scenarios:%s:cruise:speed" % propulsion_id]
    #     altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
    #     dISA = inputs["data:scenarios:dISA"]
    #     atm = AtmosphereSI(altitude_cruise, dISA)
    #     atm.true_airspeed = V_cruise
    #     q_cruise = atm.dynamic_pressure
    #     K = inputs["data:aerodynamics:CDi:K"]
    #     CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
    #
    #     partials["data:scenarios:wing_loading:cruise", "data:aerodynamics:CD0:guess"] = (
    #         0.5 * q_cruise / np.sqrt(K) / np.sqrt(CD_0_guess)
    #     )
    #     partials["data:scenarios:wing_loading:cruise", "data:aerodynamics:CDi:K"] = (
    #         -0.5 * q_cruise * np.sqrt(CD_0_guess) * (1.0 / K) ** (3 / 2)
    #     )


@ValidityDomainChecker(
    {
        "data:scenarios:wing_loading": (WS_MIN, None),  # defines a lower bound for wing loading
    }
)
class WingLoadingSelection(om.ExplicitComponent):
    """
    Wing loading selection.
    The lowest wing loading is selected for sizing the wing.
    This ensures that the wing is large enough for all flight conditions,
    i.e. it provides enough lift in all circumstances.
    This selection is achieved with an under-sizing coefficient and constraints on the wing loading.
    """

    def setup(self):
        self.add_input("data:scenarios:wing_loading:cruise", val=np.nan, units="N/m**2")
        self.add_input("data:scenarios:wing_loading:stall", val=np.nan, units="N/m**2")
        self.add_input("data:scenarios:wing_loading:k", val=1.0, units=None)
        self.add_output("data:scenarios:wing_loading", units="N/m**2")
        self.add_output("data:scenarios:wing_loading:stall:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        WS_cruise = inputs["data:scenarios:wing_loading:cruise"]
        WS_stall = inputs["data:scenarios:wing_loading:stall"]
        k_WS = inputs["data:scenarios:wing_loading:k"]

        WS = (
            k_WS * WS_cruise
        )  # [N/m**2] wing loading selection from cruise requirement with under-sizing coefficient
        WS_stall_cnstr = (
            WS_stall - WS
        ) / WS  # constraint on stall WS (selected WS should be lower than stall WS)

        outputs["data:scenarios:wing_loading"] = WS
        outputs["data:scenarios:wing_loading:stall:constraint"] = WS_stall_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        WS_cruise = inputs["data:scenarios:wing_loading:cruise"]
        WS_stall = inputs["data:scenarios:wing_loading:stall"]
        k_WS = inputs["data:scenarios:wing_loading:k"]
        WS = k_WS * WS_cruise

        partials["data:scenarios:wing_loading", "data:scenarios:wing_loading:k"] = WS_cruise
        partials["data:scenarios:wing_loading", "data:scenarios:wing_loading:cruise"] = k_WS
        partials["data:scenarios:wing_loading:stall:constraint", "data:scenarios:wing_loading:stall"] = 1.0 / WS
        partials[
            "data:scenarios:wing_loading:stall:constraint", "data:scenarios:wing_loading:cruise"
        ] = - WS_stall / (k_WS * WS_cruise ** 2)
        partials[
            "data:scenarios:wing_loading:stall:constraint", "data:scenarios:wing_loading:k"
        ] = - WS_stall / (k_WS ** 2 * WS_cruise)