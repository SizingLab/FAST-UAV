"""
Wing loading requirements.
"""

import numpy as np
import openmdao.api as om
import logging
from fastoad.openmdao.validity_checker import ValidityDomainChecker

_LOGGER = logging.getLogger(__name__)  # Logger for this module
WS_MIN = 100  # [N/m2] lower limit for the wing loading. Under this value, increasing CLmax is recommended.


class WingLoadingStall(om.ExplicitComponent):
    """
    Computes wing loading for stall speed requirement
    """

    def setup(self):
        self.add_input("mission:design_mission:stall:q", val=np.nan, units="Pa")
        self.add_input("data:aerodynamics:CLmax", val=1.3, units=None)
        self.add_output("data:loads:wing_loading:stall", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        q_stall = inputs["mission:design_mission:stall:q"]
        CL_max = inputs["data:aerodynamics:CLmax"]

        WS_stall = q_stall * CL_max  # wing loading required to meet stall speed requirement [N/m2]

        # if WS_stall < WS_MIN:
        #    _LOGGER.warning(
        #        "Very low wing loading for stall speed requirement. Consider increasing CLmax or relaxing the stall speed requirement."
        #    )

        outputs["data:loads:wing_loading:stall"] = WS_stall

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        q_stall = inputs["mission:design_mission:stall:q"]
        CL_max = inputs["data:aerodynamics:CLmax"]

        partials["data:loads:wing_loading:stall", "mission:design_mission:stall:q"] = CL_max

        partials["data:loads:wing_loading:stall", "data:aerodynamics:CLmax"] = q_stall


class WingLoadingCruise(om.ExplicitComponent):
    """
    Computes wing loading to maximize the achievable range.
    """

    def setup(self):
        self.add_input("mission:design_mission:cruise:q", val=np.nan, units="Pa")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:loads:wing_loading:cruise", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        q_cruise = inputs["mission:design_mission:cruise:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]

        WS_cruise = q_cruise * np.sqrt(
            CD_0_guess / K
        )  # wing loading that maximizes range during cruise [N/m2] (simplified drag model CD = CD_0 + K * CL^2)

        # if WS_cruise < WS_MIN:
        #    _LOGGER.warning(
        #        "Very low wing loading for optimal range. Consider increasing the cruise speed requirement."
        #    )

        outputs["data:loads:wing_loading:cruise"] = WS_cruise

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        q_cruise = inputs["mission:design_mission:cruise:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]

        partials["data:loads:wing_loading:cruise", "mission:design_mission:cruise:q"] = np.sqrt(
            CD_0_guess / K
        )
        partials["data:loads:wing_loading:cruise", "data:aerodynamics:CD0:guess"] = (
            0.5 * q_cruise / np.sqrt(K) / np.sqrt(CD_0_guess)
        )
        partials["data:loads:wing_loading:cruise", "data:aerodynamics:CDi:K"] = (
            -0.5 * q_cruise * np.sqrt(CD_0_guess) * (1 / K) ** (3 / 2)
        )


@ValidityDomainChecker(
    {
        "data:loads:wing_loading": (WS_MIN, None),  # Defines only a lower bound
    }
)
class WingLoadingSelection(om.ExplicitComponent):
    """
    Wing loading selection.
    The lowest wing loading is selected for sizing the wing.
    This ensures that the wing is large enough for all flight conditions,
    i.e. it provides enough lift in all circumstances.
    This selection achieved with an undersizing coefficient and constraints on the wing loading.
    """

    def setup(self):
        self.add_input("data:loads:wing_loading:cruise", val=np.nan, units="N/m**2")
        self.add_input("data:loads:wing_loading:stall", val=np.nan, units="N/m**2")
        self.add_input("data:loads:wing_loading:k", val=1.0, units=None)
        self.add_output("data:loads:wing_loading", units="N/m**2")
        self.add_output("data:loads:wing_load:stall:constraint", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        WS_cruise = inputs["data:loads:wing_loading:cruise"]
        WS_stall = inputs["data:loads:wing_loading:stall"]
        k_WS = inputs["data:loads:wing_loading:k"]

        WS = (
            k_WS * WS_cruise
        )  # [N/m**2] wing loading selection from cruise requirement with undersizing variable
        WS_stall_cnstr = (
            WS_stall - WS
        ) / WS  # constraint on stall WS (selected WS should be lower than stall WS)

        outputs["data:loads:wing_loading"] = WS
        outputs["data:loads:wing_load:stall:constraint"] = WS_stall_cnstr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        WS_cruise = inputs["data:loads:wing_loading:cruise"]
        WS_stall = inputs["data:loads:wing_loading:stall"]
        k_WS = inputs["data:loads:wing_loading:k"]
        WS = k_WS * WS_cruise

        partials["data:loads:wing_loading", "data:loads:wing_loading:k"] = WS_cruise
        partials["data:loads:wing_loading", "data:loads:wing_loading:cruise"] = k_WS
        partials["data:loads:wing_load:stall:constraint", "data:loads:wing_loading:stall"] = 1 / WS
        partials[
            "data:loads:wing_load:stall:constraint", "data:loads:wing_loading:cruise"
        ] = -WS_stall / (k_WS * WS_cruise**2)
        partials[
            "data:loads:wing_load:stall:constraint", "data:loads:wing_loading:k"
        ] = -WS_stall / (k_WS**2 * WS_cruise)
