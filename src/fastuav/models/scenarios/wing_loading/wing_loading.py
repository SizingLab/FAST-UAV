"""
Wing loading requirements.
"""

import numpy as np
import openmdao.api as om
import logging
from fastoad.openmdao.validity_checker import ValidityDomainChecker
from fastuav.constants import FW_PROPULSION
from stdatm import AtmosphereSI

_LOGGER = logging.getLogger(__name__)  # Logger for this module
WS_MIN = 100  # [N/m2] lower limit for the wing loading. Under this value, increasing CLmax is recommended.
T0 = 288.15  # Sea level temperature (K)
L = 0.0065   # Lapse rate (K/m)
from scipy.constants import g
R = 287.05287  # Specific gas constant for dry air (J/(kg·K))


class WingLoadingStall(om.ExplicitComponent):
    """
    Computes wing loading for stall speed (at cruise altitude) requirement.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:stall:speed:%s" % propulsion_id, val=np.nan, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:aerodynamics:CLmax", val=1.3, units=None)
        self.add_output("data:geometry:wing:loading:stall", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]

        # Flight parameters
        V_stall = inputs["mission:sizing:main_route:stall:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_stall
        q_stall = atm.dynamic_pressure

        # Wing lift parameters
        CL_max = inputs["data:aerodynamics:CLmax"]

        # Wing loading calculation
        WS_stall = q_stall * CL_max  # wing loading required to meet stall speed requirement [N/m2]

        outputs["data:geometry:wing:loading:stall"] = WS_stall

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        propulsion_id = self.options["propulsion_id"]
        V_stall_ID = "mission:sizing:main_route:stall:speed:%s" % propulsion_id
        V_stall = inputs[V_stall_ID]
    
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]

        CL_max_ID = "data:aerodynamics:CLmax"
        CL_max = inputs[CL_max_ID]

        
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_stall
        q_stall = atm.dynamic_pressure # q = 0.5 * rho * V**2
        rho = atm.density

        # ===== DENSITY DERIVATIVES (Analytic, Troposphere) =====
        # Temperature profile: T = T0 - L * h + dISA
        # Density varies as: rho ∝ (T/T0)^(g/RL - 1) --> (g/(R*L) - 1) ≈ -5.256 for ISA
        from stdatm import AtmosphereWithPartials
        datm = AtmosphereWithPartials(altitude_cruise, dISA, altitude_in_feet=False)
        # drho = datm.density
        # d(rho)/d(h):   chain rule through T
        # d(rho)/d(h) = d(rho)/d(T) * d(T)/d(h) = (g/(R*L) - 1) * rho / T * (-L)
        drho_dh = datm.partial_density_altitude # d(rho)/d(altitude)
        # d(rho)/d(dISA): stdatm treats dISA as a pure temperature offset at fixed pressure
        drho_ddISA = -rho / (T0 - L * altitude_cruise + dISA)
        

        out_ID = "data:geometry:wing:loading:stall"
        # ===== PARTIAL DERIVATIVES =====
        # ∂(WS)/∂(V_stall) = ∂(q_stall)/∂(V_stall) * CL_max = ρ * V * CL_max
        partials[out_ID, V_stall_ID] = rho * V_stall * CL_max
        # ∂(WS)/∂(CL_max) = q = 0.5 * ρ * V^2
        partials[out_ID, CL_max_ID] = q_stall
        # ∂(WS)/∂(altitude) = 0.5 * V^2 * CL_max * d(rho)/d(h)
        partials[out_ID, "mission:sizing:main_route:cruise:altitude"] = 0.5 * V_stall**2 * CL_max * drho_dh
        # ∂(WS)/∂(dISA) = 0.5 * V^2 * CL_max * d(rho)/d(dISA)
        partials[out_ID, "mission:sizing:dISA"] = 0.5 * V_stall**2 * CL_max * drho_ddISA


class WingLoadingCruise(om.ExplicitComponent):
    """
    Computes wing loading to maximize the achievable range (at cruise altitude and speed)
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:cruise:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("optimization:variables:aerodynamics:CDi:K:guess", val=0.035, units=None)
        self.add_output("data:geometry:wing:loading:cruise", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]

        # Flight parameters
        V_cruise = inputs["mission:sizing:main_route:cruise:speed:%s" % propulsion_id]
        altitude_cruise = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_cruise
        q_cruise = atm.dynamic_pressure

        # Induced drag parameter
        K = inputs["optimization:variables:aerodynamics:CDi:K:guess"]

        # Parasitic drag parameter
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Wing loading calculation
        WS_cruise = q_cruise * np.sqrt(
            CD_0_guess / K
        )  # wing loading that maximizes range during cruise [N/m2] (simplified drag model CD = CD_0 + K * CL^2)

        outputs["data:geometry:wing:loading:cruise"] = WS_cruise


@ValidityDomainChecker(
    {
        "data:geometry:wing:loading": (WS_MIN, None),  # defines a lower bound for wing loading
    }
)
class WingLoadingSelection(om.ExplicitComponent):
    """
    Wing loading selection.
    This selection is achieved with an under sizing coefficient on the estimated wing loading to meet the stall
    speed requirement.
    """

    def setup(self):
        self.add_input("data:geometry:wing:loading:stall", val=np.nan, units="N/m**2")
        self.add_input("optimization:variables:geometry:wing:loading:k", val=1.0, units=None)
        self.add_output("data:geometry:wing:loading", units="N/m**2")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        WS_stall = inputs["data:geometry:wing:loading:stall"]
        k_WS = inputs["optimization:variables:geometry:wing:loading:k"]

        WS = (
            k_WS * WS_stall
        )  # [N/m**2] wing loading selection from stall requirement with under-sizing coefficient

        outputs["data:geometry:wing:loading"] = WS

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        WS_stall = inputs["data:geometry:wing:loading:stall"]
        k_WS = inputs["optimization:variables:geometry:wing:loading:k"]

        partials["data:geometry:wing:loading", "data:geometry:wing:loading:stall"] = k_WS
        partials["data:geometry:wing:loading", "optimization:variables:geometry:wing:loading:k"] = WS_stall


# @ValidityDomainChecker(
#     {
#         "data:geometry:wing:loading": (WS_MIN, None),  # defines a lower bound for wing loading
#     }
# )
# class WingLoadingSelection(om.ExplicitComponent):
#     """
#     Wing loading selection.
#     This selection is achieved with a sizing coefficient to vary the wing loading around the estimated value for
#     best cruise performance. A constraint is added to ensure sufficient performance at low speeds (stall speed
#     condition).
#     """
#
#     def setup(self):
#         self.add_input("data:geometry:wing:loading:cruise", val=np.nan, units="N/m**2")
#         self.add_input("data:geometry:wing:loading:stall", val=np.nan, units="N/m**2")
#         self.add_input("optimization:variables:geometry:wing:loading:k", val=1.0, units=None)
#         self.add_output("data:geometry:wing:loading", units="N/m**2")
#         self.add_output("optimization:constraints:geometry:wing:loading:stall", units=None)
#
#     def setup_partials(self):
#         self.declare_partials("*", "*", method="exact")
#
#     def compute(self, inputs, outputs):
#         WS_cruise = inputs["data:geometry:wing:loading:cruise"]
#         WS_stall = inputs["data:geometry:wing:loading:stall"]
#         k_WS = inputs["optimization:variables:geometry:wing:loading:k"]
#
#         WS = (
#             k_WS * WS_cruise
#         )  # [N/m**2] wing loading selection from cruise requirement with under-sizing coefficient
#         WS_stall_cnstr = (
#             WS_stall - WS
#         ) / WS  # constraint on stall WS (selected WS should be lower than stall WS)
#
#         outputs["data:geometry:wing:loading"] = WS
#         outputs["optimization:constraints:geometry:wing:loading:stall"] = WS_stall_cnstr
#
#     def compute_partials(self, inputs, partials, discrete_inputs=None):
#         WS_cruise = inputs["data:geometry:wing:loading:cruise"]
#         WS_stall = inputs["data:geometry:wing:loading:stall"]
#         k_WS = inputs["optimization:variables:geometry:wing:loading:k"]
#         WS = k_WS * WS_cruise
#
#         partials["optimization:variables:geometry:wing:loading", "optimization:variables:geometry:wing:loading:k"] = WS_cruise
#         partials["data:geometry:wing:loading", "data:geometry:wing:loading:cruise"] = k_WS
#         partials["optimization:constraints:geometry:wing:loading:stall", "data:geometry:wing:loading:stall"] = 1.0 / WS
#         partials[
#             "optimization:constraints:geometry:wing:loading:stall", "data:geometry:wing:loading:cruise"
#         ] = - WS_stall / (k_WS * WS_cruise ** 2)
#         partials[
#             "optimization:variables:geometry:wing:loading:stall:constraint", "optimization:variables:geometry:wing:loading:k"
#         ] = - WS_stall / (k_WS ** 2 * WS_cruise)
