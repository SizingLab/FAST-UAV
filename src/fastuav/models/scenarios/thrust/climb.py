"""
Climb scenarios
"""

import numpy as np
import openmdao.api as om
from scipy.constants import g
from stdatm import AtmosphereSI, AtmosphereWithPartials

from fastuav.constants import FW_PROPULSION, MR_PROPULSION
from fastuav.models.scenarios.thrust.flight_models import MultirotorFlightModel

# ISA troposphere constants (consistent with the stdatm temperature model:
# T = T0 - L * altitude + dISA)
T0 = 288.15  # [K] sea-level standard temperature
L = 0.0065  # [K/m] troposphere temperature lapse rate


class MultirotorClimbThrust(om.ExplicitComponent):
    """
    Thrust for a climb in multirotor mode, at desired rate of climb and climb speed
    (i.e., the flight path is not necessarily vertical)
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input(
            "data:propulsion:%s:propeller:number" % propulsion_id,
            val=np.nan,
            units=None,
        )
        self.add_input("data:aerodynamics:%s:CD0" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
        self.add_input(
            "data:geometry:projected_area:front", val=0.0, units="m**2"
        )  # TODO: define front area for hybrid VTOL UAVs?
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:climb:speed:%s" % propulsion_id,
            val=0.0,
            units="m/s",
        )
        self.add_input(
            "mission:sizing:main_route:climb:rate:%s" % propulsion_id,
            val=np.nan,
            units="m/s",
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_output("data:propulsion:%s:propeller:thrust:climb" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:climb" % propulsion_id, units="rad")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        # Flight parameters
        V_v = inputs["mission:sizing:main_route:climb:rate:%s" % propulsion_id]
        V_climb = inputs["mission:sizing:main_route:climb:speed:%s" % propulsion_id]
        altitude_climb = inputs[
            "mission:sizing:main_route:cruise:altitude"
        ]  # conservative assumption
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereWithPartials(altitude_climb, dISA, altitude_in_feet=False)
        atm.true_airspeed = V_climb
        rho_air = atm.density

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]

        # Drag parameters
        C_D0 = inputs["data:aerodynamics:%s:CD0" % propulsion_id]
        C_L = 0.0  # it is assumed that the body shape produces only pressure drag and no lift
        S_top = inputs["data:geometry:projected_area:top"]
        S_front = inputs["data:geometry:projected_area:front"]

        alpha_cl = MultirotorFlightModel.get_angle_of_attack(
            m_uav_guess, V_climb, V_v, S_front, S_top, C_D0, C_L, rho_air
        )  # [rad] angle of attack
        F_pro_cl = (
            MultirotorFlightModel.get_thrust(
                m_uav_guess, V_climb, V_v, alpha_cl, S_front, S_top, C_D0, C_L, rho_air
            )
            / Npro
        )  # [N] thrust per propeller

        outputs["data:propulsion:%s:propeller:thrust:climb" % propulsion_id] = F_pro_cl
        outputs["data:propulsion:%s:propeller:AoA:climb" % propulsion_id] = alpha_cl


class FixedwingClimbThrust(om.ExplicitComponent):
    """
    Thrust for the desired rate of climb, in fixed wing configuration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:climb:speed:%s" % propulsion_id,
            val=0.0,
            units="m/s",
        )
        self.add_input(
            "mission:sizing:main_route:climb:rate:%s" % propulsion_id,
            val=np.nan,
            units="m/s",
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_output("data:propulsion:%s:propeller:thrust:climb" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:climb" % propulsion_id, units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]

        # Flight parameters
        V_v = inputs["mission:sizing:main_route:climb:rate:%s" % propulsion_id]
        V_climb = inputs["mission:sizing:main_route:climb:speed:%s" % propulsion_id]
        altitude_climb = inputs[
            "mission:sizing:main_route:cruise:altitude"
        ]  # conservative assumption
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereWithPartials(altitude_climb, dISA, altitude_in_feet=False)
        atm.true_airspeed = V_climb
        q_climb = atm.dynamic_pressure

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Weight = m_uav_guess * g

        # Induced drag parameters
        K = inputs["data:aerodynamics:CDi:K"]

        # Parasitic drag parameters
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Thrust and trim calculation (equilibrium)
        TW_climb = (
            V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS
        )  # thrust-to-weight ratio in climb conditions [-]
        F_pro_climb = TW_climb * Weight / Npro  # [N] Thrust per propeller for climb
        alpha_cl = (
            np.pi / 2
        )  # [rad] Rotor disk Angle of Attack (assumption: axial flight TODO: estimate trim?)

        outputs["data:propulsion:%s:propeller:thrust:climb" % propulsion_id] = F_pro_climb
        outputs["data:propulsion:%s:propeller:AoA:climb" % propulsion_id] = alpha_cl


class FixedwingClimbThrust_VLM(om.ExplicitComponent):
    """
    Thrust for the desired rate of climb, in fixed wing configuration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("optimization:variables:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("optimization:variables:aerodynamics:CD0:guess", val=0.04, units=None)
        # self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_input("optimization:variables:aerodynamics:CDi:K:guess", val=0.035, units=None)
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input(
            "mission:sizing:main_route:climb:speed:%s" % propulsion_id, val=0.0, units="m/s"
        )
        self.add_input(
            "mission:sizing:main_route:climb:rate:%s" % propulsion_id, val=np.nan, units="m/s"
        )
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_output("data:propulsion:%s:propeller:thrust:climb" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:climb" % propulsion_id, units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]

        # Flight parameters
        V_v = inputs["mission:sizing:main_route:climb:rate:%s" % propulsion_id]
        V_climb = inputs["mission:sizing:main_route:climb:speed:%s" % propulsion_id]
        altitude_climb = inputs[
            "mission:sizing:main_route:cruise:altitude"
        ]  # conservative assumption
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_climb, dISA)
        atm.true_airspeed = V_climb
        q_climb = atm.dynamic_pressure

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Weight = m_uav_guess * g

        # Induced drag parameters
        # K = inputs["data:aerodynamics:CDi:K"]
        K = inputs["optimization:variables:aerodynamics:CDi:K:guess"]

        # Parasitic drag parameters
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Thrust and trim calculation (equilibrium)
        TW_climb = (
            V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS
        )  # thrust-to-weight ratio in climb conditions [-]
        F_pro_climb = TW_climb * Weight / Npro  # [N] Thrust per propeller for climb
        alpha_cl = (
            np.pi / 2
        )  # [rad] Rotor disk Angle of Attack (assumption: axial flight TODO: estimate trim?)

        outputs["data:propulsion:%s:propeller:thrust:climb" % propulsion_id] = F_pro_climb
        outputs["data:propulsion:%s:propeller:AoA:climb" % propulsion_id] = alpha_cl

    def compute_partials(self, inputs, partials):
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]
        out_id_1 = "data:propulsion:%s:propeller:thrust:climb" % propulsion_id
        out_id_2 = "data:propulsion:%s:propeller:AoA:climb" % propulsion_id

        # Flight parameters
        V_v = inputs["mission:sizing:main_route:climb:rate:%s" % propulsion_id]
        V_climb = inputs["mission:sizing:main_route:climb:speed:%s" % propulsion_id]
        altitude_climb = inputs["mission:sizing:main_route:cruise:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_climb, dISA)
        atm.true_airspeed = V_climb
        q_climb = atm.dynamic_pressure
        rho = atm.density

        # ===== DENSITY DERIVATIVES (Analytic, Troposphere) =====
        # Temperature profile: T = T0 - L * h + dISA
        # Density varies as: rho ∝ (T/T0)^(g/RL - 1) --> (g/(R*L) - 1) ≈ -5.256 for ISA
        from stdatm import AtmosphereWithPartials

        datm = AtmosphereWithPartials(altitude_climb, dISA, altitude_in_feet=False)
        # drho = datm.density
        # d(rho)/d(h):   chain rule through T
        # d(rho)/d(h) = d(rho)/d(T) * d(T)/d(h) = (g/(R*L) - 1) * rho / T * (-L)
        drho_dh = datm.partial_density_altitude  # d(rho)/d(altitude)
        # d(rho)/d(dISA): stdatm treats dISA as a pure temperature offset at fixed pressure
        drho_ddISA = -rho / (T0 - L * altitude_climb + dISA)

        # Weight
        m_uav_guess = inputs["optimization:variables:weight:mtow:guess"]
        Weight = m_uav_guess * g  # [N]

        # Induced drag parameters
        K = inputs["optimization:variables:aerodynamics:CDi:K:guess"]

        # Parasitic drag parameters
        CD_0_guess = inputs["optimization:variables:aerodynamics:CD0:guess"]

        # Thrust calculation (equilibrium)
        TW_climb = (
            V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS
        )  # thrust-to-weight ratio in climb conditions [-]

        partials[out_id_1, "mission:sizing:main_route:climb:rate:%s" % propulsion_id] = (
            Weight / Npro / V_climb
        )

        partials[out_id_1, "optimization:variables:aerodynamics:CD0:guess"] = (
            q_climb / WS * Weight / Npro
        )

        partials[out_id_1, "optimization:variables:aerodynamics:CDi:K:guess"] = (
            WS / q_climb * Weight / Npro
        )

        # ∂(q_climb)/∂(V_climb) = ∂(0.5 * ρ * V^2)/∂(V_climb) = ρ * V_climb
        partials[out_id_1, "mission:sizing:main_route:climb:speed:%s" % propulsion_id] = (
            (-V_v / V_climb**2 + rho * V_climb * (CD_0_guess / WS - K / (q_climb**2) * WS))
            * Weight
            / Npro
        )
        # ∂(q_climb)/∂(h) = 0.5 * V^2 * d(rho)/d(dISA)
        partials[out_id_1, "mission:sizing:main_route:cruise:altitude"] = (
            0.5 * V_climb**2 * drho_dh * (CD_0_guess / WS - K / (q_climb**2) * WS) * Weight / Npro
        )
        # ∂(q_climb)/∂(dISA) = 0.5 * V^2 * d(rho)/d(dISA)
        partials[out_id_1, "mission:sizing:dISA"] = (
            0.5
            * V_climb**2
            * drho_ddISA
            * (CD_0_guess / WS - K / (q_climb**2) * WS)
            * Weight
            / Npro
        )

        partials[out_id_1, "data:geometry:wing:loading"] = (
            (-q_climb * CD_0_guess / WS**2 + K / q_climb) * Weight / Npro
        )

        partials[out_id_1, "optimization:variables:weight:mtow:guess"] = TW_climb * g / Npro
        partials[out_id_1, "data:propulsion:%s:propeller:number" % propulsion_id] = (
            -TW_climb * Weight / Npro**2
        )

        # AoA is assumed to be constant (axial flight), so its partials are zero.
        partials[out_id_2, "mission:sizing:main_route:climb:rate:%s" % propulsion_id] = 0.0
        partials[out_id_2, "optimization:variables:aerodynamics:CD0:guess"] = 0.0
        partials[out_id_2, "optimization:variables:aerodynamics:CDi:K:guess"] = 0.0
        partials[out_id_2, "mission:sizing:main_route:climb:speed:%s" % propulsion_id] = 0.0
        partials[out_id_2, "mission:sizing:main_route:cruise:altitude"] = 0.0
        partials[out_id_2, "mission:sizing:dISA"] = 0.0
        partials[out_id_2, "data:geometry:wing:loading"] = 0.0
        partials[out_id_2, "optimization:variables:weight:mtow:guess"] = 0.0
        partials[out_id_2, "data:propulsion:%s:propeller:number" % propulsion_id] = 0.0
