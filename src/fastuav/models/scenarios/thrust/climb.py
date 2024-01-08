"""
Climb scenarios
"""

import numpy as np
from scipy.constants import g
import openmdao.api as om
from stdatm import AtmosphereSI
from fastuav.models.scenarios.thrust.flight_models import MultirotorFlightModel
from fastuav.constants import FW_PROPULSION, MR_PROPULSION


class MultirotorClimbThrust(om.ExplicitComponent):
    """
    Thrust for a climb in multirotor mode, at desired rate of climb and climb speed
    (i.e., the flight path is not necessarily vertical)
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:aerodynamics:%s:CD0" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
        self.add_input("data:geometry:projected_area:front", val=0.0, units="m**2")  # TODO: define front area for hybrid VTOL UAVs?
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:climb:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:main_route:climb:rate:%s" % propulsion_id, val=np.nan, units="m/s")
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
        altitude_climb = inputs["mission:sizing:main_route:cruise:altitude"]  # conservative assumption
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_climb, dISA)
        atm.true_airspeed = V_climb
        rho_air = atm.density

        # Weight
        m_uav_guess = inputs["data:weight:mtow:guess"]

        # Drag parameters
        C_D0 = inputs["data:aerodynamics:%s:CD0" % propulsion_id]
        C_L = 0.0  # it is assumed that the body shape produces only pressure drag and no lift
        S_top = inputs["data:geometry:projected_area:top"]
        S_front = inputs["data:geometry:projected_area:front"]

        alpha_cl = MultirotorFlightModel.get_angle_of_attack(m_uav_guess,
                                                             V_climb,
                                                             V_v,
                                                             S_front,
                                                             S_top,
                                                             C_D0,
                                                             C_L,
                                                             rho_air)  # [rad] angle of attack
        F_pro_cl = MultirotorFlightModel.get_thrust(m_uav_guess,
                                                    V_climb,
                                                    V_v,
                                                    alpha_cl,
                                                    S_front,
                                                    S_top,
                                                    C_D0,
                                                    C_L,
                                                    rho_air) / Npro  # [N] thrust per propeller

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
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_input("mission:sizing:main_route:cruise:altitude", val=150.0, units="m")
        self.add_input("mission:sizing:main_route:climb:speed:%s" % propulsion_id, val=0.0, units="m/s")
        self.add_input("mission:sizing:main_route:climb:rate:%s" % propulsion_id, val=np.nan, units="m/s")
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
        altitude_climb = inputs["mission:sizing:main_route:cruise:altitude"]  # conservative assumption
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_climb, dISA)
        atm.true_airspeed = V_climb
        q_climb = atm.dynamic_pressure

        # Weight
        m_uav_guess = inputs["data:weight:mtow:guess"]
        Weight = m_uav_guess * g

        # Induced drag parameters
        K = inputs["data:aerodynamics:CDi:K"]

        # Parasitic drag parameters
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]

        # Thrust and trim calculation (equilibrium)
        TW_climb = (
            V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS
        )  # thrust-to-weight ratio in climb conditions [-]
        F_pro_climb = TW_climb * Weight / Npro  # [N] Thrust per propeller for climb
        alpha_cl = np.pi / 2  # [rad] Rotor disk Angle of Attack (assumption: axial flight TODO: estimate trim?)

        outputs["data:propulsion:%s:propeller:thrust:climb" % propulsion_id] = F_pro_climb
        outputs["data:propulsion:%s:propeller:AoA:climb" % propulsion_id] = alpha_cl



