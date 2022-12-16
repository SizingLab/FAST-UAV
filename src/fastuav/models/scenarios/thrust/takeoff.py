"""
Takeoff scenarios
"""

import numpy as np
from scipy.constants import g
import openmdao.api as om
from stdatm import AtmosphereSI
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION


class VerticalTakeoffThrust(om.ExplicitComponent):
    """
    Thrust for the desired vertical takeoff acceleration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("mission:sizing:thrust_weight_ratio:%s" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_output("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, units="N")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        k_maxthrust = inputs["mission:sizing:thrust_weight_ratio:%s" % propulsion_id]
        m_uav_guess = inputs["data:weight:mtow:guess"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        F_pro_to = m_uav_guess * g / Npro * k_maxthrust  # [N] Thrust per propeller

        outputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id] = F_pro_to


class LauncherTakeoff(om.ExplicitComponent):
    """
    Thrust required for takeoff assuming the use of a rail launcher or bungee, in fixed wing configuration.
    The launching system brings the UAV at the required speed for takeoff (10% margin on stall speed).
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:weight:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:geometry:wing:loading", val=np.nan, units="N/m**2")
        self.add_input("mission:sizing:main_route:takeoff:altitude", val=0.0, units="m")
        self.add_input("mission:sizing:main_route:stall:speed:%s" % propulsion_id, val=np.nan, units="m/s")
        self.add_input("mission:sizing:dISA", val=0.0, units="K")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id, units="N")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:geometry:wing:loading"]

        # Flight parameters
        V_stall = inputs["mission:sizing:main_route:stall:speed:%s" % propulsion_id]
        altitude_takeoff = inputs["mission:sizing:main_route:takeoff:altitude"]
        dISA = inputs["mission:sizing:dISA"]
        atm = AtmosphereSI(altitude_takeoff, dISA)
        atm.true_airspeed = 1.1 * V_stall  # 10% margin on the stall speed [kg/ms2]
        q_takeoff = atm.dynamic_pressure

        # Weight
        m_uav_guess = inputs["data:weight:mtow:guess"]
        Weight = m_uav_guess * g  # [N]

        # Induced drag parameters
        K = inputs["data:aerodynamics:CDi:K"]

        # Parasitic drag parameters
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]

        # Thrust calculation (equilibrium)
        TW_takeoff = (
            q_takeoff * CD_0_guess / WS + K / q_takeoff * WS
        )  # thrust-to-weight ratio at takeoff  [-]
        F_pro_takeoff = TW_takeoff * Weight / Npro  # [N] Thrust per propeller for takeoff

        outputs["data:propulsion:%s:propeller:thrust:takeoff" % propulsion_id] = F_pro_takeoff
