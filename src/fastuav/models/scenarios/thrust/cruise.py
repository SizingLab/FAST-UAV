"""
Cruise scenarios
"""

import numpy as np
from scipy.constants import g
from scipy.optimize import brentq
import openmdao.api as om
from stdatm import AtmosphereSI
from fastuav.utils.constants import FW_PROPULSION, MR_PROPULSION


class MultirotorCruiseThrust(om.ExplicitComponent):
    """
    Thrust for the desired cruise speed, in multirotor configuration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:aerodynamics:%s:CD" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:aerodynamics:%s:CL" % propulsion_id, val=np.nan, units=None)
        self.add_input("data:geometry:projected_area:top", val=np.nan, units="m**2")
        self.add_input("data:geometry:projected_area:front", val=np.nan, units="m**2")
        self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=0.0, units="m")
        self.add_input("data:scenarios:%s:cruise:speed" % propulsion_id, val=0.0, units="m/s")
        self.add_input("data:scenarios:dISA", val=0.0, units="K")
        self.add_output("data:propulsion:%s:propeller:thrust:cruise" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:cruise" % propulsion_id, units="rad")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]

        # Flight parameters
        V_cruise = inputs["data:scenarios:%s:cruise:speed" % propulsion_id]
        altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
        dISA = inputs["data:scenarios:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_cruise
        q_cruise = atm.dynamic_pressure

        # Weight  # [N]
        Mtotal_guess = inputs["data:weights:mtow:guess"]
        weight = Mtotal_guess * g

        # Drag and lift parameters
        C_D = inputs["data:aerodynamics:%s:CD" % propulsion_id]
        C_L = inputs["data:aerodynamics:%s:CL" % propulsion_id]
        S_top = inputs["data:geometry:projected_area:top"]
        S_front = inputs["data:geometry:projected_area:front"]

        # Thrust and trim calculation (equilibrium)
        func = lambda x: np.tan(x) - q_cruise * C_D * (
            S_top * np.sin(x) + S_front * np.cos(x)
        ) / (
            Mtotal_guess * g
            + q_cruise * C_L * (S_top * np.sin(x) + S_front * np.cos(x))
        )
        alpha_cr = brentq(func, 0, np.pi / 2)  # [rad] angle of attack
        S_ref = S_top * np.sin(alpha_cr) + S_front * np.cos(
            alpha_cr
        )  # [m2] reference area for drag and lift calculation
        drag = q_cruise * C_D * S_ref  # [N] drag
        lift = - q_cruise * C_L * S_ref  # [N] lift (downwards force)
        # lift = - q_cruise * C_L0 * np.sin(2*alpha) * S_top_estimated * np.sin(alpha)  # Flat plate model
        F_pro_cr = ((weight - lift) ** 2 + drag ** 2) ** (1 / 2) / Npro  # [N] thrust per propeller

        outputs["data:propulsion:%s:propeller:thrust:cruise" % propulsion_id] = F_pro_cr
        outputs["data:propulsion:%s:propeller:AoA:cruise" % propulsion_id] = alpha_cr


class FixedwingCruiseThrust(om.ExplicitComponent):
    """
    Thrust for the desired cruise speed, in fixed wing configuration.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=FW_PROPULSION, values=[FW_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_input("data:weights:mtow:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:%s:propeller:number" % propulsion_id, val=1.0, units=None)
        self.add_input("data:scenarios:wing_loading", val=np.nan, units="N/m**2")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_input("data:scenarios:%s:cruise:altitude" % propulsion_id, val=0.0, units="m")
        self.add_input("data:scenarios:%s:cruise:speed" % propulsion_id, val=0.0, units="m/s")
        self.add_input("data:scenarios:dISA", val=0.0, units="K")
        self.add_output("data:propulsion:%s:propeller:thrust:cruise" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:cruise" % propulsion_id, units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        # UAV configuration
        propulsion_id = self.options["propulsion_id"]
        Npro = inputs["data:propulsion:%s:propeller:number" % propulsion_id]
        WS = inputs["data:scenarios:wing_loading"]

        # Flight parameters
        V_cruise = inputs["data:scenarios:%s:cruise:speed" % propulsion_id]
        altitude_cruise = inputs["data:scenarios:%s:cruise:altitude" % propulsion_id]
        dISA = inputs["data:scenarios:dISA"]
        atm = AtmosphereSI(altitude_cruise, dISA)
        atm.true_airspeed = V_cruise
        q_cruise = atm.dynamic_pressure

        # Weight
        Mtotal_guess = inputs["data:weights:mtow:guess"]
        Weight = Mtotal_guess * g  # [N]

        # Induced drag parameter
        K = inputs["data:aerodynamics:CDi:K"]

        # Parasitic drag parameter
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]

        # Thrust and trim calculation (equilibrium)
        TW_cruise = (
            q_cruise * CD_0_guess / WS + K / q_cruise * WS
        )  # [-] thrust-to-weight ratio in cruise conditions
        F_pro_cruise = TW_cruise * Weight / Npro  # [N] Thrust per propeller for cruise
        alpha_cr = np.pi / 2  # [rad] Rotor disk Angle of Attack (assumption: axial flight)

        outputs["data:propulsion:%s:propeller:thrust:cruise" % propulsion_id] = F_pro_cruise
        outputs["data:propulsion:%s:propeller:AoA:cruise" % propulsion_id] = alpha_cr


class NoCruise(om.ExplicitComponent):
    """
    Simple component to declare the absence of cruise scenario.
    """

    def initialize(self):
        self.options.declare("propulsion_id", default=MR_PROPULSION, values=[MR_PROPULSION])

    def setup(self):
        propulsion_id = self.options["propulsion_id"]
        self.add_output("data:propulsion:%s:propeller:thrust:cruise" % propulsion_id, units="N")
        self.add_output("data:propulsion:%s:propeller:AoA:cruise" % propulsion_id, units="rad")

    def compute(self, inputs, outputs):
        propulsion_id = self.options["propulsion_id"]
        outputs["data:propulsion:%s:propeller:thrust:cruise" % propulsion_id] = 0.0
        outputs["data:propulsion:%s:propeller:AoA:cruise" % propulsion_id] = np.pi / 2