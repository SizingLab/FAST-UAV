"""
Thrust requirements for fixed wings.
"""

import numpy as np
from scipy.constants import g
import openmdao.api as om


class ThrustCruiseFW(om.ExplicitComponent):
    """
    Thrust for the desired cruise speed.
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=1.0, units=None)
        self.add_input("data:loads:wing_loading", val=np.nan, units="N/m**2")
        self.add_input("mission:design_mission:cruise:q", val=np.nan, units="Pa")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:thrust:cruise", units="N")
        self.add_output("mission:design_mission:cruise:AoA", units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_cruise = inputs["mission:design_mission:cruise:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]

        TW_cruise = (
            q_cruise * CD_0_guess / WS + K / q_cruise * WS
        )  # [-] thrust-to-weight ratio in cruise conditions
        F_pro_cruise = TW_cruise * Mtotal_guess * g / Npro  # [N] Thrust per propeller for cruise

        alpha_cr = np.pi / 2  # [rad] Rotor disk Angle of Attack (assumption: axial flight)

        outputs["data:propulsion:propeller:thrust:cruise"] = F_pro_cruise
        outputs["mission:design_mission:cruise:AoA"] = alpha_cr

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_cruise = inputs["mission:design_mission:cruise:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]
        TW_cruise = q_cruise * CD_0_guess / WS + K / q_cruise * WS

        partials["data:propulsion:propeller:thrust:cruise", "data:weights:MTOW:guess"] = (
            TW_cruise * g / Npro
        )
        partials["data:propulsion:propeller:thrust:cruise", "data:propulsion:propeller:number"] = (
            -TW_cruise * Mtotal_guess * g / Npro**2
        )
        partials["data:propulsion:propeller:thrust:cruise", "data:loads:wing_loading"] = (
            Mtotal_guess * g / Npro * (-q_cruise * CD_0_guess / WS**2 + K / q_cruise)
        )
        partials["data:propulsion:propeller:thrust:cruise", "mission:design_mission:cruise:q"] = (
            Mtotal_guess * g / Npro * (CD_0_guess / WS - K / q_cruise**2 * WS)
        )
        partials["data:propulsion:propeller:thrust:cruise", "data:aerodynamics:CD0:guess"] = (
            Mtotal_guess * g / Npro * q_cruise / WS
        )
        partials["data:propulsion:propeller:thrust:cruise", "data:aerodynamics:CDi:K"] = (
            Mtotal_guess * g / Npro / q_cruise * WS
        )


class ThrustClimbFW(om.ExplicitComponent):
    """
    Thrust for the desired rate of climb
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=1.0, units=None)
        self.add_input("data:loads:wing_loading", val=np.nan, units="N/m**2")
        self.add_input("mission:design_mission:climb:q", val=np.nan, units="Pa")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_input("mission:design_mission:climb:rate", val=np.nan, units="m/s")
        self.add_input("mission:design_mission:climb:speed", val=np.nan, units="m/s")
        self.add_output("data:propulsion:propeller:thrust:climb", units="N")
        self.add_output("mission:design_mission:climb:AoA", units="rad")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_climb = inputs["mission:design_mission:climb:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]
        V_v = inputs["mission:design_mission:climb:rate"]
        V_climb = inputs["mission:design_mission:climb:speed"]

        TW_climb = (
            V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS
        )  # thrust-to-weight ratio in climb conditions [-]
        F_pro_climb = TW_climb * Mtotal_guess * g / Npro  # [N] Thrust per propeller for climb

        alpha_cl = (
            np.pi / 2
        )  # [rad] Rotor disk Angle of Attack (assumption: axial flight TODO: estimate trim?)

        outputs["data:propulsion:propeller:thrust:climb"] = F_pro_climb
        outputs["mission:design_mission:climb:AoA"] = alpha_cl

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_climb = inputs["mission:design_mission:climb:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]
        V_v = inputs["mission:design_mission:climb:rate"]
        V_climb = inputs["mission:design_mission:climb:speed"]
        TW_climb = V_v / V_climb + q_climb * CD_0_guess / WS + K / q_climb * WS

        partials["data:propulsion:propeller:thrust:climb", "data:weights:MTOW:guess"] = (
            TW_climb * g / Npro
        )
        partials["data:propulsion:propeller:thrust:climb", "data:propulsion:propeller:number"] = (
            -TW_climb * Mtotal_guess * g / Npro**2
        )
        partials["data:propulsion:propeller:thrust:climb", "data:loads:wing_loading"] = (
            Mtotal_guess * g / Npro * (-q_climb * CD_0_guess / WS**2 + K / q_climb)
        )
        partials["data:propulsion:propeller:thrust:climb", "mission:design_mission:climb:q"] = (
            Mtotal_guess * g / Npro * (CD_0_guess / WS - K / q_climb**2 * WS)
        )
        partials["data:propulsion:propeller:thrust:climb", "data:aerodynamics:CD0:guess"] = (
            Mtotal_guess * g / Npro * q_climb / WS
        )
        partials["data:propulsion:propeller:thrust:climb", "data:aerodynamics:CDi:K"] = (
            Mtotal_guess * g / Npro / q_climb * WS
        )
        partials["data:propulsion:propeller:thrust:climb", "mission:design_mission:climb:rate"] = (
            Mtotal_guess * g / Npro / V_climb
        )
        partials["data:propulsion:propeller:thrust:climb", "mission:design_mission:climb:speed"] = (
            -Mtotal_guess * g / Npro * V_v / V_climb**2
        )


class ThrustTakeOffFW(om.ExplicitComponent):
    """
    It is assumed that the takeoff is achieved with a rail launcher or bungee.
    The launching system brings the UAV at the required speed for takeoff (10% margin on stall speed).
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=1.0, units=None)
        self.add_input("data:loads:wing_loading", val=np.nan, units="N/m**2")
        self.add_input("mission:design_mission:stall:q", val=np.nan, units="Pa")
        self.add_input("data:aerodynamics:CD0:guess", val=0.04, units=None)
        self.add_input("data:aerodynamics:CDi:K", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:thrust:takeoff", units="N")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_takeoff = (
            1.21 * inputs["mission:design_mission:stall:q"]
        )  # dynamic pressure at takeoff, considering a catapult launch with 10% margin on the stall speed [kg/ms2]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]

        TW_takeoff = (
            q_takeoff * CD_0_guess / WS + K / q_takeoff * WS
        )  # thrust-to-weight ratio at takeoff  [-]
        F_pro_takeoff = TW_takeoff * Mtotal_guess * g / Npro  # [N] Thrust per propeller for climb

        outputs["data:propulsion:propeller:thrust:takeoff"] = F_pro_takeoff

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        WS = inputs["data:loads:wing_loading"]
        q_takeoff = 1.21 * inputs["mission:design_mission:stall:q"]
        CD_0_guess = inputs["data:aerodynamics:CD0:guess"]
        K = inputs["data:aerodynamics:CDi:K"]
        TW_takeoff = q_takeoff * CD_0_guess / WS + K / q_takeoff * WS

        partials["data:propulsion:propeller:thrust:takeoff", "data:weights:MTOW:guess"] = (
            TW_takeoff * g / Npro
        )
        partials["data:propulsion:propeller:thrust:takeoff", "data:propulsion:propeller:number"] = (
            -TW_takeoff * Mtotal_guess * g / Npro**2
        )
        partials["data:propulsion:propeller:thrust:takeoff", "data:loads:wing_loading"] = (
            Mtotal_guess * g / Npro * (-q_takeoff * CD_0_guess / WS**2 + K / q_takeoff)
        )
        partials["data:propulsion:propeller:thrust:takeoff", "mission:design_mission:stall:q"] = (
            Mtotal_guess * g / Npro * (CD_0_guess / WS - K / q_takeoff**2 * WS)
        )
        partials["data:propulsion:propeller:thrust:takeoff", "data:aerodynamics:CD0:guess"] = (
            Mtotal_guess * g / Npro * q_takeoff / WS
        )
        partials["data:propulsion:propeller:thrust:takeoff", "data:aerodynamics:CDi:K"] = (
            Mtotal_guess * g / Npro / q_takeoff * WS
        )
