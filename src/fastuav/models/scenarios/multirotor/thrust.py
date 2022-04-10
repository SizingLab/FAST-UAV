"""
Thrust requirements for multirotors.
"""

import numpy as np
from scipy.constants import g
from scipy.optimize import brentq
import openmdao.api as om


class ThrustHoverMR(om.ExplicitComponent):
    """
    Thrust to maintain hover.
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:thrust:hover", units="N")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]

        F_pro_hov = Mtotal_guess * g / Npro  # [N] Thrust per propeller for hover

        outputs["data:propulsion:propeller:thrust:hover"] = F_pro_hov


class ThrustTakeOffMR(om.ExplicitComponent):
    """
    Thrust for the desired takeoff acceleration.
    """

    def setup(self):
        self.add_input("specifications:maxthrust:k", val=np.nan, units=None)
        self.add_input("data:propulsion:propeller:thrust:hover", val=np.nan, units="N")
        self.add_output("data:propulsion:propeller:thrust:takeoff", units="N")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        F_pro_hov = inputs["data:propulsion:propeller:thrust:hover"]
        k_maxthrust = inputs["specifications:maxthrust:k"]

        F_pro_to = F_pro_hov * k_maxthrust  # [N] max propeller thrust

        outputs["data:propulsion:propeller:thrust:takeoff"] = F_pro_to


class ThrustClimbMR(om.ExplicitComponent):
    """
    Thrust for the desired rate of climb.
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:geometry:body:surface:top", val=np.nan, units="m**2")
        self.add_input("mission:design_mission:climb:q", val=np.nan, units="Pa")
        self.add_output("data:propulsion:propeller:thrust:climb", units="N")
        self.add_output("mission:design_mission:climb:AoA", units="rad")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        C_D0 = inputs["data:aerodynamics:CD0"]
        S_top_estimated = inputs["data:geometry:body:surface:top"]
        q_climb = inputs["mission:design_mission:climb:q"]

        F_pro_cl = (
            Mtotal_guess * g + q_climb * C_D0 * S_top_estimated
        ) / Npro

        alpha_cl = np.pi / 2  # [rad] Rotor disk Angle of Attack (assumption: axial flight)

        # PROVISION FOR CLIMBING FORWARD FLIGHT (PATH ANGLE THETA)
        # theta = np.pi / 2  # [rad] flight path angle (vertical climb)
        # F_pro_cl, alpha_cl = MultirotorFlightModel.get_thrust(Mtotal_guess, V_cl, theta, S_front_estimated, S_top_estimated, C_D, C_L0, rho_air)  # [N] required thrust (and angle of attack)
        # F_pro_cl = F_pro_cl / Npro  # [N] thrust per propeller

        outputs["data:propulsion:propeller:thrust:climb"] = F_pro_cl
        outputs["mission:design_mission:climb:AoA"] = alpha_cl


class ThrustCruiseMR(om.ExplicitComponent):
    """
    Thrust for the desired cruise speed.
    """

    def setup(self):
        self.add_input("data:weights:MTOW:guess", val=np.nan, units="kg")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CD0", val=np.nan, units=None)
        self.add_input("data:aerodynamics:CLmax", val=np.nan, units=None)
        self.add_input("data:geometry:body:surface:top", val=np.nan, units="m**2")
        self.add_input("data:geometry:body:surface:front", val=np.nan, units="m**2")
        self.add_input("mission:design_mission:cruise:q", val=np.nan, units="Pa")
        self.add_output("data:propulsion:propeller:thrust:cruise", units="N")
        self.add_output("mission:design_mission:cruise:AoA", units="rad")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Mtotal_guess = inputs["data:weights:MTOW:guess"]
        Npro = inputs["data:propulsion:propeller:number"]
        C_D = inputs["data:aerodynamics:CD0"]
        C_L0 = inputs["data:aerodynamics:CLmax"]
        S_top_estimated = inputs["data:geometry:body:surface:top"]
        S_front_estimated = inputs["data:geometry:body:surface:front"]
        q_cruise = inputs["mission:design_mission:cruise:q"]

        func = lambda x: np.tan(x) - q_cruise * C_D * (
            S_top_estimated * np.sin(x) + S_front_estimated * np.cos(x)
        ) / (
            Mtotal_guess * g
            + q_cruise
            * C_L0
            * (S_top_estimated * np.sin(x) + S_front_estimated * np.cos(x))
        )
        alpha_cr = brentq(func, 0, np.pi / 2)  # [rad] angle of attack
        S_ref = S_top_estimated * np.sin(alpha_cr) + S_front_estimated * np.cos(
            alpha_cr
        )  # [m2] reference surface for drag and lift calculation
        drag = q_cruise * C_D * S_ref  # [N] drag
        lift = - q_cruise * C_L0 * S_ref # [N] lift (downwards force)
        weight = Mtotal_guess * g  # [N] weight
        F_pro_cr = ((weight - lift) ** 2 + drag**2) ** (
            1 / 2
        ) / Npro  # [N] thrust per propeller

        # lift = - 0.5 * rho_air * C_L0 * np.sin(2*alpha) * S_top_estimated * np.sin(alpha) * V_cr ** 2  # [N] downwards force (flat plate model)
        # F_pro_cr = ((weight - lift) ** 2 + drag ** 2) ** (1 / 2) / Npro  # [N] thrust per propeller

        # PROVISION FOR CLIMBING FORWARD FLIGHT (PATH ANGLE THETA)
        # theta = 0  # [rad] flight path angle (steady level flight)
        # F_pro_cr, alpha_cr = MultirotorFlightModel.get_thrust(Mtotal_guess, V_cr, theta, S_front_estimated, S_top_estimated, C_D, C_L0, rho_air)  # [N] required thrust (and angle of attack)
        # F_pro_cr = F_pro_cr / Npro # [N] thrust per propeller

        outputs["data:propulsion:propeller:thrust:cruise"] = F_pro_cr
        outputs["mission:design_mission:cruise:AoA"] = alpha_cr