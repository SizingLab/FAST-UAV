"""
Atmosphere calculations for sizing scenarios.
"""

import numpy as np
from stdatm import AtmosphereSI
import openmdao.api as om


class Atmosphere(om.ExplicitComponent):
    """
    Computes atmosphere conditions
    """

    def setup(self):
        self.add_input("mission:design_mission:climb:height", val=0.0, units="m")
        self.add_input("mission:design_mission:takeoff:altitude", val=0.0, units="m")
        self.add_input("mission:design_mission:dISA", val=0.0, units="K")
        self.add_input("mission:design_mission:stall:speed", val=0.0, units="m/s")
        self.add_input("mission:design_mission:cruise:speed", val=20.0, units="m/s")
        self.add_input("mission:design_mission:climb:speed", val=10.0, units="m/s")
        self.add_output("mission:design_mission:stall:q", units="Pa")
        self.add_output("mission:design_mission:cruise:q", units="Pa")
        self.add_output("mission:design_mission:climb:q", units="Pa")
        self.add_output("mission:design_mission:cruise:altitude", units="m")
        self.add_output("mission:design_mission:cruise:atmosphere:viscosity", units="m**2/s")
        self.add_output("mission:design_mission:cruise:atmosphere:speedofsound", units="m/s")
        self.add_output("mission:design_mission:cruise:atmosphere:density", units="kg/m**3")
        self.add_output("mission:design_mission:climb:atmosphere:density", units="kg/m**3")
        self.add_output("mission:design_mission:takeoff:atmosphere:density", units="kg/m**3")
        self.add_output("mission:design_mission:hover:atmosphere:density", units="kg/m**3")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        altitude_TO = inputs["mission:design_mission:takeoff:altitude"]
        dISA = inputs["mission:design_mission:dISA"]
        D_cl = inputs["mission:design_mission:climb:height"]
        V_stall = inputs["mission:design_mission:stall:speed"]
        V_cruise = inputs["mission:design_mission:cruise:speed"]
        V_climb = inputs["mission:design_mission:climb:speed"]

        altitude_cruise = altitude_TO + D_cl  # [m] Cruise altitude
        rho_air_stall = AtmosphereSI(altitude_cruise, dISA).density  # [kg/m3] Air density at stall conditions (min.)
        rho_air_takeoff = AtmosphereSI(altitude_TO, dISA).density  # [kg/m3] Air density at takeoff level
        rho_air_hover = AtmosphereSI(altitude_cruise, dISA).density  # [kg/m3] Air density at hovering level
        rho_air_climb = AtmosphereSI(altitude_cruise, dISA).density  # [kg/m3] Air density at climb conditions (min.)
        rho_air_cruise = AtmosphereSI(altitude_cruise, dISA).density  # [kg/m3] Air density at cruise conditions
        nu_air_cruise = AtmosphereSI(altitude_cruise,
                                     dISA).kinematic_viscosity  # [m2/s] kinematic viscosity at cruise level
        a_air_cruise = AtmosphereSI(altitude_cruise, dISA).speed_of_sound  # [m/s] speed of sound at cruise level
        q_stall = 0.5 * rho_air_stall * V_stall ** 2  # dynamic pressure at stall speed [Pa]
        q_cruise = 0.5 * rho_air_cruise * V_cruise ** 2  # dynamic pressure at cruise speed [Pa]
        q_climb = 0.5 * rho_air_climb * V_climb ** 2  # dynamic pressure at climb speed [Pa]

        outputs["mission:design_mission:cruise:altitude"] = altitude_cruise
        outputs["mission:design_mission:cruise:atmosphere:density"] = rho_air_cruise
        outputs["mission:design_mission:climb:atmosphere:density"] = rho_air_climb
        outputs["mission:design_mission:hover:atmosphere:density"] = rho_air_hover
        outputs["mission:design_mission:takeoff:atmosphere:density"] = rho_air_takeoff
        outputs["mission:design_mission:cruise:atmosphere:viscosity"] = nu_air_cruise
        outputs["mission:design_mission:cruise:atmosphere:speedofsound"] = a_air_cruise
        outputs["mission:design_mission:cruise:q"] = q_cruise
        outputs["mission:design_mission:climb:q"] = q_climb
        outputs["mission:design_mission:stall:q"] = q_stall

