"""
Gearbox model
"""
import openmdao.api as om
import numpy as np


class Gearbox(om.ExplicitComponent):
    """
    Simple Gearbox Model
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:motor:torque:nominal", val=np.nan, units="N*m")
        self.add_output("data:weight:propulsion:gearbox:mass", units="kg")
        self.add_output("data:propulsion:gearbox:gear_diameter", units="m")
        self.add_output("data:propulsion:gearbox:pinion_diameter", units="m")
        self.add_output("data:propulsion:gearbox:inner_diameter", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        N_red = inputs["data:propulsion:gearbox:N_red"]
        T_mot_nom = inputs["data:propulsion:motor:torque:nominal"]

        mg1 = 0.0309 * N_red**2 + 0.1944 * N_red + 0.6389  # Ratio input pinion to mating gear
        WF = (
            1 + 1 / mg1 + mg1 + mg1**2 + N_red**2 / mg1 + N_red**2
        )  # Weight Factor (ƩFd2/C) [-]
        k_sd = 1000  # Surface durability factor [lb/in]
        C = 2 * 8.85 * T_mot_nom / k_sd  # Coefficient (C=2T/K) [in3]
        Fd2 = WF * C  # Solid rotor volume [in3]
        Mgear = (
            Fd2 * 0.3 * 0.4535
        )  # Mass reducer [kg] (0.3 is a coefficient evaluated for aircraft application and 0.4535 to pass from lb to kg)
        Fdp2 = C * (N_red + 1) / N_red  # Solid rotor pinion volume [in3]
        dp = (Fdp2 / 0.7) ** (1 / 3) * 0.0254  # Pinion diameter [m] (0.0254 to pass from in to m)
        dg = N_red * dp  # Gear diameter [m]
        di = mg1 * dp  # Inner diameter [m]

        outputs["data:weight:propulsion:gearbox:mass"] = Mgear
        outputs["data:propulsion:gearbox:gear_diameter"] = dg
        outputs["data:propulsion:gearbox:pinion_diameter"] = dp
        outputs["data:propulsion:gearbox:inner_diameter"] = di


class NoGearbox(om.ExplicitComponent):
    """
    No gearbox: sets the value 'data:weight:propulsion:gearbox:mass' to 0.0 and reduction ratio to 1.0
    """

    def setup(self):
        self.add_output("data:weight:propulsion:gearbox:mass", units="kg")
        self.add_output("data:propulsion:gearbox:N_red", units=None)

    def compute(self, inputs, outputs):
        outputs["data:weight:propulsion:gearbox:mass"] = 0.0
        outputs["data:propulsion:gearbox:N_red"] = 1.0
