"""
Definition parameters for the motor.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class MotorDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the motor.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the motor are the maximum torque and the torque coefficient.
    """

    def setup(self):
        self.add_subsystem("battery_voltage_guess", BatteryVoltageEstimation(), promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "max_torque",
            MaxTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:max:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "torque_coefficient",
            TorqueCoefficient(),
            uncertain_outputs={"data:propulsion:motor:torque:coefficient:estimated": "N*m/A"},
        )


class MaxTorque(om.ExplicitComponent):
    """
    Estimates the maximum motor torque from the takeoff flight scenario.
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:propeller:torque:takeoff", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:k", val=np.nan, units=None)
        self.add_output("data:propulsion:motor:torque:max:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Qpro_to = inputs["data:propulsion:propeller:torque:takeoff"]
        k_mot = inputs["data:propulsion:motor:torque:k"]

        Tmot_to = Qpro_to / Nred  # [N.m] takeoff torque
        Tmot_max = k_mot * Tmot_to  # [N.m] required motor nominal torque

        outputs["data:propulsion:motor:torque:max:estimated"] = Tmot_max

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Qpro_to = inputs["data:propulsion:propeller:torque:takeoff"]
        k_mot = inputs["data:propulsion:motor:torque:k"]

        partials[
            "data:propulsion:motor:torque:max:estimated", "data:propulsion:gearbox:N_red"
        ] = (-k_mot * Qpro_to / Nred**2)
        partials[
            "data:propulsion:motor:torque:max:estimated",
            "data:propulsion:propeller:torque:takeoff",
        ] = (
            k_mot / Nred
        )
        partials[
            "data:propulsion:motor:torque:max:estimated", "data:propulsion:motor:torque:k"
        ] = (Qpro_to / Nred)


class TorqueCoefficient(om.ExplicitComponent):
    """
    Estimates the motor torque coefficient
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:battery:voltage:guess", val=np.nan, units="V")
        self.add_input("data:propulsion:propeller:speed:takeoff", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:motor:speed:k", val=np.nan, units=None)
        self.add_output("data:propulsion:motor:torque:coefficient:estimated", units="N*m/A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Wpro_to = inputs["data:propulsion:propeller:speed:takeoff"]
        V_bat_guess = inputs["data:propulsion:battery:voltage:guess"]
        k_speed_mot = inputs["data:propulsion:motor:speed:k"]

        W_to_motor = Wpro_to * Nred  # [rad/s] Motor take-off speed
        Ktmot = V_bat_guess / (k_speed_mot * W_to_motor)  # [N.m/A] or [V/(rad/s)]

        outputs["data:propulsion:motor:torque:coefficient:estimated"] = Ktmot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Nred = inputs["data:propulsion:gearbox:N_red"]
        Wpro_to = inputs["data:propulsion:propeller:speed:takeoff"]
        V_bat_guess = inputs["data:propulsion:battery:voltage:guess"]
        k_speed_mot = inputs["data:propulsion:motor:speed:k"]

        partials[
            "data:propulsion:motor:torque:coefficient:estimated", "data:propulsion:gearbox:N_red"
        ] = -V_bat_guess / (Wpro_to * k_speed_mot * Nred**2)

        partials[
            "data:propulsion:motor:torque:coefficient:estimated",
            "data:propulsion:propeller:speed:takeoff",
        ] = -V_bat_guess / (Wpro_to**2 * k_speed_mot * Nred)

        partials[
            "data:propulsion:motor:torque:coefficient:estimated", "data:propulsion:motor:speed:k"
        ] = -V_bat_guess / (Wpro_to * k_speed_mot**2 * Nred)

        partials[
            "data:propulsion:motor:torque:coefficient:estimated",
            "data:propulsion:battery:voltage:guess",
        ] = 1 / (Wpro_to * k_speed_mot * Nred)


class BatteryVoltageEstimation(om.ExplicitComponent):
    """
    Computes an estimation of battery voltage (necessary to calculate motor torque coefficient).
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:power:takeoff", val=np.nan, units="W")
        # self.add_input("data:propulsion:battery:settings:voltage:guess:k", val=np.nan, units=None)
        self.add_output("data:propulsion:battery:voltage:guess", units="V")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Ppro_to = inputs["data:propulsion:propeller:power:takeoff"]
        k_vb_guess = 1  # inputs["data:propulsion:battery:settings:voltage:guess:k"]

        V_bat_guess = k_vb_guess * 1.84 * Ppro_to**0.36  # [V] battery voltage estimation

        outputs["data:propulsion:battery:voltage:guess"] = V_bat_guess

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Ppro_to = inputs["data:propulsion:propeller:power:takeoff"]
        k_vb_guess = 1  # inputs["data:propulsion:battery:settings:voltage:guess:k"]

        partials[
            "data:propulsion:battery:voltage:guess", "data:propulsion:propeller:power:takeoff"
        ] = (k_vb_guess * 0.6624 * Ppro_to ** (-0.64))

        # partials["data:propulsion:battery:voltage:guess", "data:propulsion:battery:settings:voltage:guess:k"] = (
        #     1.84 * Ppro_to**0.36
        # )
