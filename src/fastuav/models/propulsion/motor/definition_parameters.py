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
        add_subsystem_with_deviation(
            self,
            "max_torque",
            MaxTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:max:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "velocity_constant",
            VelocityConstant(),
            uncertain_outputs={"data:propulsion:motor:speed:constant:estimated": "rad/V/s"},
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
        N_red = inputs["data:propulsion:gearbox:N_red"]
        Q_pro_to = inputs["data:propulsion:propeller:torque:takeoff"]
        k_mot = inputs["data:propulsion:motor:torque:k"]

        T_mot_to = Q_pro_to / N_red  # [N.m] takeoff torque
        T_mot_max = k_mot * T_mot_to  # [N.m] required motor nominal torque

        outputs["data:propulsion:motor:torque:max:estimated"] = T_mot_max

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        N_red = inputs["data:propulsion:gearbox:N_red"]
        Q_pro_to = inputs["data:propulsion:propeller:torque:takeoff"]
        k_mot = inputs["data:propulsion:motor:torque:k"]

        partials[
            "data:propulsion:motor:torque:max:estimated", "data:propulsion:gearbox:N_red"
        ] = (-k_mot * Q_pro_to / N_red**2)
        partials[
            "data:propulsion:motor:torque:max:estimated",
            "data:propulsion:propeller:torque:takeoff",
        ] = (
            k_mot / N_red
        )
        partials[
            "data:propulsion:motor:torque:max:estimated", "data:propulsion:motor:torque:k"
        ] = (Q_pro_to / N_red)


class VelocityConstant(om.ExplicitComponent):
    """
    Estimates the motor velocity constant
    """

    def setup(self):
        self.add_input("data:propulsion:gearbox:N_red", val=1.0, units=None)
        self.add_input("data:propulsion:propeller:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:propeller:speed:takeoff", val=np.nan, units="rad/s")
        self.add_input("data:propulsion:motor:speed:k", val=np.nan, units=None)
        self.add_output("data:propulsion:motor:speed:constant:estimated", units="rad/V/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        N_red = inputs["data:propulsion:gearbox:N_red"]
        W_pro_to = inputs["data:propulsion:propeller:speed:takeoff"]
        P_pro_to = inputs["data:propulsion:propeller:power:takeoff"]
        k_speed_mot = inputs["data:propulsion:motor:speed:k"]

        # TODO: replace W_mot_to / U_bat_guess by Kv_hat = 41.59 T_nom ** (-0.35)  (datasheet regression)
        W_mot_to = W_pro_to * N_red  # [rad/s] Motor take-off speed
        U_bat_guess = 1.84 * P_pro_to ** 0.36  # [V] battery voltage estimation
        Kv = k_speed_mot * W_mot_to / U_bat_guess  # [rad/V/s]

        outputs["data:propulsion:motor:speed:constant:estimated"] = Kv

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        N_red = inputs["data:propulsion:gearbox:N_red"]
        P_pro_to = inputs["data:propulsion:propeller:power:takeoff"]
        W_pro_to = inputs["data:propulsion:propeller:speed:takeoff"]
        k_speed_mot = inputs["data:propulsion:motor:speed:k"]

        U_bat_guess = 1.84 * P_pro_to ** 0.36  # [V] battery voltage estimation

        partials[
            "data:propulsion:motor:speed:constant:estimated", "data:propulsion:gearbox:N_red"
        ] = k_speed_mot * W_pro_to / U_bat_guess

        partials[
            "data:propulsion:motor:speed:constant:estimated",
            "data:propulsion:propeller:power:takeoff",
        ] = - 0.36 * k_speed_mot * W_pro_to * N_red * P_pro_to ** (-1.36) / 1.84

        partials[
            "data:propulsion:motor:speed:constant:estimated",
            "data:propulsion:propeller:speed:takeoff",
        ] = k_speed_mot / U_bat_guess

        partials[
            "data:propulsion:motor:speed:constant:estimated", "data:propulsion:motor:speed:k"
        ] = W_pro_to * N_red / U_bat_guess