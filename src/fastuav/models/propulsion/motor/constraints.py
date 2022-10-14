"""
Motor constraints
"""
import openmdao.api as om
import numpy as np


class MotorConstraints(om.ExplicitComponent):
    """
    Constraints definition of the motor component
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:max", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:takeoff", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:hover", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:climb", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:cruise", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:speed:constant", val=np.nan, units="rad/V/s")
        self.add_input("data:propulsion:motor:speed:constant:tol", val=0.0, units="percent")
        self.add_output("data:propulsion:motor:constraints:torque:takeoff", units=None)
        self.add_output("data:propulsion:motor:constraints:torque:climb", units=None)
        self.add_output("data:propulsion:motor:constraints:torque:hover", units=None)
        self.add_output("data:propulsion:motor:constraints:torque:cruise", units=None)
        self.add_output("data:propulsion:motor:constraints:speed:constant:min", units=None)
        self.add_output("data:propulsion:motor:constraints:speed:constant:max", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        T_mot_max = inputs["data:propulsion:motor:torque:max"]
        T_mot_nom = inputs["data:propulsion:motor:torque:nominal"]
        T_mot_hov = inputs["data:propulsion:motor:torque:hover"]
        T_mot_to = inputs["data:propulsion:motor:torque:takeoff"]
        T_mot_cl = inputs["data:propulsion:motor:torque:climb"]
        T_mot_cr = inputs["data:propulsion:motor:torque:cruise"]
        Kv = inputs["data:propulsion:motor:speed:constant"]
        k = 1 + inputs["data:propulsion:motor:speed:constant:tol"] / 100  # tolerance multiplier on prediction intervals

        # transient torque
        motor_con1 = (T_mot_max - T_mot_to) / T_mot_max  # transient torque
        motor_con2 = (T_mot_max - T_mot_cl) / T_mot_max  # transient torque

        # steady torque
        motor_con3 = (T_mot_nom - T_mot_hov) / T_mot_nom  # steady torque
        motor_con4 = (T_mot_nom - T_mot_cr) / T_mot_nom  # steady torque

        # Speed constant versus max torque : tolerance intervals
        Kv_hat = 51.52 * T_mot_max ** (-0.43)  # speed constant vs torque regression
        eps_low = -2.21  # 1st percentile on relative regression error (i.e., 99% of data are above this value)
        eps_up = 0.65  # 99th percentile on relative regression error (i.e., 99% of data are below this value)
        Kv_min = Kv_hat / (1 - k * eps_low)  # minimum bound for Kv
        Kv_max = Kv_hat / (1 - k * eps_up)  # maximum bound for Kv
        motor_con5 = (Kv - Kv_min) / Kv
        motor_con6 = (Kv_max - Kv) / Kv

        outputs["data:propulsion:motor:constraints:torque:takeoff"] = motor_con1
        outputs["data:propulsion:motor:constraints:torque:climb"] = motor_con2
        outputs["data:propulsion:motor:constraints:torque:hover"] = motor_con3
        outputs["data:propulsion:motor:constraints:torque:cruise"] = motor_con4
        outputs["data:propulsion:motor:constraints:speed:constant:min"] = motor_con5
        outputs["data:propulsion:motor:constraints:speed:constant:max"] = motor_con6

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        T_mot_max = inputs["data:propulsion:motor:torque:max"]
        T_mot_nom = inputs["data:propulsion:motor:torque:nominal"]
        T_mot_hov = inputs["data:propulsion:motor:torque:hover"]
        T_mot_to = inputs["data:propulsion:motor:torque:takeoff"]
        T_mot_cl = inputs["data:propulsion:motor:torque:climb"]
        T_mot_cr = inputs["data:propulsion:motor:torque:cruise"]
        Kv = inputs["data:propulsion:motor:speed:constant"]
        k = 1 + inputs["data:propulsion:motor:speed:constant:tol"] / 100

        Kv_hat = 51.52 * T_mot_max ** (-0.43)  # speed constant vs torque regression
        eps_low = -2.21  # 1st percentile on relative regression error (i.e., 99% of data are above this value)
        eps_up = 0.65  # 99th percentile on relative regression error (i.e., 99% of data are below this value)
        Kv_min = Kv_hat / (1 - k * eps_low)  # minimum bound for Kv
        Kv_max = Kv_hat / (1 - k * eps_up)  # maximum bound for Kv

        # Takeoff torque
        partials[
            "data:propulsion:motor:constraints:torque:takeoff",
            "data:propulsion:motor:torque:max",
        ] = (
            T_mot_to / T_mot_max**2
        )
        partials[
            "data:propulsion:motor:constraints:torque:takeoff",
            "data:propulsion:motor:torque:takeoff",
        ] = (
            -1.0 / T_mot_max
        )

        # Climb torque
        partials[
            "data:propulsion:motor:constraints:torque:climb",
            "data:propulsion:motor:torque:max",
        ] = (
            T_mot_cl / T_mot_max**2
        )
        partials[
            "data:propulsion:motor:constraints:torque:climb",
            "data:propulsion:motor:torque:climb",
        ] = (
            -1.0 / T_mot_max
        )

        # Hover torque
        partials[
            "data:propulsion:motor:constraints:torque:hover",
            "data:propulsion:motor:torque:nominal",
        ] = (
            T_mot_hov / T_mot_nom**2
        )
        partials[
            "data:propulsion:motor:constraints:torque:hover",
            "data:propulsion:motor:torque:hover",
        ] = (
            -1.0 / T_mot_nom
        )

        # Cruise torque
        partials[
            "data:propulsion:motor:constraints:torque:cruise",
            "data:propulsion:motor:torque:nominal",
        ] = (
            T_mot_cr / T_mot_nom**2
        )
        partials[
            "data:propulsion:motor:constraints:torque:cruise",
            "data:propulsion:motor:torque:cruise",
        ] = (
            -1.0 / T_mot_nom
        )

        # Tolerance intervals
        partials[
            "data:propulsion:motor:constraints:speed:constant:min",
            "data:propulsion:motor:speed:constant",
        ] = (
                Kv_min / Kv ** 2
        )

        partials[
            "data:propulsion:motor:constraints:speed:constant:min",
            "data:propulsion:motor:speed:constant:tol",
        ] = (
                - Kv_hat * eps_low / Kv / (1 - k * eps_low) ** 2 / 100
        )

        partials[
            "data:propulsion:motor:constraints:speed:constant:min",
            "data:propulsion:motor:torque:max",
        ] = (
                - (- 0.43) * Kv_min / Kv * T_mot_max ** (-1)
        )

        partials[
            "data:propulsion:motor:constraints:speed:constant:max",
            "data:propulsion:motor:speed:constant",
        ] = (
                - Kv_max / Kv ** 2
        )

        partials[
            "data:propulsion:motor:constraints:speed:constant:max",
            "data:propulsion:motor:speed:constant:tol",
        ] = (
                Kv_hat * eps_up / Kv / (1 - k * eps_up) ** 2 / 100
        )

        partials[
            "data:propulsion:motor:constraints:speed:constant:max",
            "data:propulsion:motor:torque:max",
        ] = (
                (- 0.43) * Kv_max / Kv * T_mot_max ** (-1)
        )
