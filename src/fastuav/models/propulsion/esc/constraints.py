"""
ESC constraints
"""
import openmdao.api as om
import numpy as np


class ESCConstraints(om.ExplicitComponent):
    """
    Constraints definition of the ESC component
    """

    def setup(self):
        self.add_input("data:propulsion:esc:power:max", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:cruise", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("models:propulsion:esc:voltage:tol", val=0.0, units="percent")
        self.add_output("optimization:constraints:propulsion:esc:power:takeoff", units=None)
        self.add_output("optimization:constraints:propulsion:esc:power:climb", units=None)
        self.add_output("optimization:constraints:propulsion:esc:power:cruise", units=None)
        self.add_output("optimization:constraints:propulsion:esc:voltage:battery", units=None)
        self.add_output("optimization:constraints:propulsion:esc:voltage:min", units=None)
        self.add_output("optimization:constraints:propulsion:esc:voltage:max", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        P_esc = inputs["data:propulsion:esc:power:max"]
        P_esc_to = inputs["data:propulsion:esc:power:takeoff"]
        P_esc_cl = inputs["data:propulsion:esc:power:climb"]
        P_esc_cr = inputs["data:propulsion:esc:power:cruise"]
        U_esc = inputs["data:propulsion:esc:voltage"]
        U_bat = inputs["data:propulsion:battery:voltage"]
        k = 1 + inputs["models:propulsion:esc:voltage:tol"] / 100  # tolerance multiplier on prediction intervals

        # ESC power versus operating conditions
        ESC_con0 = (P_esc - P_esc_to) / P_esc
        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (P_esc - P_esc_cr) / P_esc

        # ESC voltage versus battery voltage
        ESC_con3 = (U_esc - U_bat) / U_esc

        # Voltage versus power : tolerance intervals
        U_hat = 1.84 * P_esc ** 0.36  # [V] ESC voltage-to-power regression
        eps_low = - 13.33  # 1st percentile on regression error (i.e., 99% of data are above this value)
        eps_up = 12.78  # 99th percentile on regression error (i.e., 99% of data are below this value)
        U_min = U_hat + k * eps_low  # [V] minimum allowable voltage rating
        U_max = U_hat + k * eps_up   # [V] maximum allowable voltage rating
        ESC_con4 = (U_esc - U_min) / U_esc
        ESC_con5 = (U_max - U_esc) / U_esc

        outputs["optimization:constraints:propulsion:esc:power:takeoff"] = ESC_con0
        outputs["optimization:constraints:propulsion:esc:power:climb"] = ESC_con1
        outputs["optimization:constraints:propulsion:esc:power:cruise"] = ESC_con2
        outputs["optimization:constraints:propulsion:esc:voltage:battery"] = ESC_con3
        outputs["optimization:constraints:propulsion:esc:voltage:min"] = ESC_con4
        outputs["optimization:constraints:propulsion:esc:voltage:max"] = ESC_con5

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        P_esc = inputs["data:propulsion:esc:power:max"]
        P_esc_to = inputs["data:propulsion:esc:power:takeoff"]
        P_esc_cl = inputs["data:propulsion:esc:power:climb"]
        P_esc_cr = inputs["data:propulsion:esc:power:cruise"]
        U_esc = inputs["data:propulsion:esc:voltage"]
        U_bat = inputs["data:propulsion:battery:voltage"]
        k = 1 + inputs["models:propulsion:esc:voltage:tol"] / 100  # tolerance multiplier on prediction interval

        U_hat = 1.84 * P_esc ** 0.36  # [V] ESC voltage-to-power regression
        eps_low = - 13.33  # 1st percentile on regression error (i.e., 99% of data are above this value)
        eps_up = 12.78  # 99th percentile on regression error (i.e., 99% of data are below this value)
        U_min = U_hat + k * eps_low  # [V] minimum allowable voltage rating
        U_max = U_hat + k * eps_up  # [V] maximum allowable voltage rating

        # Takeoff power
        partials[
            "optimization:constraints:propulsion:esc:power:takeoff",
            "data:propulsion:esc:power:max"
        ] = (
                P_esc_to / P_esc ** 2
        )
        partials[
            "optimization:constraints:propulsion:esc:power:takeoff",
            "data:propulsion:esc:power:takeoff"
        ] = (
                - 1.0 / P_esc
        )

        # Climb power
        partials[
            "optimization:constraints:propulsion:esc:power:climb",
            "data:propulsion:esc:power:max"
        ] = (
            P_esc_cl / P_esc**2
        )
        partials[
            "optimization:constraints:propulsion:esc:power:climb",
            "data:propulsion:esc:power:climb"
        ] = (
            -1.0 / P_esc
        )

        # Cruise power
        partials[
            "optimization:constraints:propulsion:esc:power:cruise",
            "data:propulsion:esc:power:max"
        ] = (
            P_esc_cr / P_esc**2
        )
        partials[
            "optimization:constraints:propulsion:esc:power:cruise",
            "data:propulsion:esc:power:cruise"
        ] = (
            - 1.0 / P_esc
        )

        # Battery voltage
        partials["optimization:constraints:propulsion:esc:voltage:battery",
                 "data:propulsion:battery:voltage"] = - 1.0 / U_esc
        partials["optimization:constraints:propulsion:esc:voltage:battery",
                 "data:propulsion:esc:voltage"] = U_bat / U_esc**2

        # Tolerance intervals
        partials["optimization:constraints:propulsion:esc:voltage:min",
                 "data:propulsion:esc:voltage"] = U_min / U_esc**2
        partials["optimization:constraints:propulsion:esc:voltage:min",
                 "data:propulsion:esc:power:max"] = - 0.36 * P_esc ** (-1) * U_hat / U_esc
        partials["optimization:constraints:propulsion:esc:voltage:min",
                 "models:propulsion:esc:voltage:tol"] = - eps_low / U_esc / 100

        partials["optimization:constraints:propulsion:esc:voltage:max",
                 "data:propulsion:esc:voltage"] = - U_max / U_esc ** 2
        partials["optimization:constraints:propulsion:esc:voltage:max",
                 "data:propulsion:esc:power:max"] = 0.36 * P_esc ** (-1) * U_hat / U_esc
        partials["optimization:constraints:propulsion:esc:voltage:max",
                 "models:propulsion:esc:voltage:tol"] = eps_up / U_esc / 100

