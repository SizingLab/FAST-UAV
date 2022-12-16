"""
Battery constraints
"""
import openmdao.api as om
import numpy as np


class BatteryConstraints(om.ExplicitComponent):
    """
    Constraints definition of the Battery component
    """

    def setup(self):
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage:tol", val=0.0, units="percent")
        self.add_input("data:propulsion:battery:power", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:voltage:climb", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:voltage:cruise", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:motor:power:cruise", val=np.nan, units="W")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:takeoff", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:climb", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:cruise", units=None)
        self.add_output("data:propulsion:battery:constraints:power:takeoff", units=None)
        self.add_output("data:propulsion:battery:constraints:power:climb", units=None)
        self.add_output("data:propulsion:battery:constraints:power:cruise", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:min", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:max", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        U_bat = inputs["data:propulsion:battery:voltage"]
        P_bat = inputs["data:propulsion:battery:power"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        U_mot_cl = inputs["data:propulsion:motor:voltage:climb"]
        U_mot_cr = inputs["data:propulsion:motor:voltage:cruise"]
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        P_mot_cl = inputs["data:propulsion:motor:power:climb"]
        P_mot_cr = inputs["data:propulsion:motor:power:cruise"]
        Npro = inputs["data:propulsion:propeller:number"]
        eta_ESC = inputs[
            "data:propulsion:esc:efficiency:estimated"
        ]  # TODO: replace by 'real' efficiency (ESC catalogue output, but be careful to algebraic loops...)
        k = 1 + inputs["data:propulsion:battery:voltage:tol"] / 100  # tolerance multiplier on prediction intervals

        # Battery voltage versus operating conditions
        battery_con1 = (U_bat - U_mot_to) / U_bat
        battery_con2 = (U_bat - U_mot_cl) / U_bat
        battery_con3 = (U_bat - U_mot_cr) / U_bat

        # Battery power versus operating conditions
        battery_con4 = (P_bat - P_mot_to * Npro / eta_ESC) / P_bat
        battery_con5 = (P_bat - P_mot_cl * Npro / eta_ESC) / P_bat
        battery_con6 = (P_bat - P_mot_cr * Npro / eta_ESC) / P_bat

        # Voltage versus power : tolerance intervals
        U_hat = 0.61 * P_bat ** 0.40  # [V] battery voltage-to-power regression
        eps_low = -10.98  # 1st percentile on regression error (i.e., 99% of data are above this value)
        eps_up = 16.24  # 99th percentile on regression error (i.e., 99% of data are below this value)
        U_min = U_hat + k * eps_low  # [V] minimum allowable voltage rating
        U_max = U_hat + k * eps_up  # [V] maximum allowable voltage rating
        battery_con7 = (U_bat - U_min) / U_bat
        battery_con8 = (U_max - U_bat) / U_bat

        outputs["data:propulsion:battery:constraints:voltage:takeoff"] = battery_con1
        outputs["data:propulsion:battery:constraints:voltage:climb"] = battery_con2
        outputs["data:propulsion:battery:constraints:voltage:cruise"] = battery_con3
        outputs["data:propulsion:battery:constraints:power:takeoff"] = battery_con4
        outputs["data:propulsion:battery:constraints:power:climb"] = battery_con5
        outputs["data:propulsion:battery:constraints:power:cruise"] = battery_con6
        outputs["data:propulsion:battery:constraints:voltage:min"] = battery_con7
        outputs["data:propulsion:battery:constraints:voltage:max"] = battery_con8

    def compute_partials(self, inputs, J, discrete_inputs=None):
        U_bat = inputs["data:propulsion:battery:voltage"]
        P_bat = inputs["data:propulsion:battery:power"]
        U_mot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        U_mot_cl = inputs["data:propulsion:motor:voltage:climb"]
        U_mot_cr = inputs["data:propulsion:motor:voltage:cruise"]
        P_mot_to = inputs["data:propulsion:motor:power:takeoff"]
        P_mot_cl = inputs["data:propulsion:motor:power:climb"]
        P_mot_cr = inputs["data:propulsion:motor:power:cruise"]
        Npro = inputs["data:propulsion:propeller:number"]
        eta_ESC = inputs["data:propulsion:esc:efficiency:estimated"]
        k = 1 + inputs["data:propulsion:battery:voltage:tol"] / 100

        U_hat = 0.61 * P_bat ** 0.40  # [V] battery voltage-to-power regression
        eps_low = -10.98  # 1st percentile on regression error (i.e., 99% of data are above this value)
        eps_up = 16.24  # 99th percentile on regression error (i.e., 99% of data are below this value)
        U_min = U_hat + k * eps_low  # [V] minimum allowable voltage rating
        U_max = U_hat + k * eps_up  # [V] maximum allowable voltage rating

        # Takeoff voltage
        J[
            "data:propulsion:battery:constraints:voltage:takeoff",
            "data:propulsion:battery:voltage"
        ] = (U_mot_to / U_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:takeoff",
            "data:propulsion:motor:voltage:takeoff",
        ] = (
            -1 / U_bat
        )

        # Climb voltage
        J[
            "data:propulsion:battery:constraints:voltage:climb",
            "data:propulsion:battery:voltage"
        ] = (U_mot_cl / U_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:climb",
            "data:propulsion:motor:voltage:climb",
        ] = (
            -1 / U_bat
        )

        # Cruise voltage
        J[
            "data:propulsion:battery:constraints:voltage:cruise",
            "data:propulsion:battery:voltage"
        ] = (U_mot_cr / U_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:cruise",
            "data:propulsion:motor:voltage:cruise",
        ] = (
            -1 / U_bat
        )

        # Takeoff power
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:battery:power",
        ] = (P_mot_to * Npro / eta_ESC) / (P_bat**2)
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:motor:power:takeoff",
        ] = (
            - Npro / eta_ESC / P_bat
        )
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:propeller:number"
        ] = - P_mot_to / eta_ESC / P_bat
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
             P_mot_to * Npro / (eta_ESC**2) / P_bat
        )

        # Climb power
        J[
            "data:propulsion:battery:constraints:power:climb",
            "data:propulsion:battery:power",
        ] = (P_mot_cl * Npro / eta_ESC) / (P_bat ** 2)
        J[
            "data:propulsion:battery:constraints:power:climb",
            "data:propulsion:motor:power:climb",
        ] = (
                - Npro / eta_ESC / P_bat
        )
        J[
            "data:propulsion:battery:constraints:power:climb",
            "data:propulsion:propeller:number"
        ] = - P_mot_cl / eta_ESC / P_bat
        J[
            "data:propulsion:battery:constraints:power:climb",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
                P_mot_cl * Npro / (eta_ESC ** 2) / P_bat
        )

        # Cruise power
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:battery:power",
        ] = (P_mot_cr * Npro / eta_ESC) / (P_bat ** 2)
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:motor:power:cruise",
        ] = (
                - Npro / eta_ESC / P_bat
        )
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:propeller:number"
        ] = - P_mot_cr / eta_ESC / P_bat
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
                P_mot_cr * Npro / (eta_ESC ** 2) / P_bat
        )

        # Voltage tolerance intervals
        J[
            "data:propulsion:battery:constraints:voltage:min",
            "data:propulsion:battery:voltage"] = U_min / U_bat ** 2
        J[
            "data:propulsion:battery:constraints:voltage:min",
            "data:propulsion:battery:power"] = - 0.40 * P_bat ** (-1) * U_hat / U_bat
        J[
            "data:propulsion:battery:constraints:voltage:min",
            "data:propulsion:battery:voltage:tol"] = - eps_low / U_bat / 100

        J[
            "data:propulsion:battery:constraints:voltage:max",
            "data:propulsion:battery:voltage"] = - U_max / U_bat ** 2
        J[
            "data:propulsion:battery:constraints:voltage:max",
            "data:propulsion:battery:power"] = 0.40 * P_bat ** (-1) * U_hat / U_bat
        J[
            "data:propulsion:battery:constraints:voltage:max",
            "data:propulsion:battery:voltage:tol"] = eps_up / U_bat / 100
