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
        self.add_input("data:propulsion:battery:current:max", val=np.nan, units="A")
        self.add_input("data:propulsion:motor:voltage:takeoff", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:voltage:climb", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:voltage:cruise", val=np.nan, units="V")
        self.add_input("data:propulsion:motor:current:takeoff", val=np.nan, units="A")
        self.add_input("data:propulsion:motor:current:climb", val=np.nan, units="A")
        self.add_input("data:propulsion:motor:current:cruise", val=np.nan, units="A")
        self.add_input("data:propulsion:propeller:number", val=np.nan, units=None)
        self.add_input("data:propulsion:esc:efficiency:estimated", val=np.nan, units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:takeoff", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:climb", units=None)
        self.add_output("data:propulsion:battery:constraints:voltage:cruise", units=None)
        self.add_output("data:propulsion:battery:constraints:power:takeoff", units=None)
        self.add_output("data:propulsion:battery:constraints:power:climb", units=None)
        self.add_output("data:propulsion:battery:constraints:power:cruise", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        V_bat = inputs["data:propulsion:battery:voltage"]
        Imax = inputs["data:propulsion:battery:current:max"]
        Umot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        Umot_cl = inputs["data:propulsion:motor:voltage:climb"]
        Umot_cr = inputs["data:propulsion:motor:voltage:cruise"]
        Imot_to = inputs["data:propulsion:motor:current:takeoff"]
        Imot_cl = inputs["data:propulsion:motor:current:climb"]
        Imot_cr = inputs["data:propulsion:motor:current:cruise"]
        Npro = inputs["data:propulsion:propeller:number"]
        eta_ESC = inputs[
            "data:propulsion:esc:efficiency:estimated"
        ]  # TODO: replace by 'real' efficiency (ESC catalogue output, but be careful to algebraic loops...)

        battery_con1 = (V_bat - Umot_to) / V_bat
        battery_con2 = (V_bat - Umot_cl) / V_bat
        battery_con3 = (V_bat - Umot_cr) / V_bat
        battery_con4 = (V_bat * Imax - Umot_to * Imot_to * Npro / eta_ESC) / (V_bat * Imax)
        battery_con5 = (V_bat * Imax - Umot_cl * Imot_cl * Npro / eta_ESC) / (V_bat * Imax)
        battery_con6 = (V_bat * Imax - Umot_cr * Imot_cr * Npro / eta_ESC) / (V_bat * Imax)

        outputs["data:propulsion:battery:constraints:voltage:takeoff"] = battery_con1
        outputs["data:propulsion:battery:constraints:voltage:climb"] = battery_con2
        outputs["data:propulsion:battery:constraints:voltage:cruise"] = battery_con3
        outputs["data:propulsion:battery:constraints:power:takeoff"] = battery_con4
        outputs["data:propulsion:battery:constraints:power:climb"] = battery_con5
        outputs["data:propulsion:battery:constraints:power:cruise"] = battery_con6

    def compute_partials(self, inputs, J, discrete_inputs=None):
        V_bat = inputs["data:propulsion:battery:voltage"]
        Umot_to = inputs["data:propulsion:motor:voltage:takeoff"]
        Umot_cl = inputs["data:propulsion:motor:voltage:climb"]
        Umot_cr = inputs["data:propulsion:motor:voltage:cruise"]
        Imax = inputs["data:propulsion:battery:current:max"]
        Imot_to = inputs["data:propulsion:motor:current:takeoff"]
        Npro = inputs["data:propulsion:propeller:number"]
        eta_ESC = inputs["data:propulsion:esc:efficiency:estimated"]
        Imot_cl = inputs["data:propulsion:motor:current:climb"]
        Imot_cr = inputs["data:propulsion:motor:current:cruise"]

        J[
            "data:propulsion:battery:constraints:voltage:takeoff", "data:propulsion:battery:voltage"
        ] = (Umot_to / V_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:takeoff",
            "data:propulsion:motor:voltage:takeoff",
        ] = (
            -1 / V_bat
        )

        J[
            "data:propulsion:battery:constraints:voltage:climb", "data:propulsion:battery:voltage"
        ] = (1 / V_bat - (-Umot_cl + V_bat) / V_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:climb",
            "data:propulsion:motor:voltage:climb",
        ] = (
            -1 / V_bat
        )

        J[
            "data:propulsion:battery:constraints:voltage:cruise", "data:propulsion:battery:voltage"
        ] = (1 / V_bat - (-Umot_cr + V_bat) / V_bat**2)
        J[
            "data:propulsion:battery:constraints:voltage:cruise",
            "data:propulsion:motor:voltage:cruise",
        ] = (
            -1 / V_bat
        )

        J[
            "data:propulsion:battery:constraints:power:takeoff", "data:propulsion:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_to * Npro * Umot_to / eta_ESC) / (Imax * V_bat**2)
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:battery:current:max",
        ] = 1 / Imax - (Imax * V_bat - Imot_to * Npro * Umot_to / eta_ESC) / (Imax**2 * V_bat)
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:motor:voltage:takeoff",
        ] = (
            -Imot_to * Npro / (Imax * V_bat * eta_ESC)
        )
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:motor:current:takeoff",
        ] = (
            -Npro * Umot_to / (Imax * V_bat * eta_ESC)
        )
        J[
            "data:propulsion:battery:constraints:power:takeoff", "data:propulsion:propeller:number"
        ] = (-Imot_to * Umot_to / (Imax * V_bat * eta_ESC))
        J[
            "data:propulsion:battery:constraints:power:takeoff",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
            Imot_to * Npro * Umot_to / (Imax * V_bat * eta_ESC**2)
        )

        J[
            "data:propulsion:battery:constraints:power:climb", "data:propulsion:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_cl * Npro * Umot_cl / eta_ESC) / (Imax * V_bat**2)
        J[
            "data:propulsion:battery:constraints:power:climb", "data:propulsion:battery:current:max"
        ] = 1 / Imax - (Imax * V_bat - Imot_cl * Npro * Umot_cl / eta_ESC) / (Imax**2 * V_bat)
        J[
            "data:propulsion:battery:constraints:power:climb", "data:propulsion:motor:voltage:climb"
        ] = (-Imot_cl * Npro / (Imax * V_bat * eta_ESC))
        J[
            "data:propulsion:battery:constraints:power:climb", "data:propulsion:motor:current:climb"
        ] = (-Npro * Umot_cl / (Imax * V_bat * eta_ESC))
        J["data:propulsion:battery:constraints:power:climb", "data:propulsion:propeller:number"] = (
            -Imot_cl * Umot_cl / (Imax * V_bat * eta_ESC)
        )
        J[
            "data:propulsion:battery:constraints:power:climb",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
            Imot_cl * Npro * Umot_cl / (Imax * V_bat * eta_ESC**2)
        )

        J[
            "data:propulsion:battery:constraints:power:cruise", "data:propulsion:battery:voltage"
        ] = 1 / V_bat - (Imax * V_bat - Imot_cr * Npro * Umot_cr / eta_ESC) / (Imax * V_bat**2)
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:battery:current:max",
        ] = 1 / Imax - (Imax * V_bat - Imot_cr * Npro * Umot_cr / eta_ESC) / (Imax**2 * V_bat)
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:motor:voltage:cruise",
        ] = (
            -Imot_cr * Npro / (Imax * V_bat * eta_ESC)
        )
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:motor:current:cruise",
        ] = (
            -Npro * Umot_cr / (Imax * V_bat * eta_ESC)
        )
        J[
            "data:propulsion:battery:constraints:power:cruise", "data:propulsion:propeller:number"
        ] = (-Imot_cr * Umot_cr / (Imax * V_bat * eta_ESC))
        J[
            "data:propulsion:battery:constraints:power:cruise",
            "data:propulsion:esc:efficiency:estimated",
        ] = (
            Imot_cr * Npro * Umot_cr / (Imax * V_bat * eta_ESC**2)
        )
