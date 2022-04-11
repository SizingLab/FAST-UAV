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
        self.add_input("data:propulsion:esc:voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:battery:voltage", val=np.nan, units="V")
        self.add_input("data:propulsion:esc:power:max", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:takeoff", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:climb", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:cruise", val=np.nan, units="W")
        self.add_output("data:propulsion:esc:constraints:power:takeoff", units=None)
        self.add_output("data:propulsion:esc:constraints:power:climb", units=None)
        self.add_output("data:propulsion:esc:constraints:power:cruise", units=None)
        self.add_output("data:propulsion:esc:constraints:voltage", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        P_esc = inputs["data:propulsion:esc:power:max"]
        V_esc = inputs["data:propulsion:esc:voltage"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_esc_to = inputs["data:propulsion:esc:power:takeoff"]
        P_esc_cl = inputs["data:propulsion:esc:power:climb"]
        P_esc_cr = inputs["data:propulsion:esc:power:cruise"]

        ESC_con0 = (P_esc - P_esc_to) / P_esc
        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (P_esc - P_esc_cr) / P_esc
        ESC_con3 = (V_esc - V_bat) / V_esc

        outputs["data:propulsion:esc:constraints:power:takeoff"] = ESC_con0
        outputs["data:propulsion:esc:constraints:power:climb"] = ESC_con1
        outputs["data:propulsion:esc:constraints:power:cruise"] = ESC_con2
        outputs["data:propulsion:esc:constraints:voltage"] = ESC_con3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        P_esc = inputs["data:propulsion:esc:power:max"]
        V_esc = inputs["data:propulsion:esc:voltage"]
        V_bat = inputs["data:propulsion:battery:voltage"]
        P_esc_cl = inputs["data:propulsion:esc:power:climb"]
        P_esc_cr = inputs["data:propulsion:esc:power:cruise"]

        partials[
            "data:propulsion:esc:constraints:power:climb",
            "data:propulsion:esc:power:max",
        ] = (
            P_esc_cl / P_esc**2
        )
        partials[
            "data:propulsion:esc:constraints:power:climb",
            "data:propulsion:esc:power:climb",
        ] = (
            -1.0 / P_esc
        )

        partials[
            "data:propulsion:esc:constraints:power:cruise",
            "data:propulsion:esc:power:max",
        ] = (
            P_esc_cr / P_esc**2
        )
        partials[
            "data:propulsion:esc:constraints:power:cruise",
            "data:propulsion:esc:power:cruise",
        ] = (
            -1.0 / P_esc
        )

        partials["data:propulsion:esc:constraints:voltage", "data:propulsion:battery:voltage",] = (
            -1.0 / V_esc
        )
        partials["data:propulsion:esc:constraints:voltage", "data:propulsion:esc:voltage",] = (
            V_bat / V_esc**2
        )
