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
        self.add_input("data:ESC:voltage", val=np.nan, units="V")
        self.add_input("data:battery:voltage", val=np.nan, units="V")
        self.add_input("data:ESC:power:max", val=np.nan, units="W")
        self.add_input("data:ESC:power:takeoff", val=np.nan, units="W")
        self.add_input("data:ESC:power:climb", val=np.nan, units="W")
        self.add_input("data:ESC:power:cruise", val=np.nan, units="W")
        self.add_output("data:ESC:constraints:power:takeoff", units=None)
        self.add_output("data:ESC:constraints:power:climb", units=None)
        self.add_output("data:ESC:constraints:power:cruise", units=None)
        self.add_output("data:ESC:constraints:voltage", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        P_esc = inputs["data:ESC:power:max"]
        V_esc = inputs["data:ESC:voltage"]
        V_bat = inputs["data:battery:voltage"]
        P_esc_to = inputs["data:ESC:power:takeoff"]
        P_esc_cl = inputs["data:ESC:power:climb"]
        P_esc_cr = inputs["data:ESC:power:cruise"]

        ESC_con0 = (P_esc - P_esc_to) / P_esc
        ESC_con1 = (P_esc - P_esc_cl) / P_esc
        ESC_con2 = (P_esc - P_esc_cr) / P_esc
        ESC_con3 = (V_esc - V_bat) / V_esc

        outputs["data:ESC:constraints:power:takeoff"] = ESC_con0
        outputs["data:ESC:constraints:power:climb"] = ESC_con1
        outputs["data:ESC:constraints:power:cruise"] = ESC_con2
        outputs["data:ESC:constraints:voltage"] = ESC_con3

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        P_esc = inputs["data:ESC:power:max"]
        V_esc = inputs["data:ESC:voltage"]
        V_bat = inputs["data:battery:voltage"]
        P_esc_cl = inputs["data:ESC:power:climb"]
        P_esc_cr = inputs["data:ESC:power:cruise"]

        partials["data:ESC:constraints:power:climb", "data:ESC:power:max",] = (
            P_esc_cl / P_esc**2
        )
        partials["data:ESC:constraints:power:climb", "data:ESC:power:climb",] = (
            -1.0 / P_esc
        )

        partials["data:ESC:constraints:power:cruise", "data:ESC:power:max",] = (
            P_esc_cr / P_esc**2
        )
        partials["data:ESC:constraints:power:cruise", "data:ESC:power:cruise",] = (
            -1.0 / P_esc
        )

        partials["data:ESC:constraints:voltage", "data:battery:voltage",] = (
            -1.0 / V_esc
        )
        partials["data:ESC:constraints:voltage", "data:ESC:voltage",] = (
            V_bat / V_esc**2
        )
