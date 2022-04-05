"""
Propeller constraints
"""
import openmdao.api as om
import numpy as np


class PropellerConstraints(om.ExplicitComponent):
    """
    Constraints definition of the propeller component
    """

    def setup(self):
        self.add_input("data:propeller:geometry:diameter", val=np.nan, units="m")
        self.add_input("data:propeller:reference:ND:max", val=np.nan, units="m/s")
        self.add_input("data:propeller:advance_ratio:climb", val=np.nan, units=None)
        self.add_input("data:propeller:advance_ratio:cruise", val=np.nan, units=None)
        self.add_input("data:propeller:speed:climb", val=np.nan, units="rad/s")
        self.add_input("data:propeller:speed:cruise", val=np.nan, units="rad/s")
        self.add_input("mission:design_mission:climb:speed", val=np.nan, units="m/s")
        self.add_input("mission:design_mission:cruise:speed", val=np.nan, units="m/s")
        self.add_output("data:propeller:constraints:speed:climb", units=None)
        self.add_output("data:propeller:constraints:speed:cruise", units=None)
        self.add_output("mission:design_mission:constraints:speed:climb", units=None)
        self.add_output("mission:design_mission:constraints:speed:cruise", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Dpro = inputs["data:propeller:geometry:diameter"]
        NDmax = inputs["data:propeller:reference:ND:max"]
        J_climb = inputs["data:propeller:advance_ratio:climb"]
        J_cruise = inputs["data:propeller:advance_ratio:cruise"]
        W_pro_cl = inputs["data:propeller:speed:climb"]
        W_pro_cr = inputs["data:propeller:speed:cruise"]
        V_cl = inputs["mission:design_mission:climb:speed"]
        V_cr = inputs["mission:design_mission:cruise:speed"]

        prop_con1 = (NDmax - W_pro_cl * Dpro / 2 / np.pi) / NDmax
        prop_con2 = (NDmax - W_pro_cr * Dpro / 2 / np.pi) / NDmax
        prop_con3 = (V_cl - J_climb * W_pro_cl * Dpro / 2 / np.pi) / V_cl
        prop_con4 = (V_cr - J_cruise * W_pro_cr * Dpro / 2 / np.pi) / V_cr

        outputs["data:propeller:constraints:speed:climb"] = prop_con1
        outputs["data:propeller:constraints:speed:cruise"] = prop_con2
        outputs["mission:design_mission:constraints:speed:climb"] = prop_con3
        outputs["mission:design_mission:constraints:speed:cruise"] = prop_con4

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Dpro = inputs["data:propeller:geometry:diameter"]
        NDmax = inputs["data:propeller:reference:ND:max"]
        J_climb = inputs["data:propeller:advance_ratio:climb"]
        J_cruise = inputs["data:propeller:advance_ratio:cruise"]
        W_pro_cl = inputs["data:propeller:speed:climb"]
        W_pro_cr = inputs["data:propeller:speed:cruise"]
        V_cl = inputs["mission:design_mission:climb:speed"]
        V_cr = inputs["mission:design_mission:cruise:speed"]

        partials[
            "data:propeller:constraints:speed:climb",
            "data:propeller:speed:climb",
        ] = (
            -Dpro / NDmax / 2 / np.pi
        )
        partials[
            "data:propeller:constraints:speed:climb",
            "data:propeller:reference:ND:max",
        ] = (
            W_pro_cl * Dpro / NDmax**2 / 2 / np.pi
        )
        partials[
            "data:propeller:constraints:speed:climb",
            "data:propeller:geometry:diameter",
        ] = (
            -W_pro_cl / NDmax / 2 / np.pi
        )

        partials[
            "data:propeller:constraints:speed:cruise",
            "data:propeller:speed:cruise",
        ] = (
            -Dpro / NDmax / 2 / np.pi
        )
        partials[
            "data:propeller:constraints:speed:cruise",
            "data:propeller:reference:ND:max",
        ] = (
            W_pro_cr * Dpro / NDmax**2 / 2 / np.pi
        )
        partials[
            "data:propeller:constraints:speed:cruise",
            "data:propeller:geometry:diameter",
        ] = (
            -W_pro_cr / NDmax / 2 / np.pi
        )

        partials["mission:design_mission:constraints:speed:climb", "mission:design_mission:climb:speed",] = (
            J_climb * W_pro_cl * Dpro / V_cl**2 / 2 / np.pi
        )
        partials[
            "mission:design_mission:constraints:speed:climb",
            "data:propeller:advance_ratio:climb",
        ] = (
            -W_pro_cl * Dpro / V_cl / 2 / np.pi
        )
        partials["mission:design_mission:constraints:speed:climb", "data:propeller:speed:climb",] = (
            -J_climb * Dpro / V_cl / 2 / np.pi
        )
        partials[
            "mission:design_mission:constraints:speed:climb",
            "data:propeller:geometry:diameter",
        ] = (
            -J_climb * W_pro_cl / V_cl / 2 / np.pi
        )

        partials[
            "mission:design_mission:constraints:speed:cruise",
            "mission:design_mission:cruise:speed",
        ] = (
            J_cruise * W_pro_cr * Dpro / V_cr**2 / 2 / np.pi
        )
        partials[
            "mission:design_mission:constraints:speed:cruise",
            "data:propeller:advance_ratio:cruise",
        ] = (
            -W_pro_cr * Dpro / V_cr / 2 / np.pi
        )
        partials[
            "mission:design_mission:constraints:speed:cruise",
            "data:propeller:speed:cruise",
        ] = (
            -J_cruise * Dpro / V_cr / 2 / np.pi
        )
        partials[
            "mission:design_mission:constraints:speed:cruise",
            "data:propeller:geometry:diameter",
        ] = (
            -J_cruise * W_pro_cr / V_cr / 2 / np.pi
        )
