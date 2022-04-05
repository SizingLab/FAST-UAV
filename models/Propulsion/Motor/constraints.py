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
        self.add_input("data:motor:torque:nominal", val=np.nan, units="N*m")
        self.add_input("data:motor:torque:max", val=np.nan, units="N*m")
        self.add_input("data:motor:torque:takeoff", val=np.nan, units="N*m")
        self.add_input("data:motor:torque:hover", val=np.nan, units="N*m")
        self.add_input("data:motor:torque:climb", val=np.nan, units="N*m")
        self.add_input("data:motor:torque:cruise", val=np.nan, units="N*m")
        self.add_output("data:motor:constraints:torque:takeoff", units=None)
        self.add_output("data:motor:constraints:torque:climb", units=None)
        self.add_output("data:motor:constraints:torque:hover", units=None)
        self.add_output("data:motor:constraints:torque:cruise", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_max = inputs["data:motor:torque:max"]
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["data:motor:torque:hover"]
        Tmot_to = inputs["data:motor:torque:takeoff"]
        Tmot_cl = inputs["data:motor:torque:climb"]
        Tmot_cr = inputs["data:motor:torque:cruise"]

        motor_con1 = (Tmot_max - Tmot_to) / Tmot_max  # transient torque
        motor_con2 = (Tmot_max - Tmot_cl) / Tmot_max  # transient torque
        motor_con3 = (Tmot_nom - Tmot_hov) / Tmot_nom  # steady torque
        motor_con4 = (Tmot_nom - Tmot_cr) / Tmot_nom  # steady torque

        outputs["data:motor:constraints:torque:takeoff"] = motor_con1
        outputs["data:motor:constraints:torque:climb"] = motor_con2
        outputs["data:motor:constraints:torque:hover"] = motor_con3
        outputs["data:motor:constraints:torque:cruise"] = motor_con4

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_max = inputs["data:motor:torque:max"]
        Tmot_nom = inputs["data:motor:torque:nominal"]
        Tmot_hov = inputs["data:motor:torque:hover"]
        Tmot_to = inputs["data:motor:torque:takeoff"]
        Tmot_cl = inputs["data:motor:torque:climb"]
        Tmot_cr = inputs["data:motor:torque:cruise"]

        partials["data:motor:constraints:torque:takeoff", "data:motor:torque:max",] = (
            Tmot_to / Tmot_max**2
        )
        partials[
            "data:motor:constraints:torque:takeoff",
            "data:motor:torque:takeoff",
        ] = (
            -1.0 / Tmot_max
        )

        partials["data:motor:constraints:torque:climb", "data:motor:torque:max",] = (
            Tmot_cl / Tmot_max**2
        )
        partials["data:motor:constraints:torque:climb", "data:motor:torque:climb",] = (
            -1.0 / Tmot_max
        )

        partials[
            "data:motor:constraints:torque:hover",
            "data:motor:torque:nominal",
        ] = (
            Tmot_hov / Tmot_nom**2
        )
        partials[
            "data:motor:constraints:torque:hover",
            "data:motor:torque:cruise",
        ] = (
            -1.0 / Tmot_nom
        )

        partials[
            "data:motor:constraints:torque:cruise",
            "data:motor:torque:nominal",
        ] = (
            Tmot_cr / Tmot_nom**2
        )
        partials[
            "data:motor:constraints:torque:cruise",
            "data:motor:torque:cruise",
        ] = (
            -1.0 / Tmot_nom
        )
