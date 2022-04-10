"""
Cables radius
"""

import openmdao.api as om
import numpy as np


class Radius(om.ExplicitComponent):
    """
    Computes cables radius.
    """

    def setup(self):
        self.add_input("data:propulsion:cables:radius:reference", val=np.nan, units="m")
        self.add_input("data:propulsion:cables:current:reference", val=np.nan, units="A")
        self.add_input("data:propulsion:motor:current:hover", val=np.nan, units="A")
        self.add_output("data:propulsion:cables:radius", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        r_ref = inputs["data:propulsion:cables:radius:reference"]  # [m] radius of reference cable
        I_ref = inputs[
            "data:propulsion:cables:current:reference"
        ]  # [A] max allowable current (continuous) of reference cable
        I = inputs["data:propulsion:motor:current:hover"]  # [A] max current (continuous)

        r = r_ref * (I / I_ref) ** (1 / 1.5)  # [m] radius of cable

        outputs["data:propulsion:cables:radius"] = r