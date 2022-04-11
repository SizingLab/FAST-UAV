"""
Neutral Point module.
The neutral point (NP) is the position of center of mass where the UAV would be on the edge of stability.
"""
import openmdao.api as om
import numpy as np


class NeutralPoint(om.ExplicitComponent):
    """
    Computes the neutral of a fixed wing UAV.
    The fuselage contribution is neglected.
    Only the wing and the horizontal tail are taken into account.
    """

    def setup(self):
        self.add_input("data:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:coefficient", val=0.5, units=None)
        self.add_input("data:geometry:tail:horizontal:AR", val=4.0, units=None)
        self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_output("data:stability:neutral_point", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        AR_w = inputs["data:geometry:wing:AR"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        AR_ht = inputs["data:geometry:tail:horizontal:AR"]
        e = inputs["data:aerodynamics:CDi:e"]
        x_ac_w = inputs[
            "data:geometry:wing:MAC:C4:x"
        ]  # wing aerodynamic center (located at quarter chord of the wing)

        l_np = (
            c_MAC * V_ht * (1 - 4 / (2 + e * AR_w)) * (1 + 2 / e / AR_w) / (1 + 2 / AR_ht)
        )  # distance from neutral point to wing aerodynamic center [m]
        x_np = x_ac_w + l_np  # distance from neutral point to nose tip [m]

        outputs["data:stability:neutral_point"] = x_np
