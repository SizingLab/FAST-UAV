"""
Neutral Point module.
The neutral point (NP) is the position of center of mass where the UAV would be on the edge of stability.
"""

import numpy as np
import openmdao.api as om


class NeutralPoint(om.ExplicitComponent):
    """
    Computes the neutral of a fixed wing UAV.
    The fuselage contribution is neglected.
    Only the wing and the horizontal tail are taken into account.
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:coefficient", val=0.5, units=None)
        self.add_input("optimization:variables:geometry:tail:horizontal:AR", val=4.0, units=None)
        self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_output("data:stability:neutral_point", units="m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        AR_ht = inputs["optimization:variables:geometry:tail:horizontal:AR"]
        e = inputs["data:aerodynamics:CDi:e"]
        x_ac_w = inputs[
            "data:geometry:wing:MAC:C4:x"
        ]  # wing aerodynamic center (located at quarter chord of the wing)

        l_np = (
            c_MAC * V_ht * (1 - 4 / (2 + e * AR_w)) * (1 + 2 / e / AR_w) / (1 + 2 / AR_ht)
        )  # distance from neutral point to wing aerodynamic center [m]
        x_np = x_ac_w + l_np  # distance from neutral point to nose tip [m]

        outputs["data:stability:neutral_point"] = x_np


class NeutralPointVLM(om.ExplicitComponent):
    """
    Computes the neutral of a fixed wing UAV.
    The fuselage contribution is neglected.
    Only the wing and the horizontal tail are taken into account.
    """

    def setup(self):
        self.add_input("optimization:variables:geometry:wing:AR", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_input("data:geometry:tail:horizontal:coefficient", val=0.5, units=None)
        # self.add_input("optimization:variables:geometry:tail:horizontal:AR", val=4.0, units=None)
        # self.add_input("data:aerodynamics:CDi:e", val=np.nan, units=None)
        self.add_input("data:geometry:wing:MAC:C4:x", val=np.nan, units="m")
        self.add_input("data:aerodynamics:wing:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_input("data:aerodynamics:tail:horizontal:cruise:CL_alpha", val=np.nan, units="rad**-1")
        self.add_output("data:stability:neutral_point", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        cl_alpha_w = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl_alpha_ht = inputs["data:aerodynamics:tail:horizontal:cruise:CL_alpha"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        # AR_ht = inputs["optimization:variables:geometry:tail:horizontal:AR"]
        # e = inputs["data:aerodynamics:CDi:e"]
        x_ac_w = inputs[
            "data:geometry:wing:MAC:C4:x"
        ]  # wing aerodynamic center (located at quarter chord of the wing)

        downwash_gradient = 2 * cl_alpha_w / (np.pi * AR_w)
        # l_np = (
        #     c_MAC * V_ht * (1 - 4 / (2 + e * AR_w)) * (1 + 2 / e / AR_w) / (1 + 2 / AR_ht)
        # )  # distance from neutral point to wing aerodynamic center [m]

        l_np = (
            c_MAC * V_ht * cl_alpha_ht / cl_alpha_w * (1 - downwash_gradient)
        )  # distance from neutral point to wing aerodynamic center [m]

        x_np = x_ac_w + l_np  # distance from neutral point to nose tip [m]

        outputs["data:stability:neutral_point"] = x_np

    def compute_partials(self, inputs, partials):
        AR_w = inputs["optimization:variables:geometry:wing:AR"]
        cl_alpha_w = inputs["data:aerodynamics:wing:cruise:CL_alpha"]
        cl_alpha_ht = inputs["data:aerodynamics:tail:horizontal:cruise:CL_alpha"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]
        V_ht = inputs["data:geometry:tail:horizontal:coefficient"]
        x_ac_w = inputs["data:geometry:wing:MAC:C4:x"]

        # Computed intermediate values
        downwash_gradient = 2 * cl_alpha_w / (np.pi * AR_w)
        l_np = c_MAC * V_ht * (cl_alpha_ht / cl_alpha_w) * (1 - downwash_gradient)

        # ========== Partials ==========

        # d(x_np) / d(x_ac_w) = 1
        partials["data:stability:neutral_point", "data:geometry:wing:MAC:C4:x"] = 1.0

        # d(x_np) / d(c_MAC) = d(l_np) / d(c_MAC)
        #                    = V_ht * (cl_alpha_ht/cl_alpha_w) * (1 - downwash_gradient)
        #                    = l_np / c_MAC
        partials["data:stability:neutral_point", "data:geometry:wing:MAC:length"] = (
            l_np / c_MAC
        )

        # d(x_np) / d(V_ht) = d(l_np) / d(V_ht)
        #                   = c_MAC * (cl_alpha_ht/cl_alpha_w) * (1 - downwash_gradient)
        #                   = l_np / V_ht
        partials["data:stability:neutral_point", "data:geometry:tail:horizontal:coefficient"] = (
            l_np / V_ht
        )

        # d(x_np) / d(cl_alpha_ht) = d(l_np) / d(cl_alpha_ht)
        #                          = c_MAC * V_ht * (1/cl_alpha_w) * (1 - downwash_gradient)
        #                          = l_np / cl_alpha_ht
        partials["data:stability:neutral_point", "data:aerodynamics:tail:horizontal:cruise:CL_alpha"] = (
            l_np / cl_alpha_ht
        )

        # d(x_np) / d(cl_alpha_w) = d(l_np) / d(cl_alpha_w)
        # Using product rule on: l_np = c_MAC*V_ht*(cl_alpha_ht/cl_alpha_w)*(1-dw)
        # where dw = 2*cl_alpha_w/(pi*AR_w)
        # Result: -c_MAC*V_ht*cl_alpha_ht / cl_alpha_w^2
        partials["data:stability:neutral_point", "data:aerodynamics:wing:cruise:CL_alpha"] = (
            -c_MAC * V_ht * cl_alpha_ht / (cl_alpha_w ** 2)
        )

        # d(x_np) / d(AR_w) = d(l_np) / d(AR_w)
        # Using chain rule: d(l_np)/d(AR_w) = c_MAC*V_ht*(cl_alpha_ht/cl_alpha_w)*d(1-dw)/d(AR_w)
        # where d(1-dw)/d(AR_w) = -d(dw)/d(AR_w) = 2*cl_alpha_w/(pi*AR_w^2)
        # Result: 2*c_MAC*V_ht*cl_alpha_ht / (pi*AR_w^2)
        partials["data:stability:neutral_point", "optimization:variables:geometry:wing:AR"] = (
            2 * c_MAC * V_ht * cl_alpha_ht / (np.pi * AR_w ** 2)
        )
