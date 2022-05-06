"""
Static Margin module.
Static margin is defined as the distance between the center of gravity and the neutral point of the aircraft,
expressed as a percentage of the mean aerodynamic chord of the wing.
The greater this distance and the narrower the wing, the more stable the aircraft.
"""
import openmdao.api as om
import numpy as np


class StaticMargin(om.ExplicitComponent):
    """
    Computes the static margin of a fixed wing UAV
    """

    def setup(self):
        self.add_input("data:stability:neutral_point", val=np.nan, units="m")
        self.add_input("data:stability:CoG", val=np.nan, units="m")
        self.add_input("data:geometry:wing:MAC:length", val=np.nan, units="m")
        self.add_output("data:stability:static_margin", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        x_np = inputs["data:stability:neutral_point"]
        x_cg = inputs["data:stability:CoG"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]

        SM = (x_np - x_cg) / c_MAC  # static margin [-]

        outputs["data:stability:static_margin"] = SM

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        x_np = inputs["data:stability:neutral_point"]
        x_cg = inputs["data:stability:CoG"]
        c_MAC = inputs["data:geometry:wing:MAC:length"]

        partials["data:stability:static_margin",
                 "data:stability:neutral_point"] = 1 / c_MAC
        partials["data:stability:static_margin",
                 "data:stability:CoG"] = - 1 / c_MAC
        partials["data:stability:static_margin",
                 "data:geometry:wing:MAC:length"] = - (x_np - x_cg) / c_MAC ** 2
