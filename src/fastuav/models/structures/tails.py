"""
Tails Structures and Weights
"""
import openmdao.api as om
import numpy as np
from fastuav.models.structures.wing.estimation_models import WingStructuresEstimationModels


class HorizontalTailStructures(om.ExplicitComponent):
    """
    Computes Horizontal Tail mass
    """

    def setup(self):
        self.add_input("data:geometry:tail:horizontal:surface", val=np.nan, units="m**2")
        self.add_input("data:weight:airframe:tail:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weight:airframe:tail:horizontal:mass", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_ht = inputs["data:geometry:tail:horizontal:surface"]
        rho_skin = inputs["data:weight:airframe:tail:density"]

        m_skin = WingStructuresEstimationModels.skin(S_ht / 2, rho_skin)  # skin
        m_wing = 2 * m_skin  # total mass (both sides) [kg]

        outputs["data:weight:airframe:tail:horizontal:mass"] = m_wing


class VerticalTailStructures(om.ExplicitComponent):
    """
    Computes Vertical Tail mass
    """

    def setup(self):
        self.add_input("data:geometry:tail:vertical:surface", val=np.nan, units="m**2")
        self.add_input("data:weight:airframe:tail:density", val=np.nan, units="kg/m**2")
        self.add_output("data:weight:airframe:tail:vertical:mass", units="kg", lower=0.0)

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        S_vt = inputs["data:geometry:tail:vertical:surface"]
        rho_skin = inputs["data:weight:airframe:tail:density"]

        m_skin = WingStructuresEstimationModels.skin(S_vt, rho_skin)  # skin
        m_wing = m_skin  # total mass (both sides) [kg]

        outputs["data:weight:airframe:tail:vertical:mass"] = m_wing
