"""
Estimation models for the Electronic Speed Controller (ESC)
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class ESCEstimationModels(om.Group):
    """
    Group containing the estimation models for the Electronic Speed Controller.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weight:propulsion:esc:mass:estimated": "kg"},
        )


class Weight(om.ExplicitComponent):
    """
    Computes ESC weight
    """

    def setup(self):
        self.add_input("data:weight:propulsion:esc:mass:reference", val=np.nan, units="kg")
        self.add_input("data:propulsion:esc:power:reference", val=np.nan, units="W")
        self.add_input("data:propulsion:esc:power:estimated", val=np.nan, units="W")
        self.add_output("data:weight:propulsion:esc:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        m_esc_ref = inputs["data:weight:propulsion:esc:mass:reference"]
        P_esc_ref = inputs["data:propulsion:esc:power:reference"]
        P_esc = inputs["data:propulsion:esc:power:estimated"]

        m_esc = m_esc_ref * (P_esc / P_esc_ref)  # [kg] Mass ESC

        outputs["data:weight:propulsion:esc:mass:estimated"] = m_esc
