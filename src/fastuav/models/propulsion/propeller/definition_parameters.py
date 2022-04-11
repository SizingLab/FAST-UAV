"""
Definition parameters for the propeller.
"""
import openmdao.api as om
import numpy as np
from fastuav.models.uncertainty.uncertainty import add_subsystem_with_deviation


class PropellerDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the propeller.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the propeller are the pitch-to-diameter ratio (-) and the rotational speed per diameter
    for take-off (Hz.m).
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "beta",
            beta(),
            uncertain_outputs={"data:propulsion:propeller:beta:estimated": None},
        )
        self.add_subsystem("takeoff_speed", TakeOffSpeed(), promotes=["*"])


class TakeOffSpeed(om.ExplicitComponent):
    """
    Returns the takeoff propeller rotational speed per diameter for sizing.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:ND:max:reference", val=np.nan, units="m/s")
        self.add_input("data:propulsion:propeller:ND:k", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:ND:takeoff", units="m/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        NDmax = inputs["data:propulsion:propeller:ND:max:reference"]
        k_ND = inputs["data:propulsion:propeller:ND:k"]

        ND = NDmax * k_ND  # [m] Propeller diameter

        outputs["data:propulsion:propeller:ND:takeoff"] = ND

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        NDmax = inputs["data:propulsion:propeller:ND:max:reference"]

        partials["data:propulsion:propeller:ND:takeoff", "data:propulsion:propeller:ND:k"] = NDmax


class beta(om.ExplicitComponent):
    """
    Copy of the beta input.
    Returns the pitch-to-diameter ratio for sizing.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:beta:guess", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:beta:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        outputs["data:propulsion:propeller:beta:estimated"] = inputs[
            "data:propulsion:propeller:beta:guess"
        ]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:propulsion:propeller:beta:estimated",
            "data:propulsion:propeller:beta:guess",
        ] = 1.0
