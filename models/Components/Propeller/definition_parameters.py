"""
Definition parameters for the propeller.
"""
import openmdao.api as om
import numpy as np
from models.Uncertainty.uncertainty import add_subsystem_with_deviation


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
            uncertain_outputs={"data:propeller:geometry:beta:estimated": None},
        )
        self.add_subsystem("takeoff_speed", SpeedTO(), promotes=["*"])


class SpeedTO(om.ExplicitComponent):
    """
    Computes the propeller rotational speed per diameter for take-off.
    """

    def setup(self):
        self.add_input("data:propeller:reference:ND:max", val=np.nan, units="m/s")
        self.add_input("data:propeller:settings:ND:k", val=np.nan, units=None)
        self.add_output("data:propeller:ND:takeoff", units="m/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        NDmax = inputs["data:propeller:reference:ND:max"]
        k_ND = inputs["data:propeller:settings:ND:k"]

        ND = NDmax * k_ND  # [m] Propeller diameter

        outputs["data:propeller:ND:takeoff"] = ND

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        NDmax = inputs["data:propeller:reference:ND:max"]

        partials["data:propeller:ND:takeoff", "data:propeller:settings:ND:k"] = NDmax


class beta(om.ExplicitComponent):
    """
    Copy of the beta input.
    """

    def setup(self):
        self.add_input("data:propeller:geometry:beta:guess", val=np.nan, units=None)
        self.add_output("data:propeller:geometry:beta:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        outputs["data:propeller:geometry:beta:estimated"] = inputs[
            "data:propeller:geometry:beta:guess"
        ]

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        partials[
            "data:propeller:geometry:beta:estimated",
            "data:propeller:geometry:beta:guess",
        ] = 1.0
