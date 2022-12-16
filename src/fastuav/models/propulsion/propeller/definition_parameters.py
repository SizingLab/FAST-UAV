"""
Definition parameters for the propeller.
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class PropellerDefinitionParameters(om.Group):
    """
    Group containing the calculation of the definition parameters for the propeller.
    The definition parameters are independent variables that allow to derive all the other component's parameters,
    by using datasheets or estimation models.
    The definition parameters for the propeller are the pitch-to-diameter ratio (-) and the rotational speed per
    diameter for take-off (Hz.m).
    """
    def setup(self):
        add_subsystem_with_deviation(
            self,
            "beta",
            Beta(),
            uncertain_outputs={"data:propulsion:propeller:beta:estimated": None},
        )
        self.add_subsystem("takeoff_speed",
                           TakeOffSpeed(),
                           promotes=["*"])

        add_subsystem_with_deviation(
            self,
            "aero_coefficients",
            AerodynamicsModelParameters(),
            uncertain_outputs={"data:propulsion:propeller:Ct:model:static:estimated": None,
                               "data:propulsion:propeller:Cp:model:static:estimated": None,
                               "data:propulsion:propeller:Ct:model:dynamic:estimated": None,
                               "data:propulsion:propeller:Cp:model:dynamic:estimated": None
                               },
        )


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
        k_ND = inputs["data:propulsion:propeller:ND:k"]

        partials["data:propulsion:propeller:ND:takeoff",
                 "data:propulsion:propeller:ND:k"] = NDmax

        partials["data:propulsion:propeller:ND:takeoff",
                 "data:propulsion:propeller:ND:max:reference"] = k_ND


class Beta(om.ExplicitComponent):
    """
    Copy of the beta (pitch-to-diameter) input.
    "beta:guess" is the initial guess of the pitch-to-diameter, while "beta:estimated" is an estimated value,
    which can be different, e.g. if an uncertainty is added to this variable (see add_subsystem_with_deviation func)
    """
    def setup(self):
        self.add_input("data:propulsion:propeller:beta:guess", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:beta:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("data:propulsion:propeller:beta:estimated",
                              "data:propulsion:propeller:beta:guess",
                              val=1.0)

    def compute(self, inputs, outputs):
        outputs["data:propulsion:propeller:beta:estimated"] = inputs[
            "data:propulsion:propeller:beta:guess"
        ]


class AerodynamicsModelParameters(om.ExplicitComponent):
    """
    Sets aerodynamic model parameters for future calculation of thrust and power coefficients.
    """

    def setup(self):
        self.add_input("data:propulsion:propeller:Ct:model:static:reference",
                       val=np.array([4.27e-02, 1.44e-01]),
                       units=None)
        self.add_input("data:propulsion:propeller:Cp:model:static:reference",
                       val=np.array([-1.48e-03, 9.72e-02]),
                       units=None)
        self.add_input("data:propulsion:propeller:Ct:model:dynamic:reference",
                       val=np.array([0.02791, 0.11867, 0.27334, - 0.28852, - 0.06543, - 0.23504, 0.02104, 0.0, 0.0, 0.18677,
                                     0.197, 1.094]),
                       units=None)
        self.add_input("data:propulsion:propeller:Cp:model:dynamic:reference",
                       val=np.array([0.01813, - 0.06218, 0.35712, - 0.23774, 0.00343, - 0.1235, 0.0, 0.07549, 0.0, 0.0,
                                     0.286, 0.993]),
                       units=None)
        self.add_output("data:propulsion:propeller:Ct:model:static:estimated",
                        copy_shape="data:propulsion:propeller:Ct:model:static:reference",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:model:static:estimated",
                        copy_shape="data:propulsion:propeller:Cp:model:static:reference",
                        units=None)
        self.add_output("data:propulsion:propeller:Ct:model:dynamic:estimated",
                        copy_shape="data:propulsion:propeller:Ct:model:dynamic:reference",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:model:dynamic:estimated",
                        copy_shape="data:propulsion:propeller:Cp:model:dynamic:reference",
                        units=None)

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:propeller:Ct:model:static:estimated",
            "data:propulsion:propeller:Ct:model:static:reference",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:model:static:estimated",
            "data:propulsion:propeller:Cp:model:static:reference",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Ct:model:dynamic:estimated",
            "data:propulsion:propeller:Ct:model:dynamic:reference",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:model:dynamic:estimated",
            "data:propulsion:propeller:Cp:model:dynamic:reference",
            val=1.0
        )

    def compute(self, inputs, outputs):
        outputs["data:propulsion:propeller:Ct:model:static:estimated"] = \
            inputs["data:propulsion:propeller:Ct:model:static:reference"]
        outputs["data:propulsion:propeller:Cp:model:static:estimated"] = \
            inputs["data:propulsion:propeller:Cp:model:static:reference"]
        outputs["data:propulsion:propeller:Ct:model:dynamic:estimated"] = \
            inputs["data:propulsion:propeller:Ct:model:dynamic:reference"]
        outputs["data:propulsion:propeller:Cp:model:dynamic:estimated"] = \
            inputs["data:propulsion:propeller:Cp:model:dynamic:reference"]
