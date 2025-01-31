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
            uncertain_outputs={"data:propulsion:propeller:Ct:static:polynomial:estimated": None,
                               "data:propulsion:propeller:Cp:static:polynomial:estimated": None,
                               "data:propulsion:propeller:Ct:dynamic:polynomial:estimated": None,
                               "data:propulsion:propeller:Cp:dynamic:polynomial:estimated": None
                               },
        )


class TakeOffSpeed(om.ExplicitComponent):
    """
    Returns the takeoff propeller rotational speed per diameter for sizing.
    """
    def setup(self):
        self.add_input("models:propulsion:propeller:ND:max:reference", val=np.nan, units="m/s")
        self.add_input("optimization:variables:propulsion:propeller:ND:k", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:ND:takeoff", units="m/s")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        NDmax = inputs["models:propulsion:propeller:ND:max:reference"]
        k_ND = inputs["optimization:variables:propulsion:propeller:ND:k"]

        ND = NDmax * k_ND  # [m] Propeller diameter

        outputs["data:propulsion:propeller:ND:takeoff"] = ND

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        NDmax = inputs["models:propulsion:propeller:ND:max:reference"]
        k_ND = inputs["optimization:variables:propulsion:propeller:ND:k"]

        partials["data:propulsion:propeller:ND:takeoff",
                 "optimization:variables:propulsion:propeller:ND:k"] = NDmax

        partials["data:propulsion:propeller:ND:takeoff",
                 "models:propulsion:propeller:ND:max:reference"] = k_ND


class Beta(om.ExplicitComponent):
    """
    Copy of the beta (pitch-to-diameter) input.
    "beta" is the initial guess of the pitch-to-diameter, while "beta:estimated" is an estimated value,
    which can be different, e.g. if an uncertainty is added to this variable (see add_subsystem_with_deviation func)
    """
    def setup(self):
        self.add_input("optimization:variables:propulsion:propeller:beta", val=np.nan, units=None)
        self.add_output("data:propulsion:propeller:beta:estimated", units=None)

    def setup_partials(self):
        self.declare_partials("data:propulsion:propeller:beta:estimated",
                              "optimization:variables:propulsion:propeller:beta",
                              val=1.0)

    def compute(self, inputs, outputs):
        outputs["data:propulsion:propeller:beta:estimated"] = inputs[
            "optimization:variables:propulsion:propeller:beta"
        ]


class AerodynamicsModelParameters(om.ExplicitComponent):
    """
    Sets aerodynamic model parameters for future calculation of thrust and power coefficients.
    """

    def setup(self):
        self.add_input("models:propulsion:propeller:Ct:static:polynomial",
                       val=np.array([4.27e-02, 1.44e-01]),
                       units=None)
        self.add_input("models:propulsion:propeller:Cp:static:polynomial",
                       val=np.array([-1.48e-03, 9.72e-02]),
                       units=None)
        self.add_input("models:propulsion:propeller:Ct:dynamic:polynomial",
                       val=np.array([0.02791, 0.11867, 0.27334, - 0.28852, - 0.06543, - 0.23504, 0.02104, 0.0, 0.0, 0.18677,
                                     0.197, 1.094]),
                       units=None)
        self.add_input("models:propulsion:propeller:Cp:dynamic:polynomial",
                       val=np.array([0.01813, - 0.06218, 0.35712, - 0.23774, 0.00343, - 0.1235, 0.0, 0.07549, 0.0, 0.0,
                                     0.286, 0.993]),
                       units=None)
        self.add_output("data:propulsion:propeller:Ct:static:polynomial:estimated",
                        copy_shape="models:propulsion:propeller:Ct:static:polynomial",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:static:polynomial:estimated",
                        copy_shape="models:propulsion:propeller:Cp:static:polynomial",
                        units=None)
        self.add_output("data:propulsion:propeller:Ct:dynamic:polynomial:estimated",
                        copy_shape="models:propulsion:propeller:Ct:dynamic:polynomial",
                        units=None)
        self.add_output("data:propulsion:propeller:Cp:dynamic:polynomial:estimated",
                        copy_shape="models:propulsion:propeller:Cp:dynamic:polynomial",
                        units=None)

    def setup_partials(self):
        self.declare_partials(
            "data:propulsion:propeller:Ct:static:polynomial:estimated",
            "models:propulsion:propeller:Ct:static:polynomial",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:static:polynomial:estimated",
            "models:propulsion:propeller:Cp:static:polynomial",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Ct:dynamic:polynomial:estimated",
            "models:propulsion:propeller:Ct:dynamic:polynomial",
            val=1.0
        )
        self.declare_partials(
            "data:propulsion:propeller:Cp:dynamic:polynomial:estimated",
            "models:propulsion:propeller:Cp:dynamic:polynomial",
            val=1.0
        )

    def compute(self, inputs, outputs):
        outputs["data:propulsion:propeller:Ct:static:polynomial:estimated"] = \
            inputs["models:propulsion:propeller:Ct:static:polynomial"]
        outputs["data:propulsion:propeller:Cp:static:polynomial:estimated"] = \
            inputs["models:propulsion:propeller:Cp:static:polynomial"]
        outputs["data:propulsion:propeller:Ct:dynamic:polynomial:estimated"] = \
            inputs["models:propulsion:propeller:Ct:dynamic:polynomial"]
        outputs["data:propulsion:propeller:Cp:dynamic:polynomial:estimated"] = \
            inputs["models:propulsion:propeller:Cp:dynamic:polynomial"]
