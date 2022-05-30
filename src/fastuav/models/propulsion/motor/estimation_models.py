"""
Estimation models for the motor
"""
import openmdao.api as om
import numpy as np
from fastuav.utils.uncertainty import add_subsystem_with_deviation


class MotorEstimationModels(om.Group):
    """
    Group containing the estimation models for the motor.
    Estimation models take a reduced set of definition parameters and estimate the main component characteristics from it.
    """

    def setup(self):
        add_subsystem_with_deviation(
            self,
            "nominal_torque",
            NominalTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:nominal:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "friction_torque",
            FrictionTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:friction:estimated": "N*m"},
        )

        add_subsystem_with_deviation(
            self,
            "resistance",
            Resistance(),
            uncertain_outputs={"data:propulsion:motor:resistance:estimated": "V/A"},
        )

        add_subsystem_with_deviation(
            self,
            "weight",
            Weight(),
            uncertain_outputs={"data:weights:propulsion:motor:mass:estimated": "kg"},
        )

        self.add_subsystem("geometry", Geometry(), promotes=["*"])


class NominalTorque(om.ExplicitComponent):
    """
    Compute nominal torque
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:nominal:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_nom_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        Tmot_nom = Tmot_nom_ref * Tmot_max / Tmot_max_ref  # [N.m] nominal torque

        outputs["data:propulsion:motor:torque:nominal:estimated"] = Tmot_nom

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_nom_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "data:propulsion:motor:torque:nominal:reference"] = Tmot_max / Tmot_max_ref

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "data:propulsion:motor:torque:max:reference"] = - Tmot_nom_ref * Tmot_max / Tmot_max_ref ** 2

        partials["data:propulsion:motor:torque:nominal:estimated",
                 "data:propulsion:motor:torque:max:estimated"] = Tmot_nom_ref / Tmot_max_ref


class FrictionTorque(om.ExplicitComponent):
    """
    Computes friction torque.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:friction:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:friction:estimated", units="N*m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Tfmot_ref = inputs["data:propulsion:motor:torque:friction:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        Tfmot = Tfmot_ref * (Tmot_max / Tmot_max_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs["data:propulsion:motor:torque:friction:estimated"] = Tfmot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Tfmot_ref = inputs["data:propulsion:motor:torque:friction:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]

        partials["data:propulsion:motor:torque:friction:estimated",
                 "data:propulsion:motor:torque:friction:reference"
        ] = (Tmot_max / Tmot_max_ref) ** (3 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "data:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * Tfmot_ref * Tmot_max ** (3 / 3.5) / Tmot_max_ref ** (6.5 / 3.5)

        partials["data:propulsion:motor:torque:friction:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * Tfmot_ref / Tmot_max_ref ** (3 / 3.5) * Tmot_max ** (- 0.5 / 3.5)


class Resistance(om.ExplicitComponent):
    """
    Computes motor resistance.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:coefficient:estimated", val=np.nan, units="N*m/A")
        self.add_input("data:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance:reference", val=np.nan, units="V/A")
        self.add_input("data:propulsion:motor:torque:coefficient:reference", val=np.nan, units="N*m/A")
        self.add_output("data:propulsion:motor:resistance:estimated", units="V/A")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Rmot_ref = inputs["data:propulsion:motor:resistance:reference"]
        Ktmot_ref = inputs["data:propulsion:motor:torque:coefficient:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient:estimated"]

        Rmot = (
            Rmot_ref * (Tmot_max / Tmot_max_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2
        )  # [Ohm] motor resistance

        outputs["data:propulsion:motor:resistance:estimated"] = Rmot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Rmot_ref = inputs["data:propulsion:motor:resistance:reference"]
        Ktmot_ref = inputs["data:propulsion:motor:torque:coefficient:reference"]
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient:estimated"]

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:max:reference"
        ] = (5 / 3.5) * Rmot_ref * Tmot_max ** (-5 / 3.5) * Tmot_max_ref ** (1.5 / 3.5) * (Ktmot / Ktmot_ref) ** 2

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:resistance:reference"
        ] = (Tmot_max / Tmot_max_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:coefficient:reference"
        ] = -2 * Rmot_ref * (Tmot_max / Tmot_max_ref) ** (-5 / 3.5) * Ktmot ** 2 / Ktmot_ref ** 3

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (-5 / 3.5) * Rmot_ref / Tmot_max_ref ** (-5 / 3.5) * Tmot_max ** (-8.5 / 3.5) * (Ktmot / Ktmot_ref) ** 2

        partials["data:propulsion:motor:resistance:estimated",
                 "data:propulsion:motor:torque:coefficient:estimated"
        ] = 2 * Rmot_ref * (Tmot_max / Tmot_max_ref) ** (-5 / 3.5) * Ktmot / Ktmot_ref ** 2


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:max:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_input("data:weights:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weights:propulsion:motor:mass:estimated", units="kg")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]

        Mmot = Mmot_ref * (Tmot_max / Tmot_max_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs["data:weights:propulsion:motor:mass:estimated"] = Mmot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Tmot_max = inputs["data:propulsion:motor:torque:max:estimated"]
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]

        partials["data:weights:propulsion:motor:mass:estimated",
                 "data:propulsion:motor:torque:max:estimated"
        ] = (3 / 3.5) * Mmot_ref / Tmot_max_ref ** (3 / 3.5) * Tmot_max ** (-0.5 / 3.5)

        partials["data:weights:propulsion:motor:mass:estimated",
                 "data:propulsion:motor:torque:max:reference"
        ] = - (3 / 3.5) * Mmot_ref * Tmot_max ** (3 / 3.5) / Tmot_max_ref ** (6.5 / 3.5)

        partials["data:weights:propulsion:motor:mass:estimated",
                 "data:weights:propulsion:motor:mass:reference"
        ] = (Tmot_max / Tmot_max_ref) ** (3 / 3.5)


class Geometry(om.ExplicitComponent):
    """
    Computes motor geometry
    """

    def setup(self):
        self.add_input("data:propulsion:motor:length:reference", val=np.nan, units="m")
        self.add_input("data:weights:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_input("data:weights:propulsion:motor:mass:estimated", val=np.nan, units="kg")
        self.add_output("data:propulsion:motor:length:estimated", units="m")

    def setup_partials(self):
        self.declare_partials("*", "*", method="exact")

    def compute(self, inputs, outputs):
        Lmot_ref = inputs["data:propulsion:motor:length:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]
        Mmot = inputs["data:weights:propulsion:motor:mass:estimated"]

        Lmot = Lmot_ref * (Mmot / Mmot_ref) ** (1 / 3)  # [m] Motor length (estimated)

        outputs["data:propulsion:motor:length:estimated"] = Lmot

    def compute_partials(self, inputs, partials, discrete_inputs=None):
        Lmot_ref = inputs["data:propulsion:motor:length:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]
        Mmot = inputs["data:weights:propulsion:motor:mass:estimated"]

        partials["data:propulsion:motor:length:estimated",
                 "data:propulsion:motor:length:reference"
        ] = (Mmot / Mmot_ref) ** (1 / 3)

        partials["data:propulsion:motor:length:estimated",
                 "data:weights:propulsion:motor:mass:estimated"
        ] = (1 / 3) * Lmot_ref / Mmot_ref ** (1 / 3) * Mmot ** (- 2 / 3)

        partials["data:propulsion:motor:length:estimated",
                 "data:weights:propulsion:motor:mass:reference"
        ] = - (1 / 3) * Lmot_ref * Mmot ** (1 / 3) / Mmot_ref ** (4 / 3)

