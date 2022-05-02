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
            "max_torque",
            MaxTorque(),
            uncertain_outputs={"data:propulsion:motor:torque:max:estimated": "N*m"},
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


class MaxTorque(om.ExplicitComponent):
    """
    Compute maximum torque
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:max:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:max:estimated", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Tmot_max_ref = inputs["data:propulsion:motor:torque:max:reference"]
        Tmot = inputs["data:propulsion:motor:torque:nominal:estimated"]

        Tmot_max = Tmot_max_ref * (Tmot / Tmot_ref)  # [N.m] max torque

        outputs["data:propulsion:motor:torque:max:estimated"] = Tmot_max


class FrictionTorque(om.ExplicitComponent):
    """
    Computes friction torque.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:friction:reference", val=np.nan, units="N*m")
        self.add_output("data:propulsion:motor:torque:friction:estimated", units="N*m")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Tfmot_ref = inputs["data:propulsion:motor:torque:friction:reference"]
        Tmot = inputs["data:propulsion:motor:torque:nominal:estimated"]

        Tfmot = Tfmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [N.m] Friction torque

        outputs["data:propulsion:motor:torque:friction:estimated"] = Tfmot


class Resistance(om.ExplicitComponent):
    """
    Computes motor resistance.
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input(
            "data:propulsion:motor:torque:coefficient:estimated", val=np.nan, units="N*m/A"
        )
        self.add_input("data:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:resistance:reference", val=np.nan, units="V/A")
        self.add_input(
            "data:propulsion:motor:torque:coefficient:reference", val=np.nan, units="N*m/A"
        )
        self.add_output("data:propulsion:motor:resistance:estimated", units="V/A")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Rmot_ref = inputs["data:propulsion:motor:resistance:reference"]
        Ktmot_ref = inputs["data:propulsion:motor:torque:coefficient:reference"]
        Tmot = inputs["data:propulsion:motor:torque:nominal:estimated"]
        Ktmot = inputs["data:propulsion:motor:torque:coefficient:estimated"]

        Rmot = (
            Rmot_ref * (Tmot / Tmot_ref) ** (-5 / 3.5) * (Ktmot / Ktmot_ref) ** 2
        )  # [Ohm] motor resistance

        outputs["data:propulsion:motor:resistance:estimated"] = Rmot


class Weight(om.ExplicitComponent):
    """
    Weight calculation of an electrical Motor
    """

    def setup(self):
        self.add_input("data:propulsion:motor:torque:nominal:estimated", val=np.nan, units="N*m")
        self.add_input("data:propulsion:motor:torque:nominal:reference", val=np.nan, units="N*m")
        self.add_input("data:weights:propulsion:motor:mass:reference", val=np.nan, units="kg")
        self.add_output("data:weights:propulsion:motor:mass:estimated", units="kg")

    def setup_partials(self):
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Tmot = inputs["data:propulsion:motor:torque:nominal:estimated"]
        Tmot_ref = inputs["data:propulsion:motor:torque:nominal:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]

        Mmot = Mmot_ref * (Tmot / Tmot_ref) ** (3 / 3.5)  # [kg] Motor mass (estimated)

        outputs["data:weights:propulsion:motor:mass:estimated"] = Mmot


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
        # Finite difference all partials.
        self.declare_partials("*", "*", method="fd")

    def compute(self, inputs, outputs):
        Lmot_ref = inputs["data:propulsion:motor:length:reference"]
        Mmot_ref = inputs["data:weights:propulsion:motor:mass:reference"]
        Mmot = inputs["data:weights:propulsion:motor:mass:estimated"]

        Lmot = Lmot_ref * (Mmot / Mmot_ref) ** (1 / 3)  # [m] Motor length (estimated)

        outputs["data:propulsion:motor:length:estimated"] = Lmot
